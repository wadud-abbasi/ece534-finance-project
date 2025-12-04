# dataset.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, Any


class StockNewsDataset(Dataset):
    def __init__(
        self,
        labeled_path: str,
        prices_path: str,
        finbert_model_name: str = "ProsusAI/finbert",
        seq_len: int = 10,
        max_length: int = 64,
        min_abs_ret: float | None = None,
    ):
        """
        labeled_path: parquet/csv from merge_df.py (contains Date, date, ticker, Article_title, ret, etc.)
        prices_path: full processed_stock_prices.csv (date, ticker, open, high, low, close, adj_close, volume, ...)
        seq_len: number of days of price history (including current trading day)
        min_abs_ret: if set, can filter by |ret| >= threshold for stronger signals
        """
        super().__init__()

        # --- Load labeled data ---
        if labeled_path.endswith(".parquet"):
            df = pd.read_parquet(labeled_path)
        else:
            df = pd.read_csv(labeled_path)

        # ensure columns
        required_cols = ["date", "ticker", "Article_title", "ret"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Labeled dataset missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "ticker", "Article_title", "ret"])
        if min_abs_ret is not None:
            df = df[df["ret"].abs() >= min_abs_ret]

        df = df.reset_index(drop=True)
        self.data = df

        # --- Load full price history ---
        prices_df = pd.read_csv(prices_path, low_memory=False)
        prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce")
        prices_df = prices_df.dropna(subset=["date", "ticker"])
        prices_df = prices_df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # ensure numeric price cols
        price_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        for col in price_cols:
            if col not in prices_df.columns:
                raise ValueError(f"prices_df missing required column '{col}'")

        # build per-ticker history
        self.seq_len = seq_len
        self.price_cols = price_cols
        self.price_history: Dict[str, Dict[str, Any]] = {}

        for ticker, group in prices_df.groupby("ticker"):
            dates = group["date"].values  # datetime64
            feats = group[price_cols].values.astype(np.float32)  # (N, F)
            # also store as int days for search
            date_ints = dates.astype("datetime64[D]").astype(np.int64)
            self.price_history[ticker] = {
                "dates": dates,
                "date_ints": date_ints,
                "feats": feats,
            }

        # --- FinBERT tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def _get_price_window(self, ticker: str, date) -> np.ndarray | None:
        """
        Get seq_len days of [open, high, low, close, adj_close, volume]
        up to and including 'date' for this ticker.
        Returns (seq_len, num_features) or None if insufficient data.
        """
        if ticker not in self.price_history:
            return None

        hist = self.price_history[ticker]
        date_ints = hist["date_ints"]
        feats = hist["feats"]
        target_int = np.int64(date.astype("datetime64[D]").astype(np.int64))

        # index of last date <= target_int
        idx = np.searchsorted(date_ints, target_int, side="right") - 1
        if idx < 0:
            return None

        start = idx - (self.seq_len - 1)
        if start < 0:
            # left-pad with earliest row
            pad_rows = feats[0:1].repeat(-start, axis=0)
            window = feats[0 : idx + 1]
            window = np.concatenate([pad_rows, window], axis=0)
        else:
            window = feats[start : idx + 1]

        if window.shape[0] != self.seq_len:
            # safety
            if window.shape[0] < self.seq_len:
                pad_rows = window[0:1].repeat(self.seq_len - window.shape[0], axis=0)
                window = np.concatenate([pad_rows, window], axis=0)
            else:
                window = window[-self.seq_len :]

        return window.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        headline = str(row["Article_title"])
        ticker = str(row["ticker"])
        trade_date = row["date"]
        target = float(row["ret"])

        # price window
        price_window = self._get_price_window(ticker, trade_date)
        if price_window is None:
            # extremely rare fallback: just zero
            price_window = np.zeros((self.seq_len, len(self.price_cols)), dtype=np.float32)

        # tokenize text
        encoded = self.tokenizer(
            headline,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "price_seq": torch.from_numpy(price_window),  # (seq_len, num_features)
            "target": torch.tensor(target, dtype=torch.float32),
        }
        return item
