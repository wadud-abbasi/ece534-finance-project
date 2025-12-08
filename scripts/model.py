#!/usr/bin/env python
"""
model.py

Sequence dataset + 2-layer LSTM + LayerNorm + MLP for predicting next-day
log returns (target_ret1) from premerged daily features.

The merged dataset (from merge.py) is expected to contain at least:
  - date, ticker
  - ret0
  - target_ret1
  - emb_*             (FinBERT embeddings)
  - sent_neg, sent_neu, sent_pos (optional)
  - ret_djia, ret_nasdaqcom, ret_sp500 (optional)
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn


@dataclass
class SequenceConfig:
    lookback: int = 30
    target_col: str = "target_ret1"


def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Infer which columns should be used as input features per time step.
    We take:
      - all columns starting with 'emb_'
      - ret0
      - sent_neg, sent_neu, sent_pos if present
      - ret_djia, ret_nasdaqcom, ret_sp500 if present
    """
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    base_cols = ["ret0"]
    snt_cols = [c for c in ["sent_neg", "sent_neu", "sent_pos"] if c in df.columns]
    idx_cols = [c for c in ["ret_djia", "ret_nasdaqcom", "ret_sp500"] if c in df.columns]

    feat_cols = emb_cols + base_cols + snt_cols + idx_cols
    return feat_cols


class NewsPriceSequenceDataset(Dataset):
    """
    Builds rolling sequences for each ticker from a merged daily dataframe.

    For each ticker, with rows in chronological order, we create samples:

      X: [lookback, input_dim]  (features over last `lookback` days)
      y: scalar target (target_ret1 for the last day in the window)

    A sample is only created if:
      - there are at least `lookback` days of history for that ticker
      - target_col is not NaN on the last day
    """

    def __init__(self, df: pd.DataFrame, cfg: SequenceConfig):
        super().__init__()
        self.cfg = cfg

        # Ensure sorted order by ticker, date
        self.df = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()

        if "ticker" not in self.df.columns or "date" not in self.df.columns:
            raise ValueError("DataFrame must contain 'ticker' and 'date' columns.")

        if cfg.target_col not in self.df.columns:
            raise ValueError(f"DataFrame is missing target column '{cfg.target_col}'.")

        # Infer feature columns
        self.feature_cols = infer_feature_columns(self.df)
        if not self.feature_cols:
            raise ValueError("No feature columns found (emb_*, ret0, sent_*, index returns).")

        # Build sample index: list of (sequence_row_indices, target_value)
        self.samples: List[Tuple[np.ndarray, float]] = []
        lb = cfg.lookback

        for ticker, grp in self.df.groupby("ticker"):
            idxs = grp.index.to_numpy()
            # grp is already in chronological order because df is sorted
            for pos in range(lb - 1, len(idxs)):
                row_idx = idxs[pos]
                y = self.df.loc[row_idx, cfg.target_col]
                if pd.isna(y):
                    continue
                seq_idxs = idxs[pos - lb + 1: pos + 1]  # inclusive window
                self.samples.append((seq_idxs, float(y)))

        if len(self.samples) == 0:
            raise RuntimeError("No valid sequences constructed. "
                               "Check lookback length and data coverage.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        seq_idxs, y = self.samples[i]
        # [lookback, input_dim]
        X = self.df.loc[seq_idxs, self.feature_cols].to_numpy(dtype=np.float32)
        X_tensor = torch.from_numpy(X)              # (L, D)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return X_tensor, y_tensor

    @property
    def input_dim(self) -> int:
        return len(self.feature_cols)


class LSTMRegressorWithLN(nn.Module):
    """
    2-layer LSTM + LayerNorm + 2-layer MLP head for regression.

    Input:  (batch, lookback, input_dim)
    Output: (batch,)  scalar regression target (e.g., next-day log return)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # LayerNorm on the final hidden state
        self.ln = nn.LayerNorm(hidden_dim)

        # 2-layer MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, lookback, input_dim)
        returns: (batch,)
        """
        out, _ = self.lstm(x)             # out: (batch, lookback, hidden_dim)
        h_last = out[:, -1, :]            # last time step for each sequence
        h_norm = self.ln(h_last)          # (batch, hidden_dim)
        y_hat = self.mlp(h_norm).squeeze(-1)  # (batch,)
        return y_hat
