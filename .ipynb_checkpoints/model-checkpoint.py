"""
model.py — FinBERT → daily sentiment/embeddings → LSTM/TFT

Compatible with the Data/data dataset format:
- processed_stock_prices.csv  (date, open.., close, adj_close, ticker)
- processed_headlines_subset.csv (Date, Article_title, Stock_symbol)
- processed_indexes.csv (date, djia, nasdaqcom, sp500)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ------------------------------
# FinBERT embedder
# ------------------------------
class FinBertEmbedder(nn.Module):
    """Batch-embed texts with pretrained FinBERT. Frozen by default."""
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        pooling: str = "cls",            # "cls" or "mean"
        unfreeze_top_layers: int = 0,    # 0 keeps encoder frozen
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(self.device)

        # freeze encoder
        for p in self.model.base_model.parameters():
            p.requires_grad = False

        if unfreeze_top_layers > 0:
            enc = self.model.base_model.encoder
            layers = getattr(enc, "layer", getattr(enc, "layers", None))
            for lyr in layers[-unfreeze_top_layers:]:
                for p in lyr.parameters():
                    p.requires_grad = True

    @torch.inference_mode()
    def encode(self, texts: List[str], max_len: int = 128, batch_size: int = 64, amp: bool = True):
        """Return (pooled_embeddings [N,H], sentiment_probs [N,3])."""
        H = self.model.config.hidden_size
        pooled_list, prob_list = [], []

        dtype = torch.bfloat16 if (amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            with torch.cuda.amp.autocast(enabled=amp and self.device.type == "cuda", dtype=dtype):
                out = self.model(**toks)
                last = out.hidden_states[-1]                  # [B,T,H]
                if self.pooling == "mean":
                    mask = toks["attention_mask"].unsqueeze(-1)
                    pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
                else:
                    pooled = last[:, 0, :]                    # [B,H]

                pooled_list.append(pooled.detach().float().cpu())
                prob_list.append(out.logits.softmax(-1).detach().float().cpu())

        pooled_all = torch.cat(pooled_list, dim=0).numpy()
        probs_all = torch.cat(prob_list, dim=0).numpy()
        return pooled_all, probs_all


# ------------------------------
# Sequence dataset
# ------------------------------
@dataclass
class SequenceConfig:
    lookback: int = 30
    horizon: int = 1
    target_col: str = "target_ret1"
    group_col: str = "ticker"
    time_col: str = "date"
    feature_cols: Optional[List[str]] = None  # by default: emb_*, ret0, idx rets, sent_*


class NewsPriceSequenceDataset(torch.utils.data.Dataset):
    """Rolling windows of [lookback] → predict [horizon]."""
    def __init__(self, df: pd.DataFrame, cfg: SequenceConfig):
        self.cfg = cfg
        assert cfg.target_col in df, f"Missing {cfg.target_col}"

        if cfg.feature_cols is None:
            emb = [c for c in df.columns if c.startswith("emb_")]
            extras = [c for c in ["ret0", "ret_djia", "ret_nasdaqcom", "ret_sp500", "sent_pos", "sent_neu", "sent_neg"] if c in df]
            self.feature_cols = emb + extras
        else:
            self.feature_cols = cfg.feature_cols

        self.samples = []
        for _, g in df.sort_values([cfg.group_col, cfg.time_col]).groupby(cfg.group_col):
            g = g.reset_index(drop=True)
            X = g[self.feature_cols].astype("float32").values
            y = g[cfg.target_col].astype("float32").values
            L, H = cfg.lookback, cfg.horizon
            T = len(g)
            for t in range(L, T - H + 1):
                self.samples.append((X[t - L:t], y[t:t + H]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.from_numpy(x), torch.from_numpy(y)


# ------------------------------
# LSTM regressor
# ------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2, horizon: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x):             # x [B,L,F]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)        # [B,H]
