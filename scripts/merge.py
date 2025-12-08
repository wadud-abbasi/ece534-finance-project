#!/usr/bin/env python
"""
merge.py

Build a merged daily dataset with:
- Stock price returns (ret0) and next-day return (target_ret1)
- Daily FinBERT (fine-tuned) embeddings + sentiment scores
- Index returns (ret_djia, ret_nasdaqcom, ret_sp500)

Now supports:
- Single-process mode (default)
- Multi-process / multi-GPU embedding via torch.distributed (NCCL) with --dist_mode ddp

Benchmark timings (in seconds) are printed on rank 0:
- load_data
- finbert_init
- embed_local (this rank)
- embed_global_max (max over all ranks)
- merge
- total

Example (single GPU):

  python merge.py \
    --headlines_csv Data/data/processed_headlines_subset.csv \
    --prices_csv    Data/data/processed_stock_prices.csv \
    --indexes_csv   Data/data/processed_indexes.csv \
    --finetuned_weights models/model.safetensor \
    --output_path   Data/data/merged_lstm_dataset.parquet \
    --max_len 128 \
    --batch_size 256 \
    --amp

Example (4 GPUs on one node):

  torchrun --nproc_per_node=4 merge.py \
    --dist_mode ddp \
    --headlines_csv Data/data/processed_headlines_subset.csv \
    --prices_csv    Data/data/processed_stock_prices.csv \
    --indexes_csv   Data/data/processed_indexes.csv \
    --finetuned_weights models/model.safetensor \
    --output_path   Data/data/merged_lstm_dataset.parquet \
    --max_len 128 \
    --batch_size 256 \
    --amp
"""

import argparse
import os
import sys
import types
import importlib.machinery
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- Environment tweaks so Transformers ignores torchvision/image stack ----
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["DISABLE_TRANSFORMERS_AV"] = "1"

# Torchvision stub (same trick as in your original model.py)
torchvision_stub = types.ModuleType("torchvision")
torchvision_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision", None)

transforms_stub = types.ModuleType("torchvision.transforms")
transforms_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", None)

class _DummyInterpolationMode:
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2
    LANCZOS = 3
    BOX = 4
    HAMMING = 5

transforms_stub.InterpolationMode = _DummyInterpolationMode

io_stub = types.ModuleType("torchvision.io")
io_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", None)

torchvision_stub.transforms = transforms_stub
torchvision_stub.io = io_stub

sys.modules["torchvision"] = torchvision_stub
sys.modules["torchvision.transforms"] = transforms_stub
sys.modules["torchvision.io"] = io_stub


# ----------------- utilities -----------------

def set_seed(s: int = 1337):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def ddp_setup(dist_mode: str):
    """
    Initialize torch.distributed if dist_mode == 'ddp' and env vars are set.

    Returns:
        is_distributed: bool
        rank: int
        world_size: int
        is_main: bool
        device: torch.device
    """
    if dist_mode == "ddp" and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(
            os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count()))
        )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = rank == 0
        return True, rank, world_size, is_main, device

    # Fallback: single-process, single device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, True, device


# ----------------- data loaders -----------------

def load_prices(path: str) -> pd.DataFrame:
    """
    Load price data and compute:
      - log_close
      - ret0: same-day log return
      - target_ret1: next-day log return (regression target)
    """
    df = pd.read_csv(path)
    # Expected columns: date, volume, open, high, low, close, adj_close, ticker
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "ticker"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["log_close"] = np.log(df["close"].clip(1e-6))
    df["ret0"] = df.groupby("ticker")["log_close"].diff().fillna(0.0)
    df["target_ret1"] = df.groupby("ticker")["ret0"].shift(-1)

    return df


def load_headlines(path: str) -> pd.DataFrame:
    """
    Load headlines and normalize to date + ticker + text.

    Input columns (from your sample):
      Date, Article_title, Stock_symbol
    """
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Date": "date",
            "Article_title": "headline",
            "Stock_symbol": "ticker",
        }
    )

    # Handle " ... UTC" and make timezone-aware, then strip tz and normalize date
    raw = df["date"].astype(str).str.replace(" UTC", "", regex=False)
    dt = pd.to_datetime(raw, errors="coerce", utc=True)
    dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)  # drop tz info
    df["date"] = dt.dt.normalize()

    df = df.dropna(subset=["date", "headline", "ticker"])
    return df[["date", "ticker", "headline"]]


def load_indexes(path: str) -> pd.DataFrame:
    """
    Load daily indexes and compute log returns.
    Columns: date, djia, nasdaqcom, sp500
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    # Forward-fill raw index values
    for c in ["djia", "nasdaqcom", "sp500"]:
        if c in df.columns:
            df[c] = df[c].ffill()

    # Log returns as features
    for c in ["djia", "nasdaqcom", "sp500"]:
        if c in df.columns:
            df[f"ret_{c}"] = np.log(df[c].clip(1e-6)).diff().fillna(0.0)

    return df[["date", "ret_djia", "ret_nasdaqcom", "ret_sp500"]]


# ----------------- FinBERT embedder -----------------

class FinBertEmbedder:
    """
    Wraps a (fine-tuned) FinBERT model to produce:
      - CLS/pooled embeddings
      - sentiment probabilities (3-way softmax)

    Assumes the fine-tuned weights are compatible with ProsusAI/finbert.
    """

    def __init__(
        self,
        base_model_name: str = "ProsusAI/finbert",
        finetuned_weights: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model_name)

        # Optionally load a safetensors state dict with fine-tuned weights
        if finetuned_weights is not None and os.path.exists(finetuned_weights):
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise RuntimeError(
                    "safetensors is required to load finetuned weights. "
                    "Install with: pip install safetensors"
                ) from e

            print(
                f"[FinBERT] Loading fine-tuned weights from {finetuned_weights} ...",
                flush=True,
            )
            state_dict = load_file(finetuned_weights)
            # strict=False to tolerate minor key mismatches
            self.model.load_state_dict(state_dict, strict=False)
        elif finetuned_weights:
            print(
                f"[FinBERT] WARNING: finetuned_weights path {finetuned_weights} not found. "
                f"Using base model {base_model_name} instead.",
                flush=True,
            )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        max_len: int = 128,
        batch_size: int = 32,
        amp: bool = True,
        rank: int = 0,
    ):
        """
        Encode a list of texts into:
          - pooled embeddings: shape (N, H)
          - sentiment probs:   shape (N, 3)
        """
        all_pooled = []
        all_probs = []

        total = len(texts)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            ctx = autocast(enabled=amp and self.device.type == "cuda")
            with ctx:
                outputs = self.model(**enc)

            # Use pooler_output if available, otherwise CLS token from last_hidden_state
            pooled = outputs.pooler_output
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0, :]

            # Sentiment probabilities from logits
            probs = torch.softmax(outputs.logits, dim=-1)

            all_pooled.append(pooled.cpu())
            all_probs.append(probs.cpu())

            # Occasional progress print (per rank)
            if (start // batch_size) % 50 == 0:
                print(
                    f"[Rank {rank}] Encoded {end}/{total} headlines...",
                    flush=True,
                )

        if not all_pooled:
            # No data on this rank
            return (
                np.zeros((0, self.model.config.hidden_size), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

        pooled_arr = torch.cat(all_pooled, dim=0).numpy()
        probs_arr = torch.cat(all_probs, dim=0).numpy()
        return pooled_arr, probs_arr


def compute_daily_news_embeddings(
    news_df: pd.DataFrame,
    fb: FinBertEmbedder,
    max_len: int = 128,
    batch_size: int = 64,
    amp: bool = True,
    rank: int = 0,
) -> pd.DataFrame:
    """
    Embed each headline with FinBERT and average per (date, ticker).
    Returns DataFrame:
      date, ticker, emb_0...emb_{H-1}, sent_neg, sent_neu, sent_pos
    """
    if news_df.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    texts = news_df["headline"].astype(str).tolist()
    pooled, probs = fb.encode(
        texts,
        max_len=max_len,
        batch_size=batch_size,
        amp=amp,
        rank=rank,
    )

    H = pooled.shape[1]
    emb_cols = [f"emb_{i}" for i in range(H)]
    emb_df = pd.DataFrame(pooled, columns=emb_cols)

    snt_df = pd.DataFrame(probs, columns=["sent_neg", "sent_neu", "sent_pos"])

    out = pd.concat(
        [news_df[["date", "ticker"]].reset_index(drop=True), emb_df, snt_df],
        axis=1,
    )

    # Daily mean per (date, ticker) within this rank's shard
    grouped = out.groupby(["date", "ticker"], as_index=False).mean()
    return grouped


# ----------------- merge logic -----------------

def merge_all(
    prices: pd.DataFrame,
    daily_news: pd.DataFrame,
    idx: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge prices with daily FinBERT features and index returns, then
    forward-fill news features per ticker and index features over time.
    """
    df = prices.merge(daily_news, on=["date", "ticker"], how="left")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    snt_cols = [c for c in ["sent_neg", "sent_neu", "sent_pos"] if c in df.columns]
    feat_cols = emb_cols + snt_cols

    if feat_cols:
        # Forward-fill news features per ticker; initial NaNs -> 0
        df[feat_cols] = (
            df.groupby("ticker")[feat_cols]
              .apply(lambda g: g.ffill())
              .reset_index(level=0, drop=True)
        )
        df[feat_cols] = df[feat_cols].fillna(0.0)

    # Merge index returns on date and ffill
    df = df.merge(idx, on="date", how="left")
    for c in ["ret_djia", "ret_nasdaqcom", "ret_sp500"]:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0.0)

    # Sort for downstream sequence building
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ----------------- main -----------------

def main():
    import time as _time

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--headlines_csv",
        type=str,
        default="Data/data/processed_headlines_subset.csv",
    )
    ap.add_argument(
        "--prices_csv",
        type=str,
        default="Data/data/processed_stock_prices.csv",
    )
    ap.add_argument(
        "--indexes_csv",
        type=str,
        default="Data/data/processed_indexes.csv",
    )
    ap.add_argument(
        "--base_model_name",
        type=str,
        default="ProsusAI/finbert",
        help="Base HF model name for FinBERT.",
    )
    ap.add_argument(
        "--finetuned_weights",
        type=str,
        default="models/model.safetensor",
        help="Path to fine-tuned FinBERT safetensors weights.",
    )
    ap.add_argument(
        "--output_path",
        type=str,
        default="Data/data/merged_lstm_dataset.parquet",
        help="Where to save the merged dataset (.parquet or .csv).",
    )
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed-precision for embeddings if CUDA is available.",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--dist_mode",
        type=str,
        default="none",
        choices=["none", "ddp"],
        help="Distribution mode: 'none' = single process, 'ddp' = multi-GPU via torch.distributed.",
    )

    args = ap.parse_args()
    set_seed(args.seed)

    # ---- distributed setup ----
    is_distributed, rank, world_size, is_main, device = ddp_setup(args.dist_mode)

    if is_main:
        print(
            f"[merge.py] dist_mode={args.dist_mode}, "
            f"is_distributed={is_distributed}, "
            f"world_size={world_size}, "
            f"device={device}",
            flush=True,
        )

    t_start = _time.perf_counter()

    # ---- load data ----
    prices = load_prices(args.prices_csv)
    news = load_headlines(args.headlines_csv)
    idx = load_indexes(args.indexes_csv)

    t_after_load = _time.perf_counter()

    if is_main:
        print(f"Prices:   {prices.shape}", flush=True)
        print(f"News:     {news.shape}", flush=True)
        print(f"Indexes:  {idx.shape}", flush=True)

    # Shard news across ranks if distributed
    if is_distributed:
        # simple contiguous sharding by row index
        n = len(news)
        per_rank = (n + world_size - 1) // world_size
        start_idx = rank * per_rank
        end_idx = min(start_idx + per_rank, n)
        news_shard = news.iloc[start_idx:end_idx].reset_index(drop=True)
        if is_main:
            print(
                f"[merge.py] Total headlines: {n}, world_size={world_size}, "
                f"per_rank≈{per_rank}",
                flush=True,
            )
        print(
            f"[Rank {rank}] headlines shard: {len(news_shard)} rows "
            f"(start={start_idx}, end={end_idx})",
            flush=True,
        )
    else:
        news_shard = news

    # ---- initialize FinBERT (per rank) ----
    t_before_finbert = _time.perf_counter()

    fb = FinBertEmbedder(
        base_model_name=args.base_model_name,
        finetuned_weights=args.finetuned_weights,
        device=device,
    )

    t_after_finbert = _time.perf_counter()

    # ---- compute embeddings for this rank's shard ----
    t_before_embed = _time.perf_counter()

    daily_news_shard = compute_daily_news_embeddings(
        news_df=news_shard,
        fb=fb,
        max_len=args.max_len,
        batch_size=args.batch_size,
        amp=args.amp,
        rank=rank,
    )

    t_after_embed = _time.perf_counter()
    embed_time_local = t_after_embed - t_before_embed

    # ---- gather all shards to build global daily_news on rank 0 ----
    if is_distributed:
        # all_gather_object to collect DataFrames
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, daily_news_shard)

        # Reduce embedding time: take max across ranks (rough idea of critical path)
        t_embed = torch.tensor(
            [embed_time_local],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(t_embed, op=dist.ReduceOp.MAX)
        embed_time_global_max = float(t_embed.item())
    else:
        gathered = [daily_news_shard]
        embed_time_global_max = embed_time_local

    if is_main:
        # Concatenate & group again to be safe (if (date,ticker) spanned multiple ranks)
        daily_news = pd.concat(gathered, ignore_index=True)
        if not daily_news.empty:
            daily_news = (
                daily_news.groupby(["date", "ticker"], as_index=False).mean()
            )
        else:
            daily_news = pd.DataFrame(columns=["date", "ticker"])

        print(
            f"[merge.py] Global daily_news shape after gather: {daily_news.shape}",
            flush=True,
        )
    else:
        daily_news = None

    # ---- only rank 0 does merge + save ----
    if is_main:
        t_before_merge = _time.perf_counter()

        merged = merge_all(prices, daily_news, idx)

        # Drop rows without target_ret1 (i.e., last day per ticker)
        before = len(merged)
        merged = merged.dropna(subset=["target_ret1"]).reset_index(drop=True)
        after = len(merged)
        print(
            f"[merge.py] Dropped {before - after} rows with NaN target_ret1 "
            f"(last days per ticker).",
            flush=True,
        )

        out_path = args.output_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if out_path.lower().endswith(".parquet"):
            merged.to_parquet(out_path, index=False)
        else:
            merged.to_csv(out_path, index=False)

        t_after_merge = _time.perf_counter()
        t_end = t_after_merge

        print(
            f"[merge.py] Saved merged dataset to {out_path} with shape {merged.shape}",
            flush=True,
        )

        # ---- benchmarking summary ----
        load_time = t_after_load - t_start
        finbert_init_time = t_after_finbert - t_before_finbert
        embed_local_time = embed_time_local
        embed_global_time = embed_time_global_max
        merge_time = t_after_merge - t_before_merge
        total_time = t_end - t_start

        print("\n[merge.py] Benchmark (seconds):", flush=True)
        print(f"  load_data        = {load_time:.3f}", flush=True)
        print(f"  finbert_init     = {finbert_init_time:.3f}", flush=True)
        print(f"  embed_local_rank0= {embed_local_time:.3f}", flush=True)
        print(f"  embed_global_max = {embed_global_time:.3f}", flush=True)
        print(f"  merge+save       = {merge_time:.3f}", flush=True)
        print(f"  total            = {total_time:.3f}", flush=True)

    # clean up process group if needed
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
