#!/usr/bin/env python
"""
Fine-tune FinBERT/BERT to predict stock price direction
('decreasing', 'stable', 'increasing') from news headlines.

Assumptions:
- Headlines CSV has columns:
    Date, Article_title, Stock_symbol
- Prices CSV has columns:
    date, adj_close, ticker
- The full data files are:
    processed_headlines_subset.csv
    processed_stock_prices.csv
  (same schema as the *_HEAD previews you uploaded, just without "_HEAD").
"""

import argparse
import os

# Force Transformers to ignore torchvision/image stack
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["DISABLE_TRANSFORMERS_AV"] = "1"

import sys
import types
import importlib.machinery

# ---- Torchvision stub to satisfy Transformers checks ----
# We don't use vision/video features; this just keeps imports from breaking.

# Create a fake torchvision module
torchvision_stub = types.ModuleType("torchvision")
torchvision_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision", None)

# Fake torchvision.transforms submodule
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

# Fake torchvision.io submodule
io_stub = types.ModuleType("torchvision.io")
io_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", None)

# Wire stubs onto the root module
torchvision_stub.transforms = transforms_stub
torchvision_stub.io = io_stub

# Register everything in sys.modules so imports see them as real
sys.modules["torchvision"] = torchvision_stub
sys.modules["torchvision.transforms"] = transforms_stub
sys.modules["torchvision.io"] = io_stub

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# Ordered label list so IDs are consistent everywhere
LABELS = ["decreasing", "stable", "increasing"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# 1. Label construction from prices
# ---------------------------------------------------------------------------

def build_label_df(
    price_df: pd.DataFrame,
    horizon_trading_days: int = 1,
    stable_threshold: float = 0.005,
) -> pd.DataFrame:
    """
    For a single ticker, create daily labels based on future returns
    over `horizon_trading_days` *trading* days.

    r_t = (adj_close_{t+h} - adj_close_t) / adj_close_t

    - r_t < -stable_threshold      -> 'decreasing'
    - |r_t| <= stable_threshold    -> 'stable'
    - r_t > stable_threshold       -> 'increasing'

    Returns DataFrame with columns: ["date", "label", "ret"].
    """
    df = price_df.sort_values("date").reset_index(drop=True).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "label", "ret"])

    # Price at t + h
    df["target_adj_close"] = df["adj_close"].shift(-horizon_trading_days)

    # Drop trailing rows without a future price
    df = df.iloc[:-horizon_trading_days].copy()

    # Future return
    df["ret"] = (df["target_adj_close"] - df["adj_close"]) / df["adj_close"]

    # Map returns to labels
    conditions = [
        df["ret"] < -stable_threshold,
        df["ret"].abs() <= stable_threshold,
    ]
    choices = ["decreasing", "stable"]
    df["label"] = np.select(conditions, choices, default="increasing")

    return df[["date", "label", "ret"]]


def build_labeled_dataset(
    headlines_csv: str,
    prices_csv: str,
    horizon_trading_days: int = 1,
    stable_threshold: float = 0.005,
) -> pd.DataFrame:
    """
    Join news headlines with price-direction labels per ticker.

    Logic:
    - For each ticker, compute per-day labels using `build_label_df`.
    - For each headline, find the last trading day STRICTLY BEFORE the
      headline timestamp, and use that day’s label (future move from that day).

    This approximates:
      "Given this headline at time T, what happens to price over the next
       `horizon_trading_days` trading days, measured from the last close
       before T?"

    Returns a DataFrame with original headline columns plus:
        label (string), ret (float)
    """
    headlines_df = pd.read_csv(headlines_csv, low_memory=False)
    print("Loaded headlines:", headlines_df.shape, flush=True)

    prices_df = pd.read_csv(prices_csv, low_memory=False)
    print("Loaded prices:", prices_df.shape, flush=True)

    # Parse datetimes
    # Headlines often look like "2023-12-16 23:00:00 UTC"
    headlines_df["Date"] = pd.to_datetime(
        headlines_df["Date"].astype(str).str.replace(" UTC", ""),
        errors="coerce",
    )
    prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce")

    all_records = []

    for sym in sorted(headlines_df["Stock_symbol"].dropna().unique()):
        hsym = headlines_df[headlines_df["Stock_symbol"] == sym].copy()
        psym = prices_df[prices_df["ticker"] == sym].copy()

        if psym.empty:
            continue

        label_df = build_label_df(
            psym,
            horizon_trading_days=horizon_trading_days,
            stable_threshold=stable_threshold,
        )
        label_df = label_df.sort_values("date").reset_index(drop=True)
        if label_df.empty:
            continue

        # Normalize to date-only for mapping
        label_dates = label_df["date"].dt.normalize().values
        headline_dates = hsym["Date"].dt.normalize().values

        # For each headline date, map to the last trading day STRICTLY BEFORE it
        # (so we don't peek at same-day close that happens after the headline).
        idx = np.searchsorted(label_dates, headline_dates, side="left") - 1
        valid_mask = idx >= 0
        if not valid_mask.any():
            continue

        hsym_valid = hsym.loc[valid_mask].copy()
        mapped_idx = idx[valid_mask]

        hsym_valid["label"] = label_df["label"].values[mapped_idx]
        hsym_valid["ret"] = label_df["ret"].values[mapped_idx]

        all_records.append(hsym_valid)

    if not all_records:
        # No joinable rows
        return pd.DataFrame(
            columns=list(headlines_df.columns) + ["label", "ret"]
        )

    ds = pd.concat(all_records, ignore_index=True)
    return ds


# ---------------------------------------------------------------------------
# 2. Train/val split (time-based)
# ---------------------------------------------------------------------------

def split_dataset_time_order(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: earliest -> train, latest -> val.
    Avoids training on future data and validating on the past.
    """
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    n = len(df_sorted)
    if n == 0:
        return df_sorted, df_sorted  # both empty

    n_val = max(1, int(n * val_fraction))
    n_train = max(0, n - n_val)

    train_df = df_sorted.iloc[:n_train].reset_index(drop=True)
    val_df = df_sorted.iloc[n_train:].reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# 3. Dataset + metrics
# ---------------------------------------------------------------------------

class NewsDataset(Dataset):
    """
    Simple PyTorch dataset that holds headlines and label IDs.
    """

    def __init__(
        self,
        texts: pd.Series,
        labels: pd.Series,
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True).astype(int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = label
        return enc


def compute_metrics(eval_pred):
    """
    Simple accuracy metric.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}


# ---------------------------------------------------------------------------
# 4. Main training entrypoint
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Build labeled (headline, label) dataset ----
    df = build_labeled_dataset(
        headlines_csv=args.headlines_csv,
        prices_csv=args.prices_csv,
        horizon_trading_days=args.horizon_days,
        stable_threshold=args.stable_threshold,
    )

    # Drop rows without needed fields
    df = df.dropna(subset=["Article_title", "label"]).copy()
    df = df[df["label"].isin(LABEL2ID.keys())].copy()

    if df.empty:
        raise RuntimeError(
            "No rows left after labeling & cleanup. "
            "Check your CSV paths and labeling parameters."
        )

    # Map string labels to numeric IDs
    df["label_id"] = df["label"].map(LABEL2ID)

    # Time-based split to avoid look-ahead bias
    train_df, val_df = split_dataset_time_order(df, val_fraction=args.val_fraction)

    print(f"Total labeled headlines: {len(df)}")
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")
    print("Label distribution (total):")
    print(df["label"].value_counts())

    # ---- Hugging Face model + tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = NewsDataset(
        texts=train_df["Article_title"],
        labels=train_df["label_id"],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = NewsDataset(
        texts=val_df["Article_title"],
        labels=val_df["label_id"],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # FSDP config (optional)
    fsdp_config = args.fsdp if args.fsdp else None
    fsdp_wrap_cls = "BertLayer" if args.fsdp else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",  # disable wandb, etc.

        # Multi-GPU / FSDP
        fsdp=fsdp_config,  # e.g. "full_shard" or "" (disabled)
        fsdp_transformer_layer_cls_to_wrap=fsdp_wrap_cls,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if len(train_dataset) > 0 else None,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training set is empty. Check your val_fraction or data size.")

    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT/FinBERT to predict price direction from headlines."
    )

    parser.add_argument(
        "--headlines_csv",
        type=str,
        required=True,
        help="Path to headlines CSV (e.g., processed_headlines_subset.csv).",
    )
    parser.add_argument(
        "--prices_csv",
        type=str,
        required=True,
        help="Path to stock prices CSV (e.g., processed_stock_prices.csv).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ProsusAI/finbert",
        help="HF model name or local path (e.g., ProsusAI/finbert, bert-base-uncased).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finbert_price_direction",
        help="Where to save checkpoints and final model.",
    )
    parser.add_argument(
        "--horizon_days",
        type=int,
        default=1,
        help="Number of TRADING days ahead to measure return over.",
    )
    parser.add_argument(
        "--stable_threshold",
        type=float,
        default=0.005,
        help="Abs(return) below this is 'stable' (0.005 = ±0.5%).",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of data used for validation (time-based split).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=96,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )

    # FSDP / parallelism flags
    parser.add_argument(
        "--fsdp",
        type=str,
        default="",
        help='FSDP config string passed to TrainingArguments.fsdp '
             '(e.g. "full_shard" or "" to disable).',
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (helps memory at some speed cost).",
    )

    args = parser.parse_args()
    main(args)
