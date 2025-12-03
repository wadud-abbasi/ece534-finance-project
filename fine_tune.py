#!/usr/bin/env python
"""
Fine-tune FinBERT/BERT to predict stock price direction
('decreasing', 'stable', 'increasing') from a pre-labeled Parquet file.

Expected data file (by default):
    data/data/labeled_headlines.parquet

The Parquet file should contain at least:
    - a text column        (e.g., "Article_title")
    - a label column       (string labels like 'decreasing', 'stable', 'increasing'
                            OR numeric labels 0,1,2)
Optionally:
    - a date column        (e.g., "Date") for time-based splitting

You can configure column names with CLI flags:
    --text_column, --label_column, --date_column
"""

import argparse
import os

# ---------------------------------------------------------------------------
# Torchvision stub to satisfy Transformers checks (we don't actually use it)
# ---------------------------------------------------------------------------

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["DISABLE_TRANSFORMERS_AV"] = "1"

import sys
import types
import importlib.machinery

# Root torchvision module (as a "package")
torchvision_stub = types.ModuleType("torchvision")
torchvision_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision", None)
torchvision_stub.__path__ = []

# torchvision.transforms (package)
transforms_stub = types.ModuleType("torchvision.transforms")
transforms_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", None)
transforms_stub.__path__ = []


class _DummyInterpolationMode:
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2
    LANCZOS = 3
    BOX = 4
    HAMMING = 5


transforms_stub.InterpolationMode = _DummyInterpolationMode

# torchvision.io stub
io_stub = types.ModuleType("torchvision.io")
io_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", None)

# torchvision.transforms.v2 (package)
v2_stub = types.ModuleType("torchvision.transforms.v2")
v2_stub.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms.v2", None)
v2_stub.__path__ = []

# torchvision.transforms.v2.functional stub
functional_stub = types.ModuleType("torchvision.transforms.v2.functional")
functional_stub.__spec__ = importlib.machinery.ModuleSpec(
    "torchvision.transforms.v2.functional", None
)

# Wire things together
v2_stub.functional = functional_stub

torchvision_stub.transforms = transforms_stub
torchvision_stub.io = io_stub

# Register in sys.modules so imports succeed
sys.modules["torchvision"] = torchvision_stub
sys.modules["torchvision.transforms"] = transforms_stub
sys.modules["torchvision.io"] = io_stub
sys.modules["torchvision.transforms.v2"] = v2_stub
sys.modules["torchvision.transforms.v2.functional"] = functional_stub

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

LABELS = ["decreasing", "stable", "increasing"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Splitting utilities
# ---------------------------------------------------------------------------


def split_dataset_time_order(
    df: pd.DataFrame,
    date_column: str,
    val_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split using the given date column (earliest -> train, latest -> val).
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    df_sorted = df.sort_values(date_column).reset_index(drop=True)

    n = len(df_sorted)
    if n == 0:
        return df_sorted, df_sorted

    n_val = max(1, int(n * val_fraction))
    n_train = max(0, n - n_val)

    train_df = df_sorted.iloc[:n_train].reset_index(drop=True)
    val_df = df_sorted.iloc[n_train:].reset_index(drop=True)
    return train_df, val_df


def split_dataset_random(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random split if no usable date column exists.
    """
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    if n == 0:
        return df_shuffled, df_shuffled

    n_val = max(1, int(n * val_fraction))
    n_train = max(0, n - n_val)

    train_df = df_shuffled.iloc[:n_train].reset_index(drop=True)
    val_df = df_shuffled.iloc[n_train:].reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# Dataset + metrics
# ---------------------------------------------------------------------------


class NewsDataset(Dataset):
    """
    PyTorch dataset that holds text + label IDs and tokenizes on the fly.
    """

    def __init__(
        self,
        texts: pd.Series,
        labels: pd.Series,
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts.reset_index(drop=True).astype(str)
        self.labels = labels.reset_index(drop=True).astype(int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = int(self.labels.iloc[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = label
        return enc


def compute_accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argmax(logits, axis=-1)
    return float((preds == labels).mean())


# ---------------------------------------------------------------------------
# Data loading / preparation
# ---------------------------------------------------------------------------


def load_and_prepare_data(
    data_path: str,
    text_column: str,
    label_column: str,
    date_column: Optional[str],
    val_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Parquet file and prepare train/val splits.

    - Accepts string labels ('decreasing', 'stable', 'increasing') or numeric (0/1/2).
    - Drops rows with missing text/label.
    - Uses time-based split if date_column is provided and exists; else random split.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Support both parquet and csv just in case
    ext = os.path.splitext(data_path)[1].lower()
    print(f"Loading data from: {data_path} (ext={ext})", flush=True)
    if ext == ".parquet":
        df = pd.read_parquet(data_path)
    elif ext == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(
            f"Unsupported file extension {ext!r}. Use a .parquet or .csv file."
        )

    print("Raw data shape:", df.shape, flush=True)

    # Check required columns
    if text_column not in df.columns:
        raise ValueError(
            f"text_column '{text_column}' not found in columns: {list(df.columns)}"
        )
    if label_column not in df.columns:
        raise ValueError(
            f"label_column '{label_column}' not found in columns: {list(df.columns)}"
        )

    # Keep only necessary columns plus date if present
    keep_cols = {text_column, label_column}
    if date_column is not None and date_column in df.columns:
        keep_cols.add(date_column)
    df = df[list(keep_cols)].copy()

    # Drop missing text/labels
    df = df.dropna(subset=[text_column, label_column])

    # Normalize / map labels
    if pd.api.types.is_numeric_dtype(df[label_column]):
        df["label_id"] = df[label_column].astype(int)
        valid_ids = set(ID2LABEL.keys())
        df = df[df["label_id"].isin(valid_ids)]
    else:
        df[label_column] = df[label_column].astype(str).str.lower().str.strip()
        df = df[df[label_column].isin(LABEL2ID.keys())].copy()
        df["label_id"] = df[label_column].map(LABEL2ID)

    df.rename(columns={text_column: "text"}, inplace=True)

    if df.empty:
        raise RuntimeError("No usable rows after cleaning and label mapping.")

    print("Cleaned data shape:", df.shape, flush=True)
    print("Label distribution:")
    print(df["label_id"].value_counts().sort_index().rename(index=ID2LABEL))

    # Train/val split
    if date_column is not None and date_column in df.columns:
        print(f"Using time-based split on column '{date_column}'", flush=True)
        train_df, val_df = split_dataset_time_order(
            df, date_column=date_column, val_fraction=val_fraction
        )
    else:
        print("Using random split (no valid date_column provided).", flush=True)
        train_df, val_df = split_dataset_random(df, val_fraction=val_fraction)

    print(f"Train size: {len(train_df)}  |  Val size: {len(val_df)}", flush=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError(
            "Train or val set ended up empty. Check your data and val_fraction."
        )

    return train_df, val_df


# ---------------------------------------------------------------------------
# Training loop (manual, no Trainer)
# ---------------------------------------------------------------------------


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(1, epochs + 1):
        # --------------------
        # Train
        # --------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            labels = batch["labels"].to(device)
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k != "labels"
            }

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # --------------------
        # Validation
        # --------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].to(device)
                inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k != "labels"
                }

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    train_df, val_df = load_and_prepare_data(
        data_path=args.data_path,
        text_column=args.text_column,
        label_column=args.label_column,
        date_column=args.date_column,
        val_fraction=args.val_fraction,
    )

    # 2. Load model & tokenizer
    print(f"Loading model and tokenizer from '{args.model_name}'", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model.to(device)

    # 3. Datasets & loaders
    train_dataset = NewsDataset(
        texts=train_df["text"],
        labels=train_df["label_id"],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = NewsDataset(
        texts=val_df["text"],
        labels=val_df["label_id"],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training set is empty after all filtering.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # 4. Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 5. Save model & tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT/FinBERT on a labeled Parquet headlines dataset."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/data/labeled_headlines.parquet",
        help="Path to labeled Parquet file.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="Article_title",
        help="Name of the text column in the Parquet/CSV file.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label column in the Parquet/CSV file.",
    )
    parser.add_argument(
        "--date_column",
        type=str,
        default="Date",
        help="Name of the date column for time-based splitting. "
        "If not present or empty, random splitting is used.",
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
        default="./finbert_price_direction_parquet",
        help="Where to save checkpoints and final model.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of data used for validation.",
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
        help="Batch size.",
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

    parser.add_argument(
        "--fsdp",
        type=str,
        default="",
        help="(Ignored in this script) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (helps memory at some speed cost).",
    )

    args = parser.parse_args()

    # If user passes empty string for date_column, treat as None
    if args.date_column == "" or args.date_column is None:
        args.date_column = None

    main(args)
