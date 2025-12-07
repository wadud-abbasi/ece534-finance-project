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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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
    rank: int = 0,
    world_size: int = 1,
    patience: int = 3,
    max_grad_norm: float = 1.0,
    use_lr_scheduler: bool = True,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # Learning rate scheduler (ReduceLROnPlateau - reduces LR when validation plateaus)
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        )
    else:
        scheduler = None
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        if world_size > 1 and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
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
            
            # Gradient clipping to prevent exploding gradients
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Aggregate metrics across all processes if distributed
        if world_size > 1:
            # Create tensors for reduction
            train_loss_tensor = torch.tensor(train_loss, device=device)
            train_total_tensor = torch.tensor(train_total, device=device)
            train_correct_tensor = torch.tensor(train_correct, device=device)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
            
            train_loss = train_loss_tensor.item()
            train_total = train_total_tensor.item()
            train_correct = train_correct_tensor.item()
        
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
        
        # Aggregate validation metrics across all processes if distributed
        if world_size > 1:
            val_loss_tensor = torch.tensor(val_loss, device=device)
            val_total_tensor = torch.tensor(val_total, device=device)
            val_correct_tensor = torch.tensor(val_correct, device=device)
            
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
            
            val_loss = val_loss_tensor.item()
            val_total = val_total_tensor.item()
            val_correct = val_correct_tensor.item()

        avg_val_loss = val_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        
        # Update learning rate scheduler based on validation loss
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            if rank == 0 and current_lr < lr * 0.9:  # Only print if LR actually changed
                print(f"  Learning rate reduced to: {current_lr:.2e}", flush=True)

        # Early stopping logic (only rank 0 makes decisions)
        if rank == 0:
            # Check if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1
                improved = False
            
            # Print epoch results
            improvement_str = " *" if improved else ""
            print(
                f"Epoch {epoch}: "
                f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}"
                f"{improvement_str}",
                flush=True,
            )
            
            # Check for early stopping (only if patience > 0)
            if patience > 0 and patience_counter >= patience:
                print(
                    f"\nEarly stopping triggered after {epoch} epochs. "
                    f"No improvement in validation loss for {patience} epochs. "
                    f"Best validation loss: {best_val_loss:.4f}",
                    flush=True,
                )
                break
        
        # Synchronize all processes before continuing (for distributed training)
        if world_size > 1:
            # Create a flag tensor to check if we should stop
            should_stop = torch.tensor(
                (patience > 0 and patience_counter >= patience) if rank == 0 else False,
                device=device
            )
            dist.broadcast(should_stop, src=0)
            if should_stop.item():
                # All processes break together
                break
    
    # Training completed - model state is already at the final epoch
    if rank == 0:
        print(f"\nTraining completed. Final model state saved.", flush=True)


# ---------------------------------------------------------------------------
# Distributed training setup
# ---------------------------------------------------------------------------


def setup_distributed():
    """
    Initialize distributed training using SLURM environment variables.
    Returns (rank, world_size, local_rank, device)
    """
    # Check if we're in a distributed environment
    if "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        
        # MASTER_ADDR and MASTER_PORT should be set by the SLURM script
        # If not set, use defaults
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
    else:
        # Single process (non-distributed)
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    
    # Set CUDA device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    # Initialize process group if distributed
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size,
        )
        print(
            f"Rank {rank}: Initialized process group - world_size={world_size}, "
            f"local_rank={local_rank}, device={device}, "
            f"master_addr={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            flush=True,
        )
    else:
        print(f"Running in single-process mode, device={device}", flush=True)
    
    return rank, world_size, local_rank, device


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    
    # Only rank 0 creates output directory
    if rank == 0:
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
    if rank == 0:
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

    print(f"Rank {rank}: Using device: {device}", flush=True)
    model.to(device)
    
    # Wrap model with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if rank == 0:
            print("Model wrapped with DistributedDataParallel", flush=True)

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

    # Use DistributedSampler for distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            collate_fn=data_collator,
            num_workers=0,
        )
        if rank == 0:
            print(f"Using DistributedSampler: each process sees {len(train_sampler)} train samples", flush=True)
    else:
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
        rank=rank,
        world_size=world_size,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm,
        use_lr_scheduler=args.use_lr_scheduler,
    )

    # 5. Save model & tokenizer (only from rank 0)
    if rank == 0:
        # Unwrap DDP model if needed
        if world_size > 1:
            model_to_save = model.module
        else:
            model_to_save = model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model and tokenizer saved to {args.output_dir}", flush=True)
    
    # Cleanup distributed training
    cleanup_distributed()


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
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs to wait for validation improvement before early stopping. "
        "Set to 0 to disable early stopping.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping. Set to 0 to disable gradient clipping.",
    )
    parser.add_argument(
        "--no_lr_scheduler",
        action="store_false",
        dest="use_lr_scheduler",
        help="Disable learning rate scheduler (ReduceLROnPlateau). Default: enabled.",
    )

    args = parser.parse_args()

    # If user passes empty string for date_column, treat as None
    if args.date_column == "" or args.date_column is None:
        args.date_column = None

    main(args)
