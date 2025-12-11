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
@@ -45,17 +50,36 @@ class _DummyInterpolationMode:
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
@@ -66,13 +90,11 @@ class _DummyInterpolationMode:
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
AutoModelForSequenceClassification,
AutoTokenizer,
DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
@@ -88,6 +110,7 @@ class _DummyInterpolationMode:
# Splitting utilities
# ---------------------------------------------------------------------------


def split_dataset_time_order(
df: pd.DataFrame,
date_column: str,
@@ -138,6 +161,7 @@ def split_dataset_random(
# Dataset + metrics
# ---------------------------------------------------------------------------


class NewsDataset(Dataset):
"""
   PyTorch dataset that holds text + label IDs and tokenizes on the fly.
@@ -171,17 +195,16 @@ def __getitem__(self, idx):
return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
def compute_accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}
    return float((preds == labels).mean())


# ---------------------------------------------------------------------------
# Main training logic
# Data loading / preparation
# ---------------------------------------------------------------------------


def load_and_prepare_data(
data_path: str,
text_column: str,
@@ -199,15 +222,29 @@ def load_and_prepare_data(
if not os.path.exists(data_path):
raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading Parquet data from: {data_path}", flush=True)
    df = pd.read_parquet(data_path)
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
        raise ValueError(f"text_column '{text_column}' not found in columns: {list(df.columns)}")
        raise ValueError(
            f"text_column '{text_column}' not found in columns: {list(df.columns)}"
        )
if label_column not in df.columns:
        raise ValueError(f"label_column '{label_column}' not found in columns: {list(df.columns)}")
        raise ValueError(
            f"label_column '{label_column}' not found in columns: {list(df.columns)}"
        )

# Keep only necessary columns plus date if present
keep_cols = {text_column, label_column}
@@ -220,13 +257,10 @@ def load_and_prepare_data(

# Normalize / map labels
if pd.api.types.is_numeric_dtype(df[label_column]):
        # Assume labels are already integer-encoded (0..2).
        # If there are out-of-range values, this will drop them.
df["label_id"] = df[label_column].astype(int)
valid_ids = set(ID2LABEL.keys())
df = df[df["label_id"].isin(valid_ids)]
else:
        # String labels -> map via LABEL2ID
df[label_column] = df[label_column].astype(str).str.lower().str.strip()
df = df[df[label_column].isin(LABEL2ID.keys())].copy()
df["label_id"] = df[label_column].map(LABEL2ID)
@@ -253,17 +287,112 @@ def load_and_prepare_data(
print(f"Train size: {len(train_df)}  |  Val size: {len(val_df)}", flush=True)

if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError("Train or val set ended up empty. Check your data and val_fraction.")
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

    # -----------------------------------------------------------------------
# 1. Load data
    # -----------------------------------------------------------------------
train_df, val_df = load_and_prepare_data(
data_path=args.data_path,
text_column=args.text_column,
@@ -272,9 +401,7 @@ def main(args):
val_fraction=args.val_fraction,
)

    # -----------------------------------------------------------------------
    # 2. Hugging Face model + tokenizer
    # -----------------------------------------------------------------------
    # 2. Load model & tokenizer
print(f"Loading model and tokenizer from '{args.model_name}'", flush=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(
@@ -284,6 +411,14 @@ def main(args):
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
@@ -297,54 +432,37 @@ def main(args):
max_length=args.max_length,
)

    if len(train_dataset) == 0:
        raise RuntimeError("Training set is empty after all filtering.")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    fsdp_config = args.fsdp if args.fsdp else None
    fsdp_wrap_cls = "BertLayer" if args.fsdp else None

    # -----------------------------------------------------------------------
    # 3. TrainingArguments + Trainer
    # -----------------------------------------------------------------------
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
        fsdp=fsdp_config,
        fsdp_transformer_layer_cls_to_wrap=fsdp_wrap_cls,
        gradient_checkpointing=args.gradient_checkpointing,
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

    trainer = Trainer(
    # 4. Train
    train_model(
model=model,
        args=training_args,
        train_dataset=train_dataset if len(train_dataset) > 0 else None,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
)

    if len(train_dataset) == 0:
        raise RuntimeError("Training set is empty after all filtering.")

    # -----------------------------------------------------------------------
    # 4. Train and save
    # -----------------------------------------------------------------------
    trainer.train()
    trainer.save_model(args.output_dir)
    # 5. Save model & tokenizer
    model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"Model and tokenizer saved to {args.output_dir}", flush=True)

@@ -368,20 +486,20 @@ def main(args):
"--text_column",
type=str,
default="Article_title",
        help="Name of the text column in the Parquet file.",
        help="Name of the text column in the Parquet/CSV file.",
)
parser.add_argument(
"--label_column",
type=str,
default="label",
        help="Name of the label column in the Parquet file.",
        help="Name of the label column in the Parquet/CSV file.",
)
parser.add_argument(
"--date_column",
type=str,
default="Date",
help="Name of the date column for time-based splitting. "
             "If not present or empty, random splitting is used.",
        "If not present or empty, random splitting is used.",
)

parser.add_argument(
@@ -412,7 +530,7 @@ def main(args):
"--batch_size",
type=int,
default=16,
        help="Per-device batch size.",
        help="Batch size.",
)
parser.add_argument(
"--epochs",
@@ -433,13 +551,11 @@ def main(args):
help="Weight decay.",
)

    # FSDP / parallelism flags
parser.add_argument(
"--fsdp",
type=str,
default="",
        help='FSDP config string passed to TrainingArguments.fsdp '
             '(e.g. "full_shard" or "" to disable).',
        help="(Ignored in this script) kept for CLI compatibility.",
)
parser.add_argument(
"--gradient_checkpointing",
