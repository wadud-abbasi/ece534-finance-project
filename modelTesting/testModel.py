import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless backend for NERSC/login nodes
import matplotlib.pyplot as plt


# -----------------------------
# Model definition (matches checkpoint structure)
# -----------------------------
class TinyLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]       # (B, hidden_dim)
        last = self.ln(last)
        return self.mlp(last).squeeze(-1)  # (B,)


# -----------------------------
# Sequence builder
# -----------------------------
def build_sequences(df, feature_cols, target_col, date_col, seq_len):
    """
    Returns:
      dates: np.array of datetime (one per prediction)
      X: np.array [N, seq_len, F]
      y: np.array [N]
    """
    values = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(df[date_col]).to_numpy()

    n = len(df)
    if n < seq_len:
        raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}")

    X_list, y_list, d_list = [], [], []

    for i in range(n - seq_len + 1):
        j = i + seq_len - 1  # index of prediction day
        X_list.append(values[i : i + seq_len])
        y_list.append(targets[j])
        d_list.append(dates[j])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    d = np.array(d_list)

    return d, X, y


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model_path",
        type=str,
        default="models/lstm_ddp_baby/lstm_best.pt",
        help="Path to lstm_best.pt",
    )
    ap.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .parquet or .csv dataset",
    )
    ap.add_argument(
        "--date_col",
        type=str,
        default="date",
        help="Name of date column in dataset",
    )
    ap.add_argument(
        "--target_col",
        type=str,
        default="adj_close",
        help="Name of target price column",
    )
    ap.add_argument(
        "--seq_len",
        type=int,
        default=30,
        help="Sequence length used during training",
    )
    ap.add_argument(
        "--start_year",
        type=int,
        default=2015,
        help="Only use data from this year onwards",
    )
    ap.add_argument(
        "--output_png",
        type=str,
        default="lstm_predictions_vs_actual.png",
        help="Output PNG filename for plot",
    )

    args = ap.parse_args()

    # -------------------------
    # 1) Load checkpoint & infer model dims
    # -------------------------
    print(f"Loading checkpoint from {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
    hidden_dim = state_dict["lstm.weight_hh_l0"].shape[1]
    num_layers = len(
        [k for k in state_dict.keys() if k.startswith("lstm.weight_ih_l") and "reverse" not in k]
    )

    print(f"Inferred input_dim  = {input_dim}")
    print(f"Inferred hidden_dim = {hidden_dim}")
    print(f"Inferred num_layers = {num_layers}")

    model = TinyLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    model.eval()

    # -------------------------
    # 2) Load dataset
    # -------------------------
    print(f"Loading data from {args.data_path}")
    if args.data_path.endswith(".parquet") or args.data_path.endswith(".pq"):
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path)

    # Ensure date column is datetime; sort + filter by year
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.sort_values(args.date_col)

    start_date = dt.datetime(args.start_year, 1, 1)
    df = df[df[args.date_col] >= start_date].reset_index(drop=True)
    print(f"Rows from {start_date.date()} onward: {len(df)}")

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in dataframe columns.")

    # -------------------------
    # 3) Auto feature selection: keep all non-emb, and just enough emb_* to reach input_dim
    # -------------------------
    num_df = df.select_dtypes(include=[np.number])

    if args.target_col not in num_df.columns:
        raise ValueError(
            f"Target column '{args.target_col}' must be numeric; "
            f"got numeric cols: {num_df.columns.tolist()}"
        )

    num_cols = [c for c in num_df.columns if c != args.target_col]

    emb_cols = [c for c in num_cols if c.startswith("emb_")]
    non_emb_cols = [c for c in num_cols if not c.startswith("emb_")]

    print(f"Total numeric (excluding target): {len(num_cols)}")
    print(f"  Non-embedding cols: {len(non_emb_cols)}")
    print(f"  Embedding cols:     {len(emb_cols)}")

    if len(non_emb_cols) > input_dim:
        raise ValueError(
            f"You have {len(non_emb_cols)} non-embedding numeric columns but "
            f"model only accepts {input_dim} features. You must drop some non-emb cols manually."
        )

    needed_from_emb = input_dim - len(non_emb_cols)
    if needed_from_emb < 0:
        raise ValueError(
            f"Computed negative needed_from_emb ({needed_from_emb}). "
            "Something is inconsistent with feature engineering."
        )
    if needed_from_emb > len(emb_cols):
        raise ValueError(
            f"Need {needed_from_emb} embedding columns to reach {input_dim} features, "
            f"but only {len(emb_cols)} emb_* columns are available."
        )

    # Keep only the first `needed_from_emb` embedding columns, and drop the rest
    keep_emb_cols = emb_cols[:needed_from_emb]
    keep_emb_set = set(keep_emb_cols)

    # Preserve original numeric column order while dropping excess embeddings
    feature_cols = [
        c for c in num_cols
        if (not c.startswith("emb_")) or (c in keep_emb_set)
    ]

    if len(feature_cols) != input_dim:
        raise ValueError(
            f"After auto-selection, feature_cols has length {len(feature_cols)}, "
            f"but model expects input_dim={input_dim}."
        )

    print(f"Using {len(feature_cols)} feature columns as model input.")
    print("First 10 feature cols:", feature_cols[:10])
    print("Last 10 feature cols:", feature_cols[-10:])

    # -------------------------
    # 4) Build sequences and targets
    # -------------------------
    dates, X_np, y_np = build_sequences(
        df, feature_cols, args.target_col, args.date_col, args.seq_len
    )
    print(f"Built {X_np.shape[0]} sequences of length {args.seq_len}")
    print(f"X shape: {X_np.shape}, y shape: {y_np.shape}")

    # -------------------------
    # 5) Run predictions (CPU)
    # -------------------------
    X = torch.from_numpy(X_np)
    preds = []

    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i : i + batch_size]
            yb = model(xb)
            preds.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    print("Predictions shape:", preds.shape)

    # -------------------------
    # 6) Plot predictions vs actual and save
    # -------------------------
    print("Plotting predictions vs actual...")
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_np, label="Actual", linewidth=1.5)
    plt.plot(dates, preds, label="Predicted", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel(args.target_col)
    plt.title(f"LSTM predictions vs {args.target_col} (>= {args.start_year})")
    plt.legend()
    plt.tight_layout()

    out_path = args.output_png
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
