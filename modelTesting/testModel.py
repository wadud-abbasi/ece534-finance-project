import argparse
import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -----------------------------
# Model definition (matches checkpoint)
# -----------------------------
class TinyLSTMModel(nn.Module):
    def __init__(self, input_dim=135, hidden_dim=256, num_layers=2):
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
        # x: (B, T, input_dim)
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # (B, hidden_dim)
        last = self.ln(last)
        return self.mlp(last).squeeze(-1)   # (B,)


# -----------------------------
# Helper: load data (csv or parquet)
# -----------------------------
def load_frame(path: str) -> pd.DataFrame:
    if path.endswith(".parquet") or path.endswith(".pq"):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


# -----------------------------
# Build sequences & targets
# -----------------------------
def build_sequences(df, feature_cols, target_col, date_col, seq_len):
    """
    Returns:
      dates: list of datetime (one per prediction, at end of each window)
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
        j = i + seq_len - 1  # index of end of window
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
        default="../models/lstm_ddp_baby/lstm_best.pt",
        help="Path to lstm_best.pt",
    )
    ap.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to your daily data (csv or parquet)",
    )
    ap.add_argument(
        "--date_col",
        type=str,
        default="date",
        help="Name of date column",
    )
    ap.add_argument(
        "--target_col",
        type=str,
        default="adj_close",  # <-- change if your price column is named differently
        help="Name of target price column",
    )
    ap.add_argument(
        "--seq_len",
        type=int,
        default=30,           # <-- set to the same seq_len you used in training
        help="Sequence length used for LSTM",
    )
    ap.add_argument(
        "--start_year",
        type=int,
        default=2015,
        help="Only use data from this year onwards",
    )

    args = ap.parse_args()

    # 1) Load checkpoint and build model
    print(f"Loading checkpoint from {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    # Infer input_dim from weights, but we know it's 135 from earlier
    input_dim = state_dict["lstm.weight_ih_l0"].shape[1]  # (4*hidden, input_dim)
    print(f"Inferred input_dim = {input_dim}")

    model = TinyLSTMModel(input_dim=input_dim, hidden_dim=256, num_layers=2)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()

    # 2) Load data
    print(f"Loading data from {args.data_path}")
    df = load_frame(args.data_path)

    # Ensure date column is datetime and filter by year >= 2015
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.sort_values(args.date_col)

    start_date = dt.datetime(args.start_year, 1, 1)
    df = df[df[args.date_col] >= start_date].reset_index(drop=True)
    print(f"Filtered to rows from {start_date.date()} onwards: {len(df)} rows")

    # 3) Select feature columns
    #    Strategy: use all numeric cols except the target column
    num_df = df.select_dtypes(include=[np.number])
    if args.target_col not in num_df.columns:
        raise ValueError(
            f"Target column '{args.target_col}' must be numeric and present in data."
        )

    feature_cols = [c for c in num_df.columns if c != args.target_col]
    print(f"Using {len(feature_cols)} feature columns as input.")

    if len(feature_cols) != input_dim:
        raise ValueError(
            f"Feature dim mismatch: model expects {input_dim}, "
            f"but got {len(feature_cols)} numeric feature columns.\n"
            f"Feature cols: {feature_cols}"
        )

    # 4) Build sequences
    dates, X_np, y_np = build_sequences(
        df, feature_cols, args.target_col, args.date_col, args.seq_len
    )

    print(f"Built {len(X_np)} sequences of length {args.seq_len}")

    # 5) Run predictions in batches
    X = torch.from_numpy(X_np)  # [N, T, F]
    preds = []

    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i : i + batch_size]
            yb = model(xb)  # (B,)
            preds.append(yb.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # 6) Plot predictions vs actual
    print("Plotting predictions vs actual prices...")
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_np, label="Actual Price")
    plt.plot(dates, preds, label="Predicted Price")
    plt.xlabel("Date")
    plt.ylabel(args.target_col)
    plt.title(f"LSTM predictions vs actual {args.target_col} (from {args.start_year})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
