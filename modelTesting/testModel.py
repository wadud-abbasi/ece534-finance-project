import argparse
import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.ln(last)
        return self.mlp(last).squeeze(-1)


def build_sequences(df, feature_cols, target_col, date_col, seq_len):
    values = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(df[date_col]).to_numpy()

    n = len(df)
    if n < seq_len:
        raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}")

    X_list, y_list, d_list = [], [], []

    for i in range(n - seq_len + 1):
        j = i + seq_len - 1
        X_list.append(values[i : i + seq_len])
        y_list.append(targets[j])
        d_list.append(dates[j])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    d = np.array(d_list)

    return d, X, y


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model_path",
        type=str,
        default="models/lstm_ddp_baby/lstm_best.pt",
    )
    ap.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to your .parquet (or .csv) file",
    )
    ap.add_argument(
        "--date_col",
        type=str,
        default="date",
        help="Date column name in parquet",
    )
    ap.add_argument(
        "--target_col",
        type=str,
        default="adj_close",  # change if needed
        help="Target price column name",
    )
    ap.add_argument(
        "--seq_len",
        type=int,
        default=30,
        help="Sequence length used for training",
    )
    ap.add_argument(
        "--start_year",
        type=int,
        default=2015,
        help="Only use data from this year onward",
    )

    args = ap.parse_args()

    # 1) Load model
    ckpt = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
    print("Inferred input_dim:", input_dim)

    model = TinyLSTMModel(input_dim=input_dim)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    model.eval()

    # 2) Load parquet or csv
    if args.data_path.endswith(".parquet") or args.data_path.endswith(".pq"):
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path)

    # 3) Filter by date >= 2015-01-01
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.sort_values(args.date_col)
    start_date = dt.datetime(args.start_year, 1, 1)
    df = df[df[args.date_col] >= start_date].reset_index(drop=True)
    print(f"Rows from {start_date.date()} onward:", len(df))

    # 4) Determine feature columns (numeric minus target)
    num_df = df.select_dtypes(include=[np.number])
    if args.target_col not in num_df.columns:
        raise ValueError(f"Target col {args.target_col} not numeric / missing in data.")

    feature_cols = [c for c in num_df.columns if c != args.target_col]

    if len(feature_cols) != input_dim:
        raise ValueError(
            f"Model expects {input_dim} features but got {len(feature_cols)}.\n"
            f"Feature cols: {feature_cols}"
        )

    # 5) Build sequences & targets
    dates, X_np, y_np = build_sequences(
        df, feature_cols, args.target_col, args.date_col, args.seq_len
    )
    print("Built sequences:", X_np.shape)

    # 6) Run model to get predictions
    X = torch.from_numpy(X_np)
    preds = []

    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i : i + batch_size]
            yb = model(xb)
            preds.append(yb.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # 7) Plot
    print("Plotting…")
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_np, label="Actual", linewidth=1.5)
    plt.plot(dates, preds, label="Predicted", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel(args.target_col)
    plt.title(f"LSTM predictions vs {args.target_col} (>= {args.start_year})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
