import argparse
import os
import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")   # headless for NERSC
import matplotlib.pyplot as plt


# -----------------------------
# Model definition
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
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.ln(last)
        return self.mlp(last).squeeze(-1)


# -----------------------------
# Build windows & prediction dataset
# -----------------------------
def build_windows_with_returns(df, feature_cols, target_col, date_col, ticker_col, lookback):
    """
    For each ticker, build rolling windows of length `lookback`.
    Returns a DataFrame with columns:
      date        : prediction date (window end)
      next_date   : date whose return we realize
      ticker
      real_ret    : actual next-day return (from target_col)
      features    : we'll pass separately as X array
    """
    tickers = df[ticker_col].unique()
    X_list = []
    rows = []

    for tic in tickers:
        df_t = df[df[ticker_col] == tic].sort_values(date_col).reset_index(drop=True)
        if len(df_t) <= lookback:
            continue

        values = df_t[feature_cols].to_numpy(dtype=np.float32)
        rets = df_t[target_col].to_numpy(dtype=np.float32)
        dates = df_t[date_col].to_numpy()

        # window ends at i, realized return is at index i (ret from i->i+1),
        # so we also record next_date = dates[i+1]
        for i in range(lookback - 1, len(df_t) - 1):
            window = values[i - lookback + 1 : i + 1]  # (lookback, F)
            X_list.append(window)
            rows.append(
                {
                    "date": dates[i],            # prediction made at end of this day
                    "next_date": dates[i + 1],   # realized on next day
                    "ticker": tic,
                    "real_ret": rets[i],         # actual (target_ret1)
                }
            )

    if not X_list:
        raise ValueError("No windows created; check lookback / data coverage.")

    X_np = np.stack(X_list, axis=0)
    meta_df = pd.DataFrame(rows)
    return X_np, meta_df


# -----------------------------
# Backtest using returns only
# -----------------------------
def backtest_with_returns(preds_df, initial_capital=5000.0, top_k=10):
    """
    preds_df columns:
      date       : prediction date
      next_date  : date when return is realized
      ticker
      pred_ret   : model-predicted return
      real_ret   : true next-day return (clipped)
    """

    # Sort by prediction date
    preds_df = preds_df.sort_values(["date", "ticker"])
    trade_dates = sorted(preds_df["date"].unique())
    if len(trade_dates) < 2:
        raise ValueError("Not enough trade dates for backtest.")

    # ---- Buy & Hold weights (equal-weight on first prediction date universe) ----
    first_date = trade_dates[0]
    first_universe = preds_df.loc[preds_df["date"] == first_date, "ticker"].unique()
    n_assets = len(first_universe)
    if n_assets == 0:
        raise ValueError("No tickers on first trade date for buy & hold.")

    w_bh = {tic: 1.0 / n_assets for tic in first_universe}

    # ---- Initialize capitals ----
    cap_bh = initial_capital
    cap_strat = initial_capital

    dates_plot = []
    bh_curve = []
    strat_curve = []

    for d in trade_dates:
        day = preds_df[preds_df["date"] == d]

        # Map ticker -> real_ret and pred_ret
        real_map = dict(zip(day["ticker"], day["real_ret"]))
        pred_map = dict(zip(day["ticker"], day["pred_ret"]))

        # ---- Buy & Hold daily return ----
        r_bh = 0.0
        # normalize weights over tickers that actually have returns this day
        active_tics = [tic for tic in w_bh.keys() if tic in real_map]
        if active_tics:
            norm = sum(w_bh[tic] for tic in active_tics)
            for tic in active_tics:
                wt = w_bh[tic] / norm
                r_bh += wt * real_map[tic]
        # else, r_bh stays 0

        cap_bh *= (1.0 + r_bh)

        # ---- LSTM strategy daily return ----
        # Select top_k tickers by predicted return, only if pred_ret > 0
        day_sorted = day.sort_values("pred_ret", ascending=False)
        day_sorted = day_sorted[day_sorted["pred_ret"] > 0]
        if top_k is not None:
            day_sorted = day_sorted.head(top_k)

        r_strat = 0.0
        for _, row in day_sorted.iterrows():
            r_strat += 0.01 * row["real_ret"]  # 1% of capital into each

        cap_strat *= (1.0 + r_strat)

        # Record using the *next_date* (when return is realized) for x-axis
        next_date = day["next_date"].iloc[0]
        dates_plot.append(next_date)
        bh_curve.append(cap_bh)
        strat_curve.append(cap_strat)

    return np.array(dates_plot), np.array(bh_curve), np.array(strat_curve)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model_path",
        type=str,
        default="../models/lstm_ddp_baby/lstm_best.pt",
        help="Path to LSTM checkpoint (relative to this script)",
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
        help="Name of date column",
    )
    ap.add_argument(
        "--ticker_col",
        type=str,
        default="ticker",
        help="Name of ticker column",
    )
    ap.add_argument(
        "--start_year",
        type=int,
        default=2015,
        help="Backtest start year",
    )
    ap.add_argument(
        "--end_year",
        type=int,
        default=2025,
        help="Backtest end year (inclusive)",
    )
    ap.add_argument(
        "--output_png",
        type=str,
        default="lstm_strategy_vs_buyhold.png",
        help="Output PNG filename",
    )
    args = ap.parse_args()

    # ---- Load checkpoint ----
    ckpt = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    cfg = ckpt.get("cfg", {})

    input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
    hidden_dim = state_dict["lstm.weight_hh_l0"].shape[1]
    num_layers = len(
        [k for k in state_dict.keys() if k.startswith("lstm.weight_ih_l") and "reverse" not in k]
    )
    lookback = int(cfg.get("lookback", 10))
    target_col = cfg.get("target_col", "target_ret1")

    print(f"input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    print(f"lookback={lookback}, target_col={target_col}")

    model = TinyLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ---- Load data ----
    if args.data_path.endswith(".parquet") or args.data_path.endswith(".pq"):
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path)

    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.sort_values(args.date_col)

    start_date = dt.datetime(args.start_year, 1, 1)
    end_date = dt.datetime(args.end_year, 12, 31)
    df = df[(df[args.date_col] >= start_date) & (df[args.date_col] <= end_date)].reset_index(drop=True)
    print(f"Filtered rows between {start_date.date()} and {end_date.date()}: {len(df)}")

    for col in [args.ticker_col, target_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe.")

    # ---- Auto feature selection (same as before) ----
    num_df = df.select_dtypes(include=[np.number])
    if target_col not in num_df.columns:
        raise ValueError(f"Target column '{target_col}' must be numeric.")

    num_cols = [c for c in num_df.columns if c != target_col]
    emb_cols = [c for c in num_cols if c.startswith("emb_")]
    non_emb_cols = [c for c in num_cols if not c.startswith("emb_")]

    print(f"Total numeric (excluding target): {len(num_cols)}")
    print(f"  Non-embedding cols: {len(non_emb_cols)}")
    print(f"  Embedding cols:     {len(emb_cols)}")

    if len(non_emb_cols) > input_dim:
        raise ValueError(
            f"{len(non_emb_cols)} non-emb cols > model input_dim={input_dim}; "
            "need to drop some manually."
        )

    needed_from_emb = input_dim - len(non_emb_cols)
    if needed_from_emb > len(emb_cols):
        raise ValueError(
            f"Need {needed_from_emb} emb_* cols but only {len(emb_cols)} available."
        )

    keep_emb = set(emb_cols[:needed_from_emb])
    feature_cols = [
        c for c in num_cols if (not c.startswith("emb_")) or (c in keep_emb)
    ]
    if len(feature_cols) != input_dim:
        raise ValueError(
            f"Feature count {len(feature_cols)} != model input_dim {input_dim}"
        )

    print("First 10 feature cols:", feature_cols[:10])
    print("Last 10 feature cols:", feature_cols[-10:])

    # ---- Build windows & meta ----
    X_np, meta_df = build_windows_with_returns(
        df, feature_cols, target_col, args.date_col, args.ticker_col, lookback
    )
    print(f"Built windows: X shape = {X_np.shape}, meta rows = {len(meta_df)}")

    # ---- Run model predictions ----
    X = torch.from_numpy(X_np)
    preds = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i : i + batch_size]
            yb = model(xb)
            preds.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    # ---- Assemble prediction dataframe ----
    preds_df = meta_df.copy()
    preds_df["pred_ret"] = preds

    # Clip realized returns to avoid insane outliers from bad data
    preds_df["real_ret"] = np.clip(preds_df["real_ret"], -1.0, 1.0)

    print("Sample of preds_df:")
    print(preds_df.head())

    # ---- Backtest ----
    dates, bh_curve, strat_curve = backtest_with_returns(preds_df, initial_capital=5000.0, top_k=10)

    print("First 5 dates:", dates[:5])
    print("First 5 Buy&Hold values:", bh_curve[:5])
    print("First 5 Strategy values:", strat_curve[:5])

    # ---- Plot ----
    plt.figure(figsize=(14, 6))
    plt.plot(dates, bh_curve, label="Buy & Hold (equal-weight)", linewidth=1.5)
    plt.plot(dates, strat_curve, label="LSTM Strategy (top 10, 1% each)", linewidth=1.5)
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.title("LSTM Strategy vs Buy & Hold (starting at $5000)")
    plt.legend()
    plt.tight_layout()

    out_path = args.output_png
    plt.savefig(out_path, dpi=200)
    print("Saved equity curve plot to:", os.path.abspath(out_path))


if __name__ == "__main__":
    main()
