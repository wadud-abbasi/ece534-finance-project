import argparse
import os
import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")   # for headless environments (NERSC)
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
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]      # (B, hidden_dim)
        last = self.ln(last)
        return self.mlp(last).squeeze(-1)   # (B,)


# -----------------------------
# Helper: safe price lookup
# -----------------------------
def get_price(price_series, date, ticker):
    """Safe lookup: returns np.nan if missing."""
    try:
        return float(price_series.loc[(date, ticker)])
    except KeyError:
        return np.nan


# -----------------------------
# Backtest: buy & hold vs LSTM strategy
# -----------------------------
def backtest_strategies(preds_df,
                        price_series,
                        initial_capital=5000.0,
                        top_k=10):
    """
    preds_df columns: ['date', 'ticker', 'pred_ret', 'price_today', 'price_next']
    price_series: Series indexed by (date, ticker) -> price (adj_close)

    Returns:
      dates: np.array of dates
      bh_curve: np.array of buy & hold equity
      strat_curve: np.array of LSTM strategy equity
    """
    trade_dates = sorted(preds_df["date"].unique())
    if len(trade_dates) < 2:
        raise ValueError("Not enough trade dates for backtest.")

    # -------- BUY & HOLD setup --------
    first_date = trade_dates[0]
    first_universe = preds_df.loc[preds_df["date"] == first_date, "ticker"].unique()
    n_assets = len(first_universe)
    if n_assets == 0:
        raise ValueError("No tickers on first trade date for buy & hold.")

    bh_shares = {}
    bh_cash = 0.0

    equal_invest = initial_capital / n_assets
    for tic in first_universe:
        p0 = get_price(price_series, first_date, tic)
        if np.isnan(p0) or p0 <= 0:
            continue
        bh_shares[tic] = equal_invest / p0

    # Value buy & hold on first date (should be ~ initial_capital)
    bh_equity0 = bh_cash
    for tic, sh in bh_shares.items():
        p0 = get_price(price_series, first_date, tic)
        if np.isnan(p0):
            continue
        bh_equity0 += sh * p0

    # -------- LSTM strategy setup --------
    strat_cash = initial_capital
    strat_shares = {}

    strat_equity0 = strat_cash  # no positions yet

    # Record curves (start at first date with 5000)
    curve_dates = [first_date]
    bh_equity_curve = [bh_equity0]
    strat_equity_curve = [strat_equity0]

    # Iterate through all days where we have predictions (today) and a next day
    for i in range(len(trade_dates) - 1):
        d = trade_dates[i]
        next_d = trade_dates[i + 1]

        day_preds = preds_df[preds_df["date"] == d].copy()
        if day_preds.empty:
            continue

        # ----- Buy & Hold: value at next day -----
        bh_value_next = bh_cash
        for tic, sh in bh_shares.items():
            if sh <= 0:
                continue
            p_next = get_price(price_series, next_d, tic)
            if np.isnan(p_next):
                continue
            bh_value_next += sh * p_next

        # ----- LSTM Strategy -----
        # 1) Value at current date (before trades)
        strat_value_d = strat_cash
        for tic, sh in strat_shares.items():
            if sh <= 0:
                continue
            p_d = get_price(price_series, d, tic)
            if np.isnan(p_d):
                continue
            strat_value_d += sh * p_d

        # 2) SELL positions with predicted_ret < 0
        day_pred_map = dict(zip(day_preds["ticker"], day_preds["pred_ret"]))
        to_remove = []
        for tic, sh in strat_shares.items():
            if sh <= 0:
                continue
            pred_r = day_pred_map.get(tic, None)
            if pred_r is None:
                continue
            if pred_r < 0:
                p_d = get_price(price_series, d, tic)
                if np.isnan(p_d):
                    continue
                strat_cash += sh * p_d
                to_remove.append(tic)
        for tic in to_remove:
            strat_shares[tic] = 0.0

        # 3) Recompute equity after sells
        strat_value_d_after_sells = strat_cash
        for tic, sh in strat_shares.items():
            if sh <= 0:
                continue
            p_d = get_price(price_series, d, tic)
            if np.isnan(p_d):
                continue
            strat_value_d_after_sells += sh * p_d

        # 4) BUY top-k predicted tickers: 1% of current equity each
        day_preds_sorted = day_preds.sort_values("pred_ret", ascending=False)
        top = day_preds_sorted.head(top_k)

        for _, row in top.iterrows():
            tic = row["ticker"]
            p_d = row["price_today"]
            if p_d <= 0 or np.isnan(p_d):
                continue

            equity_now = strat_value_d_after_sells
            invest_amt = 0.01 * equity_now

            if invest_amt <= 0 or strat_cash <= 0:
                break

            invest_amt = min(invest_amt, strat_cash)
            sh_buy = invest_amt / p_d
            strat_cash -= invest_amt
            strat_shares[tic] = strat_shares.get(tic, 0.0) + sh_buy

        # 5) Value strategy at next day prices
        strat_value_next = strat_cash
        for tic, sh in strat_shares.items():
            if sh <= 0:
                continue
            p_next = get_price(price_series, next_d, tic)
            if np.isnan(p_next):
                continue
            strat_value_next += sh * p_next

        curve_dates.append(next_d)
        bh_equity_curve.append(bh_value_next)
        strat_equity_curve.append(strat_value_next)

    return np.array(curve_dates), np.array(bh_equity_curve), np.array(strat_equity_curve)


# -----------------------------
# Main script
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model_path",
        type=str,
        default="../models/lstm_ddp_baby/lstm_best.pt",
        help="Path to LSTM checkpoint",
    )
    ap.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .parquet or .csv with features",
    )
    ap.add_argument(
        "--date_col",
        type=str,
        default="date",
        help="Date column name",
    )
    ap.add_argument(
        "--ticker_col",
        type=str,
        default="ticker",
        help="Ticker / symbol column name",
    )
    ap.add_argument(
        "--price_col",
        type=str,
        default="adj_close",
        help="Price column used for valuation",
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

    # ---------- Load checkpoint & model ----------
    ckpt = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    cfg = ckpt.get("cfg", {})

    input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
    hidden_dim = state_dict["lstm.weight_hh_l0"].shape[1]
    num_layers = len(
        [k for k in state_dict.keys()
         if k.startswith("lstm.weight_ih_l") and "reverse" not in k]
    )
    lookback = int(cfg.get("lookback", 10))
    target_col = cfg.get("target_col", "target_ret1")

    print(f"input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    print(f"lookback={lookback}, target_col={target_col}")

    model = TinyLSTMModel(input_dim=input_dim,
                          hidden_dim=hidden_dim,
                          num_layers=num_layers)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ---------- Load data ----------
    if args.data_path.endswith(".parquet") or args.data_path.endswith(".pq"):
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path)

    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.sort_values(args.date_col)
    start_date = dt.datetime(args.start_year, 1, 1)
    end_date = dt.datetime(args.end_year, 12, 31)
    df = df[(df[args.date_col] >= start_date) &
            (df[args.date_col] <= end_date)].reset_index(drop=True)

    print(f"Filtered rows between {start_date.date()} and {end_date.date()}: {len(df)}")

    for col in [args.ticker_col, args.price_col, target_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe.")

    # ---------- Auto feature selection ----------
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
            f"Need {needed_from_emb} emb_* columns but only {len(emb_cols)} available."
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

    # ---------- Build per-ticker windows & prediction dataset ----------
    tickers = df[args.ticker_col].unique()
    print(f"Found {len(tickers)} tickers.")

    X_list = []
    rec_ticker = []
    rec_date = []
    rec_price_today = []
    rec_price_next = []

    for tic in tickers:
        df_t = df[df[args.ticker_col] == tic].sort_values(args.date_col).reset_index(drop=True)
        if len(df_t) <= lookback:
            continue

        values = df_t[feature_cols].to_numpy(dtype=np.float32)
        prices = df_t[args.price_col].to_numpy(dtype=np.float32)
        dates = df_t[args.date_col].to_numpy()

        # window ending at i (today), realized next-day return using i+1
        for i in range(lookback - 1, len(df_t) - 1):
            window = values[i - lookback + 1 : i + 1]  # (lookback, F)
            X_list.append(window)
            rec_ticker.append(tic)
            rec_date.append(dates[i])
            rec_price_today.append(prices[i])
            rec_price_next.append(prices[i + 1])

    if not X_list:
        raise ValueError("No windows were created; check lookback and data coverage.")

    X_np = np.stack(X_list, axis=0)
    print(f"Built windows: X shape = {X_np.shape}")

    # ---------- Run model predictions ----------
    X = torch.from_numpy(X_np)
    preds = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i : i + batch_size]
            yb = model(xb)
            preds.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    preds_df = pd.DataFrame({
        "date": rec_date,
        "ticker": rec_ticker,
        "pred_ret": preds,
        "price_today": rec_price_today,
        "price_next": rec_price_next,
    })

    # ---------- Build price lookup series ----------
    price_df = df[[args.date_col, args.ticker_col, args.price_col]].drop_duplicates()
    price_series = price_df.set_index([args.date_col, args.ticker_col])[args.price_col]

    # ---------- Backtest ----------
    dates, bh_curve, strat_curve = backtest_strategies(
        preds_df, price_series, initial_capital=5000.0, top_k=10
    )

    print("First 5 dates:", dates[:5])
    print("First 5 Buy&Hold values:", bh_curve[:5])
    print("First 5 Strategy values:", strat_curve[:5])

    # ---------- Plot ----------
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
