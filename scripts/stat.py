#!/usr/bin/env python
"""
backtest_lstm.py

Evaluate a trained LSTM (lstm_best.pt or checkpoint_*.pt) by simulating
a simple long/flat strategy starting from $1000 per ticker on the TEST split.

Strategy (per ticker, per day):
  - If model's predicted 1-day return > threshold -> go fully long that day
  - Else stay in cash

We compare this to buy-and-hold for the same ticker and compute:
  - Final capital (strategy vs buy-and-hold)
  - Outperformance multiple (strategy_final / buyhold_final)
  - Directional accuracy on days we invest, etc.

Outputs:
  - backtest_per_ticker.csv   (one row per ticker)
  - backtest_summary.txt      (human-readable summary)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import LSTMRegressorWithLN, SequenceConfig  # reuse your model class


# ---------- Utility: same time-based split as train.py ----------

def time_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Time-based split by unique dates. The remainder is test.
    """
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    uniq_dates = sorted(df["date"].dropna().unique())
    n = len(uniq_dates)
    if n < 3:
        raise RuntimeError("Not enough unique dates for train/val/test split.")

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    # Ensure at least 1 date for test
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    train_dates = set(uniq_dates[:n_train])
    val_dates = set(uniq_dates[n_train:n_train + n_val])
    test_dates = set(uniq_dates[n_train + n_val:])

    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    return train_df, val_df, test_df


# ---------- Load merged dataset with same column logic as train.py ----------

def load_merged_df(merged_path: str, emb_dim_cap: int) -> pd.DataFrame:
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Merged dataset not found: {merged_path}")

    emb_cols = [f"emb_{i}" for i in range(emb_dim_cap)]
    base_cols = [
        "ret0",
        "sent_neg",
        "sent_neu",
        "sent_pos",
        "ret_djia",
        "ret_nasdaqcom",
        "ret_sp500",
    ]
    keep_cols = ["date", "ticker", "target_ret1"] + emb_cols + base_cols

    if merged_path.lower().endswith(".parquet"):
        df = pd.read_parquet(merged_path, columns=keep_cols)
    else:
        df = pd.read_csv(merged_path)
        existing = [c for c in keep_cols if c in df.columns]
        df = df[existing].copy()
        missing = [c for c in keep_cols if c not in df.columns]
        for c in missing:
            if c in ("date", "ticker"):
                continue
            df[c] = 0.0

    # normalize date
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "ticker", "target_ret1"]).reset_index(drop=True)

    # don't keep float16 here; cast to float32 for inference stability
    numeric_cols = [c for c in df.columns if c not in ("date", "ticker")]
    for c in numeric_cols:
        df[c] = df[c].astype(np.float32)

    return df


# ---------- Backtest core ----------

def backtest_ticker(
    df_ticker: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    feature_cols,
    lookback: int,
    start_capital: float = 1000.0,
    pred_threshold: float = 0.0,
    batch_size: int = 512,
):
    """
    Backtest a single ticker DataFrame sorted by date.

    Returns a dict with final capital, buy-and-hold capital, stats.
    """
    df_ticker = df_ticker.sort_values("date").reset_index(drop=True)

    if len(df_ticker) <= lookback:
        return None  # not enough data for one sequence

    feats = df_ticker[feature_cols].to_numpy(dtype=np.float32)
    actual_ret = df_ticker["target_ret1"].to_numpy(dtype=np.float32)

    n = len(df_ticker)
    D = feats.shape[1]

    capital = start_capital
    bh_capital = start_capital

    n_trades = 0
    correct_dir = 0
    trade_returns = []

    model.eval()

    # we will iterate in chronological order, building sequences on the fly
    with torch.no_grad():
        idx = lookback
        while idx < n:
            # process a batch of endpoints [idx, idx+batch_size)
            end = min(n, idx + batch_size)
            B = end - idx
            # build (B, lookback, D) tensor
            seqs = np.zeros((B, lookback, D), dtype=np.float32)
            for j in range(B):
                end_idx = idx + j
                start_idx = end_idx - lookback
                seqs[j] = feats[start_idx:end_idx]

            X = torch.from_numpy(seqs).to(device)
            preds = model(X).cpu().numpy().reshape(-1)

            # walk through each day in the batch
            for j in range(B):
                day_idx = idx + j
                pred_r = float(preds[j])
                real_r = float(actual_ret[day_idx])

                # buy-and-hold always invested
                bh_capital *= (1.0 + real_r)

                if pred_r > pred_threshold:
                    # invest for that day
                    capital *= (1.0 + real_r)
                    n_trades += 1
                    trade_returns.append(real_r)
                    if real_r == 0.0:
                        pass
                    elif (pred_r > 0 and real_r > 0) or (pred_r < 0 and real_r < 0):
                        correct_dir += 1
                else:
                    # stay in cash
                    pass

            idx = end

    if n_trades == 0:
        dir_acc = np.nan
        avg_trade_ret = np.nan
    else:
        dir_acc = correct_dir / n_trades
        avg_trade_ret = float(np.mean(trade_returns))

    bh_return = bh_capital / start_capital - 1.0
    strat_return = capital / start_capital - 1.0
    outperf_mult = capital / bh_capital if bh_capital > 0 else np.nan

    return {
        "final_capital": capital,
        "final_capital_buyhold": bh_capital,
        "strategy_return": strat_return,
        "buyhold_return": bh_return,
        "outperformance_multiple": outperf_mult,
        "n_trades": n_trades,
        "directional_accuracy": dir_acc,
        "avg_trade_return": avg_trade_ret,
        "n_days": n - lookback,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--merged_path",
        type=str,
        required=True,
        help="Path to merged_lstm_dataset parquet (same as used in train.py).",
    )
    ap.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt), e.g. models/lstm_ddp_baby/lstm_best.pt",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory where backtest outputs will be written.",
    )
    ap.add_argument(
        "--emb_dim_cap",
        type=int,
        default=128,
        help="Number of FinBERT embedding dims used in training (emb_0..emb_{cap-1}).",
    )
    ap.add_argument(
        "--pred_threshold",
        type=float,
        default=0.0,
        help="Only invest when predicted return > this threshold.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference.",
    )
    ap.add_argument(
        "--start_capital",
        type=float,
        default=1000.0,
        help="Starting capital per ticker.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for inference (per ticker).",
    )

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- device ----------
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}", flush=True)

    # ---------- load df & split ----------
    df = load_merged_df(args.merged_path, args.emb_dim_cap)
    print(f"Loaded merged dataset: {df.shape}", flush=True)

    train_df, val_df, test_df = time_splits(df, train_ratio=0.7, val_ratio=0.15)
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}", flush=True)

    # ---------- load checkpoint & model ----------
    ckpt = torch.load(args.ckpt_path, map_location=device)
    input_dim = ckpt["input_dim"]
    cfg_dict = ckpt.get("cfg", {})
    lookback = int(cfg_dict.get("lookback", 30))

    print(f"Checkpoint loaded from {args.ckpt_path}")
    print(f"input_dim={input_dim}, lookback={lookback}", flush=True)

    # Rebuild model with same architecture as training script
    model = LSTMRegressorWithLN(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # determine feature columns in the same order train.py used
    emb_cols = [f"emb_{i}" for i in range(args.emb_dim_cap)]
    base_cols = [
        "ret0",
        "sent_neg",
        "sent_neu",
        "sent_pos",
        "ret_djia",
        "ret_nasdaqcom",
        "ret_sp500",
    ]
    feature_cols = emb_cols + base_cols

    # sanity check
    missing_feats = [c for c in feature_cols if c not in test_df.columns]
    if missing_feats:
        raise RuntimeError(f"Missing feature columns in test_df: {missing_feats}")

    # ---------- run backtest per ticker ----------
    ticker_stats = []
    for ticker, df_t in test_df.groupby("ticker"):
        stats = backtest_ticker(
            df_ticker=df_t,
            model=model,
            device=device,
            feature_cols=feature_cols,
            lookback=lookback,
            start_capital=args.start_capital,
            pred_threshold=args.pred_threshold,
            batch_size=args.batch_size,
        )
        if stats is None:
            continue
        stats["ticker"] = ticker
        ticker_stats.append(stats)

    if not ticker_stats:
        raise RuntimeError("No tickers had enough test data to backtest.")

    stats_df = pd.DataFrame(ticker_stats)
    stats_df = stats_df[
        [
            "ticker",
            "n_days",
            "n_trades",
            "final_capital",
            "final_capital_buyhold",
            "strategy_return",
            "buyhold_return",
            "outperformance_multiple",
            "directional_accuracy",
            "avg_trade_return",
        ]
    ]

    per_ticker_path = outdir / "backtest_per_ticker.csv"
    stats_df.to_csv(per_ticker_path, index=False)
    print(f"Saved per-ticker backtest stats to {per_ticker_path}", flush=True)

    # ---------- aggregate metrics ----------
    total_tickers = len(stats_df)
    total_trades = int(stats_df["n_trades"].sum())
    # weighted directional accuracy (by trades)
    weighted_dir_acc = (
        np.nansum(stats_df["directional_accuracy"] * stats_df["n_trades"]) /
        max(1, total_trades)
        if total_trades > 0 else np.nan
    )

    # average & median final capital
    avg_final_cap = float(stats_df["final_capital"].mean())
    med_final_cap = float(stats_df["final_capital"].median())

    avg_bh_cap = float(stats_df["final_capital_buyhold"].mean())
    med_bh_cap = float(stats_df["final_capital_buyhold"].median())

    # Median Outperformance Multiple (MOM)
    mom = float(stats_df["outperformance_multiple"].median())
    mean_om = float(stats_df["outperformance_multiple"].mean())

    summary_lines = [
        f"Total tickers backtested: {total_tickers}",
        f"Total trades taken (across all tickers): {total_trades}",
        f"Weighted directional accuracy (on invested days): {weighted_dir_acc:.4f}",
        "",
        f"Average final capital (strategy): ${avg_final_cap:,.2f}",
        f"Median final capital (strategy): ${med_final_cap:,.2f}",
        f"Average final capital (buy-and-hold): ${avg_bh_cap:,.2f}",
        f"Median final capital (buy-and-hold): ${med_bh_cap:,.2f}",
        "",
        f"Median Outperformance Multiple (MOM): {mom:.4f}",
        f"Mean Outperformance Multiple: {mean_om:.4f}",
        "",
        "Interpretation:",
        "  MOM = median(strategy_final / buyhold_final across tickers).",
        "  MOM > 1.0 means the model-driven strategy typically beats buy-and-hold;",
        "  MOM < 1.0 means it underperforms on a typical ticker.",
    ]

    summary_path = outdir / "backtest_summary.txt"
    with open(summary_path, "w") as f:
        for line in summary_lines:
            print(line)
            f.write(line + "\n")

    print(f"\nBacktest summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
