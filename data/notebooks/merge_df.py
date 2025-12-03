#!/usr/bin/env python
"""
Build labeled (headline, price-move) dataset from raw headlines + prices.

- Uses SAME label construction logic as fine_tune.py
- NO model / transformers work: just writes out the merged dataframe.

For each headline:
- Find the last trading day ON OR BEFORE the headline's calendar date
- Attach that day's price row (date, open, high, low, close, adj_close, volume, ticker)
- Attach label (decreasing / stable / increasing) based on future return
"""

import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ---------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------
LABELS = ["decreasing", "stable", "increasing"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}


# ---------------------------------------------------------------------
# 1. Label construction from prices
# ---------------------------------------------------------------------
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

    Returns DataFrame with columns:
        ["date", "label", "ret"] + available price columns:
        ["volume", "open", "high", "low", "close", "adj_close", "ticker"]
    """
    df = price_df.sort_values("date").reset_index(drop=True).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "label", "ret"])

    if "adj_close" not in df.columns:
        raise ValueError("prices_df must contain an 'adj_close' column.")

    # Compute future adj_close and return
    df["target_adj_close"] = df["adj_close"].shift(-horizon_trading_days)
    # Drop the last h rows which don't have a future target
    df = df.iloc[:-horizon_trading_days].copy()

    df["ret"] = (df["target_adj_close"] - df["adj_close"]) / df["adj_close"]

    conditions = [
        df["ret"] < -stable_threshold,
        df["ret"].abs() <= stable_threshold,
    ]
    choices = ["decreasing", "stable"]
    df["label"] = np.select(conditions, choices, default="increasing")

    # Keep date/label/ret + price features from that date
    extra_cols = ["volume", "open", "high", "low", "close", "adj_close", "ticker"]
    cols_to_keep = ["date", "label", "ret"] + [c for c in extra_cols if c in df.columns]

    return df[cols_to_keep]


# ---------------------------------------------------------------------
# 2. Build labeled dataset (parallel over symbols)
# ---------------------------------------------------------------------
def build_labeled_dataset(
    headlines_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizon_trading_days: int = 1,
    stable_threshold: float = 0.005,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Join news headlines with price-direction labels per ticker.

    For each headline, use the last trading day ON OR BEFORE its date
    and take that day's label (future move over `horizon_trading_days`),
    along with that day's price fields.

    Parallelized over symbols using joblib.
    """

    # Parse datetimes once
    headlines_df = headlines_df.copy()
    prices_df = prices_df.copy()

    headlines_df["Date"] = pd.to_datetime(
        headlines_df["Date"].astype(str).str.replace(" UTC", ""),
        errors="coerce",
    )
    prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce")

    # Group once instead of refiltering in each loop iteration
    if "Stock_symbol" not in headlines_df.columns:
        raise ValueError("headlines_df must contain 'Stock_symbol' column.")
    if "ticker" not in prices_df.columns:
        raise ValueError("prices_df must contain 'ticker' column.")

    headline_groups = headlines_df.groupby("Stock_symbol")
    price_groups = prices_df.groupby("ticker")

    headline_syms = set(headline_groups.groups.keys())
    price_syms = set(price_groups.groups.keys())
    symbols: List[str] = sorted(headline_syms & price_syms)

    if not symbols:
        print("No overlapping symbols between headlines and prices.", flush=True)
        return pd.DataFrame(columns=list(headlines_df.columns) + ["label", "ret"])

    print(
        f"Found {len(symbols)} symbols with both headlines and prices. "
        f"Using n_jobs={n_jobs}",
        flush=True,
    )

    # Function to process a single symbol
    def process_symbol(sym: str) -> pd.DataFrame | None:
        hsym = headline_groups.get_group(sym).copy()
        psym = price_groups.get_group(sym).copy()

        if psym.empty or hsym.empty:
            return None

        # Build label dataframe (one row per trading day for this ticker)
        label_df = build_label_df(
            psym,
            horizon_trading_days=horizon_trading_days,
            stable_threshold=stable_threshold,
        )
        if label_df.empty:
            return None

        label_df = label_df.sort_values("date").reset_index(drop=True)

        # Align each headline with the last trading day ON OR BEFORE its calendar date
        label_dates = label_df["date"].dt.normalize().values
        headline_dates = hsym["Date"].dt.normalize().values

        # side="right" -> index of first label_date > headline_date, then -1 => <=
        idx = np.searchsorted(label_dates, headline_dates, side="right") - 1
        valid_mask = idx >= 0
        if not valid_mask.any():
            return None

        hsym_valid = hsym.loc[valid_mask].copy()
        mapped_idx = idx[valid_mask]

        # Columns we want to copy from label_df (per matched trading day)
        price_cols = [
            "date",       # trading date used for the label
            "volume",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "ticker",
            "label",
            "ret",
        ]

        for col in price_cols:
            if col in label_df.columns:
                hsym_valid[col] = label_df[col].values[mapped_idx]

        # If ticker wasn't in label_df for some reason, fall back to symbol
        if "ticker" not in hsym_valid.columns:
            hsym_valid["ticker"] = sym

        return hsym_valid

    print("Starting parallel per-symbol processing...", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_symbol)(sym) for sym in symbols
    )

    non_empty = [r for r in results if r is not None]
    if not non_empty:
        print("All symbol-level results were empty after filtering.", flush=True)
        return pd.DataFrame(columns=list(headlines_df.columns) + ["label", "ret"])

    ds = pd.concat(non_empty, ignore_index=True)
    return ds


# ---------------------------------------------------------------------
# 3. Main CLI entrypoint
# ---------------------------------------------------------------------
def main(args):
    t0 = time.time()

    # ----------------- Read inputs -----------------
    print(f"Reading headlines from {args.headlines_csv}")
    headlines_df = pd.read_csv(
        args.headlines_csv,
        low_memory=False,
        nrows=args.sample_headlines if args.sample_headlines > 0 else None,
    )
    print(f"Loaded headlines: {headlines_df.shape}", flush=True)

    print(f"Reading prices from {args.prices_csv}")
    prices_df = pd.read_csv(
        args.prices_csv,
        low_memory=False,
        nrows=args.sample_prices if args.sample_prices > 0 else None,
    )
    print(f"Loaded prices: {prices_df.shape}", flush=True)

    # ----------------- Build labeled dataset -----------------
    print("Building labeled dataset...", flush=True)
    t1 = time.time()
    df = build_labeled_dataset(
        headlines_df,
        prices_df,
        horizon_trading_days=args.horizon_days,
        stable_threshold=args.stable_threshold,
        n_jobs=args.n_jobs,
    )
    t2 = time.time()
    print(f"Labeled dataset shape before cleanup: {df.shape}")
    print(f"Labeling + join took {t2 - t1:.1f} seconds", flush=True)

    if df.empty:
        print("No labeled rows produced; exiting without writing file.", flush=True)
        return

    # ----------------- Cleanup & label ids -----------------
    # Drop rows missing title or label
    if "Article_title" not in df.columns:
        raise ValueError("Expected 'Article_title' column in headlines_df.")

    df = df.dropna(subset=["Article_title", "label"]).copy()
    df = df[df["label"].isin(LABEL2ID.keys())].copy()
    df["label_id"] = df["label"].map(LABEL2ID)

    print(f"Final labeled dataset shape: {df.shape}")
    print("Label distribution:")
    print(df["label"].value_counts(), flush=True)

    # Peek at a few rows
    sample_cols = [
        col for col in [
            "Date",
            "date",
            "Stock_symbol",
            "ticker",
            "Article_title",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "label",
            "ret",
            "label_id",
        ] if col in df.columns
    ]

    print("\nSample rows:")
    print(df[sample_cols].head(10))

    # ----------------- Save to disk -----------------
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    ext = os.path.splitext(args.output_path)[1].lower()

    if ext == ".parquet":
        df.to_parquet(args.output_path, index=False)
    else:
        # default to CSV for any other extension
        df.to_csv(args.output_path, index=False)

    print(f"\nSaved labeled dataset to {args.output_path}")
    print(f"Total runtime: {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute labeled (headline → price direction) dataset."
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
        help="Path to prices CSV (e.g., processed_stock_prices.csv).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="labeled_headlines.parquet",
        help="Where to save merged dataset (.parquet or .csv).",
    )
    parser.add_argument(
        "--horizon_days",
        type=int,
        default=1,
        help="Number of TRADING days ahead for return calculation.",
    )
    parser.add_argument(
        "--stable_threshold",
        type=float,
        default=0.005,
        help="Abs(return) below this is 'stable' (0.005 = ±0.5%%).",
    )
    parser.add_argument(
        "--sample_headlines",
        type=int,
        default=0,
        help="If > 0, only read this many headline rows (for quick debug).",
    )
    parser.add_argument(
        "--sample_prices",
        type=int,
        default=0,
        help="If > 0, only read this many price rows (for quick debug).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for symbol-level processing (joblib).",
    )

    args = parser.parse_args()
    main(args)
