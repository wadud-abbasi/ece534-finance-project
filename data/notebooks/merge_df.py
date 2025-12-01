#!/usr/bin/env python
"""
Build labeled (headline, price-move) dataset from raw headlines + prices.

This uses the SAME label construction logic as fine_tune.py but does
NO model / transformers work. It just writes out the merged dataframe.
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd

# Same ordered label list as in fine_tune.py
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

    Returns DataFrame with columns: ["date", "label", "ret"].
    """
    df = price_df.sort_values("date").reset_index(drop=True).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "label", "ret"])

    df["target_adj_close"] = df["adj_close"].shift(-horizon_trading_days)
    df = df.iloc[:-horizon_trading_days].copy()

    df["ret"] = (df["target_adj_close"] - df["adj_close"]) / df["adj_close"]

    conditions = [
        df["ret"] < -stable_threshold,
        df["ret"].abs() <= stable_threshold,
    ]
    choices = ["decreasing", "stable"]
    df["label"] = np.select(conditions, choices, default="increasing")

    return df[["date", "label", "ret"]]


def build_labeled_dataset(
    headlines_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizon_trading_days: int = 1,
    stable_threshold: float = 0.005,
) -> pd.DataFrame:
    """
    Join news headlines with price-direction labels per ticker.

    For each headline, use the last trading day STRICTLY BEFORE its date
    and take that day's label (future move over `horizon_trading_days`).
    """

    # Parse datetimes
    headlines_df["Date"] = pd.to_datetime(
        headlines_df["Date"].astype(str).str.replace(" UTC", ""),
        errors="coerce",
    )
    prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce")

    all_records = []

    for sym in sorted(headlines_df["Stock_symbol"].dropna().unique()):
        hsym = headlines_df[headlines_df["Stock_symbol"] == sym].copy()
        psym = prices_df[prices_df["ticker"] == sym].copy()

        if psym.empty:
            continue

        label_df = build_label_df(
            psym,
            horizon_trading_days=horizon_trading_days,
            stable_threshold=stable_threshold,
        )
        label_df = label_df.sort_values("date").reset_index(drop=True)
        if label_df.empty:
            continue

        label_dates = label_df["date"].dt.normalize().values
        headline_dates = hsym["Date"].dt.normalize().values

        # Last trading day strictly before headline date
        idx = np.searchsorted(label_dates, headline_dates, side="left") - 1
        valid_mask = idx >= 0
        if not valid_mask.any():
            continue

        hsym_valid = hsym.loc[valid_mask].copy()
        mapped_idx = idx[valid_mask]

        hsym_valid["label"] = label_df["label"].values[mapped_idx]
        hsym_valid["ret"] = label_df["ret"].values[mapped_idx]

        all_records.append(hsym_valid)

    if not all_records:
        return pd.DataFrame(
            columns=list(headlines_df.columns) + ["label", "ret"]
        )

    ds = pd.concat(all_records, ignore_index=True)
    return ds


# ---------------------------------------------------------------------
# 2. Main
# ---------------------------------------------------------------------
def main(args):
    t0 = time.time()
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

    print("Building labeled dataset...", flush=True)
    t1 = time.time()
    df = build_labeled_dataset(
        headlines_df,
        prices_df,
        horizon_trading_days=args.horizon_days,
        stable_threshold=args.stable_threshold,
    )
    t2 = time.time()
    print(f"Labeled dataset shape before cleanup: {df.shape}")
    print(f"Labeling + join took {t2 - t1:.1f} seconds", flush=True)

    # Drop bad rows, map labels to IDs
    df = df.dropna(subset=["Article_title", "label"]).copy()
    df = df[df["label"].isin(LABEL2ID.keys())].copy()
    df["label_id"] = df["label"].map(LABEL2ID)

    print(f"Final labeled dataset shape: {df.shape}")
    print("Label distribution:")
    print(df["label"].value_counts(), flush=True)

    # Peek at a few rows
    print("\nSample rows:")
    print(df[["Date", "Stock_symbol", "Article_title", "label", "ret"]].head(10))

    # Save to disk
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

    args = parser.parse_args()
    main(args)
