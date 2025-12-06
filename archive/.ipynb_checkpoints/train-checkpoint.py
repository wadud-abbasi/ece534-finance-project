"""
train.py — uses Data/data/{processed_stock_prices.csv, processed_headlines_subset.csv, processed_indexes.csv}

Usage (single node, 4 GPUs):
  torchrun --nproc_per_node=4 train.py --outdir runs/run1 --amp

Multi-node: torchrun will read env (MASTER_ADDR/PORT, NODE_RANK, WORLD_SIZE).
"""

import argparse, os, warnings, random, math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import FinBertEmbedder, SequenceConfig, NewsPriceSequenceDataset, LSTMRegressor

# ----------------- utils -----------------
def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ddp_setup():
    """Initialize torch.distributed if launched with torchrun/srun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = int(os.environ.get("RANK", "0")) == 0
        return True, is_main, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, True, device

# ----------------- loaders for your CSVs -----------------
def load_prices(path: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected columns from your sample
    # date, volume, open, high, low, close, adj_close, ticker
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if tickers: df = df[df["ticker"].isin(tickers)].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    # same-day and next-day log returns
    df["log_close"] = np.log(df["close"].clip(1e-6))
    df["ret0"] = df.groupby("ticker")["log_close"].diff().fillna(0.0)
    df["target_ret1"] = df.groupby("ticker")["ret0"].shift(-1)
    return df

def load_headlines(path: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # columns: Date, Article_title, Stock_symbol
    df = df.rename(columns={
        "Date": "date",
        "Article_title": "headline",
        "Stock_symbol": "ticker",
    })
    # parse to date (drop time, keep UTC day)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()
    df = df.dropna(subset=["date", "headline", "ticker"])
    if tickers: df = df[df["ticker"].isin(tickers)]
    return df

def load_indexes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # columns: date, djia, nasdaqcom, sp500
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date")
    # forward-fill for holidays, etc.
    df[["djia","nasdaqcom","sp500"]] = df[["djia","nasdaqcom","sp500"]].ffill()
    # log returns as features (align to same-day)
    for c in ["djia","nasdaqcom","sp500"]:
        df[f"ret_{c}"] = np.log(df[c].clip(1e-6)).diff().fillna(0.0)
    return df[["date","ret_djia","ret_nasdaqcom","ret_sp500"]]

# ----------------- FinBERT daily embedding -----------------
def compute_daily_news_embeddings(news_df: pd.DataFrame, amp: bool = True, max_len: int = 128, batch_size: int = 64) -> pd.DataFrame:
    """Embed each headline and average per (date, ticker)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fb = FinBertEmbedder(pooling="cls", unfreeze_top_layers=0, device=device)

    texts = news_df["headline"].astype(str).tolist()
    pooled, probs = fb.encode(texts, max_len=max_len, batch_size=batch_size, amp=amp)

    H = pooled.shape[1]
    emb_cols = [f"emb_{i}" for i in range(H)]
    e = pd.DataFrame(pooled, columns=emb_cols)
    s = pd.DataFrame(probs, columns=["sent_neg","sent_neu","sent_pos"])
    out = pd.concat([news_df[["date","ticker"]].reset_index(drop=True), e, s], axis=1)

    # daily average per (date, ticker)
    grouped = out.groupby(["date","ticker"], as_index=False).mean()
    return grouped

# ----------------- merge & split -----------------
def merge_all(prices: pd.DataFrame, daily_news: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """Join prices with daily FinBERT features and index returns."""
    df = prices.merge(daily_news, on=["date","ticker"], how="left")
    # news ffill by ticker (use last known embedding if no news that day)
    emb = [c for c in df if c.startswith("emb_")]
    snt = [c for c in ["sent_neg","sent_neu","sent_pos"] if c in df]
    if emb + snt:
        df[emb + snt] = df.groupby("ticker")[emb + snt].apply(lambda g: g.ffill()).reset_index(level=0, drop=True)
        df[emb + snt] = df[emb + snt].fillna(0.0)

    df = df.merge(idx, on="date", how="left")
    for c in ["ret_djia","ret_nasdaqcom","ret_sp500"]:
        if c in df: df[c] = df[c].ffill().fillna(0.0)

    return df

def time_splits(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    df = df.sort_values(["date","ticker"]).reset_index(drop=True)
    uniq_dates = df["date"].drop_duplicates().sort_values().to_list()
    n = len(uniq_dates)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    d_train_end = uniq_dates[n_train - 1]
    d_val_end = uniq_dates[n_train + n_val - 1]

    train = df[df["date"] <= d_train_end]
    val = df[(df["date"] > d_train_end) & (df["date"] <= d_val_end)]
    test = df[df["date"] > d_val_end]
    return train, val, test

# ----------------- training -----------------
def train_lstm(train_df, val_df, cfg: SequenceConfig, outdir: Path, epochs=10, batch_size=64, lr=1e-3, num_workers=4, amp=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_ds = NewsPriceSequenceDataset(train_df, cfg)
    va_ds = NewsPriceSequenceDataset(val_df, cfg)

    def feat_dim(df):
        return len([c for c in df if c.startswith("emb_")] +
                   [c for c in ["ret0","sent_pos","sent_neu","sent_neg","ret_djia","ret_nasdaqcom","ret_sp500"] if c in df])

    input_dim = feat_dim(train_df)
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = LSTMRegressor(input_dim=input_dim, hidden_dim=256, num_layers=2, dropout=0.2, horizon=cfg.horizon).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    best, best_path = float("inf"), outdir / "lstm_best.pt"
    outdir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train(); tr_loss = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                pred = model(xb).squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr.dataset)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += torch.nn.functional.mse_loss(model(xb).squeeze(-1), yb).item() * xb.size(0)
        va_loss /= len(va.dataset)

        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[LSTM] epoch {ep:03d}  train {tr_loss:.6f}  val {va_loss:.6f}  (best {best:.6f})")

    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    return model, best_path

@torch.no_grad()
def predict(model: torch.nn.Module, df: pd.DataFrame, cfg: SequenceConfig, batch_size=256) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    ds = NewsPriceSequenceDataset(df, cfg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, ys = [], []
    for xb, yb in dl:
        preds.append(model(xb.to(device)).squeeze(-1).cpu().numpy())
        ys.append(yb.squeeze(-1).numpy())
    return np.concatenate(preds), np.concatenate(ys)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news_csv", default="Data/data/processed_headlines_subset.csv")
    ap.add_argument("--prices_csv", default="Data/data/processed_stock_prices.csv")
    ap.add_argument("--indexes_csv", default="Data/data/processed_indexes.csv")
    ap.add_argument("--tickers", type=str, default=None, help="comma list filter, e.g. A,MSFT,^IXIC")

    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", required=True)

    args = ap.parse_args()
    set_seed(args.seed)
    _, is_main, device = ddp_setup()

    # --- load data in your schema ---
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    prices = load_prices(args.prices_csv, tickers)
    news = load_headlines(args.news_csv, tickers)
    idx = load_indexes(args.indexes_csv)

    # --- FinBERT embeddings (do this on rank0 only to avoid duplication) ---
    if is_main:
        print("Embedding headlines with FinBERT...")
        daily_news = compute_daily_news_embeddings(news, amp=args.amp, max_len=128, batch_size=64)
    else:
        daily_news = None

    # broadcast to all ranks
    if torch.distributed.is_initialized():
        objs = [daily_news] if is_main else [None]
        torch.distributed.broadcast_object_list(objs, src=0)
        daily_news = objs[0]

    # --- merge & split ---
    merged = merge_all(prices, daily_news, idx)
    train_df, val_df, test_df = time_splits(merged, 0.7, 0.15)
    cfg = SequenceConfig(lookback=args.lookback, horizon=args.horizon, target_col="target_ret1")

    # --- train LSTM ---
    if is_main: print("Training LSTM...")
    outdir = Path(args.outdir)
    model, ckpt = train_lstm(train_df, val_df, cfg, outdir, epochs=args.epochs,
                             batch_size=args.batch_size, lr=args.lr,
                             num_workers=args.num_workers, amp=args.amp)

    # --- eval ---
    pred_val, y_val = predict(model, val_df, cfg)
    pred_test, y_test = predict(model, test_df, cfg)

    def mse(a,b): return float(np.mean((a-b)**2))
    def mae(a,b): return float(np.mean(np.abs(a-b)))

    if is_main:
        metrics = dict(
            lstm_val_mse=mse(y_val, pred_val),
            lstm_val_mae=mae(y_val, pred_val),
            lstm_test_mse=mse(y_test, pred_test),
            lstm_test_mae=mae(y_test, pred_test),
        )
        print("Metrics:", metrics)
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(Path(args.outdir) / "metrics.csv", index=False)
        print("Done. Artifacts:", args.outdir)

if __name__ == "__main__":
    main()
