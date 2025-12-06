# train.py
import os
import math
import argparse
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset import StockNewsDataset
from model import HybridFinBERTModel


# ---------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------
def setup_ddp():
    """
    Initialize torch.distributed using env variables set by torchrun / srun.
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # LOCAL_RANK is set by torchrun; on Perlmutter with srun+torchrun this is correct
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--labeled_path", type=str, required=True,
                    help="Path to labeled_headlines parquet/csv")
    ap.add_argument("--prices_path", type=str, required=True,
                    help="Path to processed_stock_prices.csv")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--min_abs_ret", type=float, default=None)

    ap.add_argument("--finbert_model_name", type=str, default="ProsusAI/finbert")
    ap.add_argument("--freeze_finbert", action="store_true", default=True)

    ap.add_argument("--output_dir", type=str, default="./checkpoints")
    ap.add_argument("--log_every", type=int, default=50)

    # debug flag to kill dataloader multiprocessing if needed
    ap.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader num_workers (set to 0 if you see hangs)")

    args = ap.parse_args()
    return args


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[Rank 0] Using world_size={world_size}")
        print(f"[Rank 0] Saving checkpoints to {args.output_dir}")

    # ---------------- Dataset & DataLoader ----------------
    if rank == 0:
        print("[Rank 0] Loading dataset...")

    # Important: dataset __init__ MUST be cheap; see notes below
    dataset = StockNewsDataset(
        labeled_path=args.labeled_path,
        prices_path=args.prices_path,
        finbert_model_name=args.finbert_model_name,
        seq_len=args.seq_len,
        max_length=args.max_length,
        min_abs_ret=args.min_abs_ret,
    )

    # Sanity check that we actually got past __init__
    if rank == 0:
        print(f"[Rank 0] Dataset loaded, len={len(dataset)}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ---------------- Model, optimizer, scaler ----------------
    if rank == 0:
        print("[Rank 0] Building model...")

    model = HybridFinBERTModel(
        finbert_model_name=args.finbert_model_name,
        price_num_features=6,  # [open, high, low, close, adj_close, volume]
        freeze_finbert=args.freeze_finbert,
    )
    model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler()

    # ---------------- Training loop ----------------
    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        if rank == 0:
            print(f"\n[Rank 0] Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            price_seq = batch["price_seq"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                preds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    price_seq=price_seq,
                )
                loss = torch.nn.functional.mse_loss(preds, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            if rank == 0 and global_step % args.log_every == 0:
                with torch.no_grad():
                    mae = (preds - target).abs().mean().item()
                print(
                    f"[Rank 0] step={global_step} "
                    f"loss={loss.item():.6f} "
                    f"mae={mae:.6f} "
                    f"batch_size={target.size(0)}"
                )

        # -------- Save checkpoint (rank 0 only) --------
        if rank == 0:
            ckpt_path = os.path.join(
                args.output_dir,
                f"hybrid_epoch{epoch+1}_step{global_step}.pt",
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[Rank 0] Saved checkpoint to {ckpt_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
