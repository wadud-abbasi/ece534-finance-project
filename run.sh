MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
GPUS_PER_NODE=1

export MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE

srun python -m torch.distributed.run \
  --nproc_per_node=${GPUS_PER_NODE} \
  --nnodes=${NNODES} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
    --labeled_path /global/cfs/cdirs/m4431/sp2160/Data/FNSPID/labeled_headlines.parquet \
    --prices_path /global/cfs/cdirs/m4431/sp2160/Data/FNSPID/processed_stock_prices.csv \
    --epochs 1 \
    --batch_size 8 \
    --seq_len 6 \
    --freeze_finbert \
    --output_dir /global/cfs/cdirs/m4431/sp2160/Checkpoints/2node_1gpu_trial
