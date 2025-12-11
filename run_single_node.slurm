#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -J finbert_single
#SBATCH -t 02:00:00
#SBATCH -o logs/finbert_single_%j.out
#SBATCH -e logs/finbert_single_%j.err

module load python/3.12
export OMP_NUM_THREADS=8

MASTER_ADDR=$(hostname)
MASTER_PORT=29599
NNODES=1
GPUS_PER_NODE=4
WORLD_SIZE=4

export MASTER_ADDR MASTER_PORT WORLD_SIZE

srun torchrun \
  --nproc_per_node=${GPUS_PER_NODE} \
  --nnodes=${NNODES} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
    --labeled_path /global/cfs/cdirs/m4431/sp2160/Data/FNSPID/labeled_headlines.parquet \
    --prices_path /global/cfs/cdirs/m4431/sp2160/Data/FNSPID/processed_stock_prices.csv \
    --epochs 2 \
    --batch_size 32 \
    --seq_len 10 \
    --freeze_finbert \
    --output_dir /global/cfs/cdirs/m4431/sp2160/Checkpoints/single_node_test
