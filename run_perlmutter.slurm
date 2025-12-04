#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 04:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -J finbert_hybrid
#SBATCH -o logs/finbert_hybrid_%j.out
#SBATCH -e logs/finbert_hybrid_%j.err

module load pytorch/2.3.0  # or your preferred module
# or your own env:
# module load python
# conda activate nersc-python

export OMP_NUM_THREADS=8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

export MASTER_ADDR MASTER_PORT WORLD_SIZE

srun torchrun \
  --nproc_per_node=${GPUS_PER_NODE} \
  --nnodes=${NNODES} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
    --labeled_path /global/cfs/cdirs/m4431/sp2160/Data/FNSPID/labeled_headlines.parquet \
    --prices_path /global/cfs/cdirs/m4431/sp2160/Data/FNSPID/processed_stock_prices.csv \
    --epochs 4 \    
    --batch_size 32 \
    --seq_len 10 \
    --freeze_finbert \
    --output_dir /global/cfs/cdirs/m4431/sp2160/Checkpoints/finbert_hybrid
