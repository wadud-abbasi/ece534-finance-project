#!/bin/bash
#SBATCH -A m4331_g              # TODO: replace with your NERSC account
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00                  # walltime for training
#SBATCH -N 1                         # 4 nodes
#SBATCH -G 4                        # 16 GPUs total
#SBATCH -J lstm-train-16g
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err

module load python                   # or module load pytorch
source $SCRATCH/ece534-envs/.venv/bin/activate

cd ~/ece534-finance-project

export OMP_NUM_THREADS=8

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

srun --ntasks-per-node=1 --gpus-per-task=4 torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
    --merged_path data/data/merged_lstm_dataset.parquet \
    --outdir outputs/lstm_16gpus_4nodes \
    --lookback 30 \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --num_workers 4 \
    --dist_mode ddp \
    --amp
