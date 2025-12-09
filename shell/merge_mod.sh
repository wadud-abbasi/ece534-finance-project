#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00          # give yourself time
#SBATCH -N 1
#SBATCH -G 1                 # request 1 GPU on the node
#SBATCH -J lstm-train-1g
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err

module load python
source $SCRATCH/ece534-envs/.venv/bin/activate

cd $HOME/ece534-finance-project

export OMP_NUM_THREADS=8

# IMPORTANT: no torchrun, single task, single GPU
srun --ntasks=1 --gpus-per-task=1 python scripts/train.py \
  --merged_path Data/data/merged_lstm_dataset_half.parquet \
  --outdir outputs/lstm_1g_1n \
  --lookback 30 \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --dist_mode none \
  --amp
