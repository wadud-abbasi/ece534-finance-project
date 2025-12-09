#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular              # use regular for multi-node, not debug
#SBATCH -t 03:00:00
#SBATCH -N 8                    # number of nodes
#SBATCH --ntasks-per-node=4     # 4 tasks per node
#SBATCH --gpus-per-node=4       # 1 GPU per task, 4 GPUs per node
#SBATCH -J lstm-train-ddp
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err

module load python
source $SCRATCH/ece534-envs/.venv/bin/activate

cd $HOME/ece534-finance-project

export OMP_NUM_THREADS=8

# pick rank 0 node as master
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Slurm will set RANK, WORLD_SIZE, etc. for each task.
# train.py will see dist_mode=ddp and use those.
srun python scripts/train.py \
  --merged_path data/data/merged_lstm_dataset_nickel.parquet \
  --outdir models/lstm_ddp_baby \
  --lookback 10 \
  --epochs 3 \
  --batch_size 16 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --dist_mode fsdp \
  --amp \
  --save_every 1

