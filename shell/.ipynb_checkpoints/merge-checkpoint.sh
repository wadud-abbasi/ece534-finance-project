#!/bin/bash
#SBATCH -A m4431_g              # your NERSC account
#SBATCH -C gpu                  # GPU nodes on Perlmutter
#SBATCH -q debug              # queue
#SBATCH -t 00:20:00             # walltime for embedding + merge
#SBATCH -N 1                    # 1 node
#SBATCH -G 4                    # 4 GPUs total
#SBATCH -J finbert-merge-4g
#SBATCH -o merge_%j.out
#SBATCH -e merge_%j.err

# Load your environment
module load python
source $SCRATCH/ece534-envs/.venv/bin/activate

# keep HF cache off $HOME (no TRANSFORMERS_CACHE to avoid warning)
export HF_HOME=$SCRATCH/hf_cache

cd ~/ece534-finance-project

export OMP_NUM_THREADS=8

# Master address for torchrun
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29501

# One launcher per node, each spawning 4 local processes (GPUs)
srun --ntasks-per-node=1 --gpus-per-task=4 torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/merge.py \
    --dist_mode ddp \
    --headlines_csv data/data/processed_headlines_subset.csv \
    --prices_csv    data/data/processed_stock_prices.csv \
    --indexes_csv   data/data/processed_indexes.csv \
    --finetuned_weights models/model.safetensors \
    --output_path   data/data/merged_lstm_dataset_half.parquet \
    --max_len 128 \
    --batch_size 256 \
    --amp \
    --subset_fraction 0.5

