#!/bin/bash -l
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -J finbert_finetune
#SBATCH -o ../logs/finbert_finetune.%j.out
#SBATCH -e ../logs/finbert_finetune.%j.err

module load python
# Activate venv from project root (script runs from scripts/ directory)
source ../.venv/bin/activate || {
    echo "ERROR: Virtual environment not found at ../.venv/bin/activate" >&2
    echo "Please create it with: python -m venv .venv" >&2
    exit 1
}

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

srun python ../fine_tune.py \
    --data_path ../data/data/labeled_headlines.parquet \
    --text_column Article_title \
    --label_column label \
    --date_column Date \
    --model_name /global/homes/a/amanp23/Finbert \
    --output_dir ../outputs/finbert_labeled_output \
    --patience 0 \
    --batch_size 16 \
    --epochs 10 \
    --lr 5e-6 \
    --weight_decay 0.1 \
    --gradient_checkpointing
