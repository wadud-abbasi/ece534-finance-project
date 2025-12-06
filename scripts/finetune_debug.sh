#!/bin/bash -l
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
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

srun python ../fine_tune.py \
    --data_path ../data/data/labeled_headlines.parquet \
    --text_column Article_title \
    --label_column label \
    --date_column Date \
    --model_name ProsusAI/finbert \
    --output_dir ../outputs/finbert_labeled_output \
    --batch_size 16 \
    --epochs 3 \
    --lr 2e-5 \
    --weight_decay 0.01 \
    --gradient_checkpointing
