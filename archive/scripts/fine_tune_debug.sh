#!/bin/bash -l
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q debug                 # ≤ 30 minutes
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -J finbert_debug
#SBATCH -o logs/finbert_debug.%j.out
#SBATCH -e logs/finbert_debug.%j.err

module load python
# source ~/.bashrc
# conda activate your_env_name

cd /global/u2/w/wa176/ece534-finance-project

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TRANSFORMERS_NO_TORCHVISION=1
export DISABLE_TRANSFORMERS_AV=1

mkdir -p logs
mkdir -p outputs

HEADLINES_CSV=data/data/processed_headlines_subset.csv
PRICES_CSV=data/data/processed_stock_prices.csv

srun python -u fine_tune.py \
  --headlines_csv "${HEADLINES_CSV}" \
  --prices_csv "${PRICES_CSV}" \
  --model_name ProsusAI/finbert \
  --output_dir outputs/finbert_price_direction_debug \
  --horizon_days 1 \
  --stable_threshold 0.005 \
  --val_fraction 0.2 \
  --max_length 96 \
  --batch_size 16 \
  --epochs 1 \
  --lr 2e-5 \
  --weight_decay 0.01
