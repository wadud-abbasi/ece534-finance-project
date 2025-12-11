#!/bin/bash -l
#SBATCH -A m4431
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -J finbert_finetune
#SBATCH -o logs/finbert_finetune.%j.out
#SBATCH -e logs/finbert_finetune.%j.err

module load python
# source ~/.bashrc
# conda activate your_env_name

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TRANSFORMERS_NO_TORCHVISION=1   # 🔴 add this line

mkdir -p logs
mkdir -p outputs

HEADLINES_CSV=data/data/processed_headlines_subset.csv
PRICES_CSV=data/data/processed_stock_prices.csv
"""
fine_tune.py \
  --data_path data/data/labeled_headlines_baby.parquet \
  --text_column Article_title \
  --label_column label \
  --date_column Date \
  --model_name /finbert \
  --batch_size 8 \
  --epochs 1 \
  --output_dir outputs/test_baby
"""
srun python -u fine_tune.py \
  --headlines_csv "${HEADLINES_CSV}" \
  --prices_csv "${PRICES_CSV}" \
  --model_name ProsusAI/finbert \
  --output_dir outputs/finbert_price_direction \
  --horizon_days 1 \
  --stable_threshold 0.005 \
  --val_fraction 0.2 \
  --max_length 96 \
  --batch_size 16 \
  --epochs 3 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --fsdp full_shard \
  --gradient_checkpointing
