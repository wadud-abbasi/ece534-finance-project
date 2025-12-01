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
source .venv/bin/activate

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
