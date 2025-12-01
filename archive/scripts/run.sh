#!/bin/bash
#SBATCH -J finbert-lstm-16g
#SBATCH -A m4431
#SBATCH -q shared_interative
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4         # 1 task per GPU

#SBATCH --gpus-per-task=1           # 4 GPUs per node total
#SBATCH --cpus-per-task=32          # 32 cores per GPU (policy)
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
module purge
# module load pytorch
# source $HOME/ece534-finance-project/.venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=cores
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=hsn

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

OUTDIR="$SCRATCH/finbert_runs/${SLURM_JOB_ID}"
mkdir -p logs

srun --kill-on-bad-exit=1 --export=ALL bash -lc '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE=$SLURM_NPROCS
  python -u train.py \
    --outdir '"$OUTDIR"' \
    --lookback 30 --horizon 1 \
    --epochs 10 --batch_size 64 --lr 1e-3 \
    --num_workers '"$OMP_NUM_THREADS"' \
    --amp
'
echo "Finished. Artifacts in $OUTDIR"
