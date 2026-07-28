#!/bin/bash
# Phase 1 — SLURM batch job: single-accelerator GPU rung.
#SBATCH --job-name=churn-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1               # physical accelerator count
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

SCRATCH=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$SCRATCH"
cp "$SLURM_SUBMIT_DIR/data/Churn_Dataset.csv" "$SCRATCH/"

# Telemetry: sample nvidia-smi at 1 Hz for the life of the job.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
           --format=csv -l 1 > "telemetry/nvidia-smi-$SLURM_JOB_ID.csv" &
SMI_PID=$!
trap 'kill $SMI_PID' EXIT

apptainer exec --nv --bind "$SCRATCH:/scratch" containers/churn-cuda.sif \
    python src/train_jax.py \
      --data /scratch/Churn_Dataset.csv \
      --backend gpu \
      --batch-size 1024 --epochs 50 \
      --out benchmarks/results/slurm_gpu.json
