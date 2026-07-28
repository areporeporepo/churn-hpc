#!/bin/bash
# Phase 1 — SLURM batch job: distributed multi-node rung (jax.distributed over 2 nodes).
#SBATCH --job-name=churn-multinode
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# High-speed internode pipe: coordinator on the first node's IB interface.
COORD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun bash -c '
  SCRATCH=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
  mkdir -p "$SCRATCH"
  cp "$SLURM_SUBMIT_DIR/data/Churn_Dataset.csv" "$SCRATCH/"
  apptainer exec --nv --bind "$SCRATCH:/scratch" containers/churn-cuda.sif \
    python src/train_jax.py \
      --data /scratch/Churn_Dataset.csv \
      --backend gpu --distributed \
      --coordinator '"$COORD"':12345 \
      --num-processes "$SLURM_NTASKS" --process-id "$SLURM_PROCID" \
      --batch-size 2048 --epochs 50 \
      --out benchmarks/results/slurm_multinode_rank${SLURM_PROCID}.json
'
