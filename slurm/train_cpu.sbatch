#!/bin/bash
# Phase 1 — SLURM batch job: single-node CPU rung of the scaling matrix.
# Target: Stanford ME344 cluster (hpcc-cluster-49, Rocky Linux 9).
# Exact hardware constraints declared below; environment comes from the
# pre-built container (or a frozen shared conda env) — nothing pip-installed on nodes.
#SBATCH --job-name=churn-cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # exact CPU core request
#SBATCH --mem=16G                  # system memory bound
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# Data ingestion tier: stage the CSV from shared NFS to node-local scratch
# so the input pipeline never touches NFS during the training loop.
SCRATCH=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$SCRATCH"
cp "$SLURM_SUBMIT_DIR/data/Churn_Dataset.csv" "$SCRATCH/"

# Immutable environment via Apptainer (Docker image converted once, read-only .sif)
apptainer exec --bind "$SCRATCH:/scratch" containers/churn-cpu.sif \
    python src/train_jax.py \
      --data /scratch/Churn_Dataset.csv \
      --backend cpu --threads "$SLURM_CPUS_PER_TASK" \
      --batch-size 512 --epochs 50 \
      --out benchmarks/results/slurm_cpu.json
