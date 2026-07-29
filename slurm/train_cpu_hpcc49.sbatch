#!/bin/bash
# Measured single-node SLURM rung on hpcc-cluster-49 (32-core x86_64 head node,
# SLURM 22.05 + Apptainer 1.5, image = pinned pytorch/pytorch:2.7.1 sif).
# Thread-scaling matrix: 1/8/32 threads at bs512 + 8 threads at bs4000.
#SBATCH --job-name=churn-cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Data ingestion tier: stage CSV from shared FS to node-local scratch.
SCRATCH=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$SCRATCH"
cp data/Churn_Dataset.csv "$SCRATCH/"

for cfg in "1 512 slurm-1c-bs512" "8 512 slurm-8c-bs512" "32 512 slurm-32c-bs512" "8 4000 slurm-8c-bs4000"; do
  set -- $cfg
  apptainer exec --bind "$SCRATCH:/scratch" --bind "$SLURM_SUBMIT_DIR:/work" \
    containers/churn-cpu.sif \
    python /work/src/train_torch.py --data /scratch/Churn_Dataset.csv \
      --device cpu --threads "$1" --batch-size "$2" --epochs 400 \
      --label "$3" --out "/work/results/$3.json"
done
rm -rf "$SCRATCH"
echo SLURM_MATRIX_DONE
