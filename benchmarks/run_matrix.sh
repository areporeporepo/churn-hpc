#!/bin/bash
# Phase 3 — Scaling matrix: three discrete scale steps + supporting sweeps.
#   Step 1: single-core CPU        (JAX/XLA, 1 thread)
#   Step 2: full-node CPU          (JAX/XLA, all cores)
#   Step 3: single accelerator GPU (PyTorch on Apple M4 GPU via MPS)
# Extras: PyTorch CPU (framework delta), bfloat16 (mitigation test),
#         batch-size sweep on the GPU (saturation curve).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
DATA=data/Churn_Dataset.csv
OUT=benchmarks/results
EPOCHS=${EPOCHS:-50}

$PY src/train_jax.py   --data $DATA --backend cpu --threads 1  --epochs $EPOCHS --label "cpu-1core-jax"   --out $OUT/cpu_1core_jax.json
$PY src/train_jax.py   --data $DATA --backend cpu --threads 10 --epochs $EPOCHS --label "cpu-10core-jax"  --out $OUT/cpu_10core_jax.json
$PY src/train_jax.py   --data $DATA --backend cpu --threads 10 --epochs $EPOCHS --bf16 --label "cpu-10core-jax-bf16" --out $OUT/cpu_10core_jax_bf16.json
$PY src/train_torch.py --data $DATA --device cpu  --threads 10 --epochs $EPOCHS --label "cpu-10core-torch" --out $OUT/cpu_10core_torch.json
$PY src/train_torch.py --data $DATA --device mps               --epochs $EPOCHS --label "gpu-mps-torch"    --out $OUT/gpu_mps_torch.json

# GPU batch-size saturation sweep
for bs in 128 512 2048 4000; do
  $PY src/train_torch.py --data $DATA --device mps --batch-size $bs --epochs $EPOCHS \
      --label "gpu-mps-bs$bs" --out $OUT/gpu_mps_bs$bs.json
done

# JAX batch sweep on CPU (XLA fusion behaviour vs batch)
for bs in 128 2048 4000; do
  $PY src/train_jax.py --data $DATA --backend cpu --threads 10 --batch-size $bs --epochs $EPOCHS \
      --label "cpu-10core-jax-bs$bs" --out $OUT/cpu_10core_jax_bs$bs.json
done
echo "matrix complete -> $OUT"
