#!/bin/bash
# Phase 2 — Stage the dataset from shared storage to node-local scratch.
# Usage: stage_data.sh <src_csv> [scratch_dir]
set -euo pipefail

SRC=${1:?usage: stage_data.sh <src_csv> [scratch_dir]}
SCRATCH=${2:-${SLURM_TMPDIR:-/tmp/$USER/churn-scratch}}

mkdir -p "$SCRATCH"
DST="$SCRATCH/$(basename "$SRC")"

# Skip the copy if an identical file is already staged (idempotent re-runs).
if [[ -f "$DST" ]] && cmp -s "$SRC" "$DST"; then
    echo "already staged: $DST"
else
    cp "$SRC" "$DST"
    echo "staged: $SRC -> $DST"
fi
echo "$DST"
