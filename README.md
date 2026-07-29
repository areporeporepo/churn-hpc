# churn-hpc - Customer Churn DNN on the Modern AI Infrastructure Stack

**Executive summary.** A multilayer-perceptron churn classifier (JAX/Flax +
PyTorch, 91-94% test accuracy vs an 86% majority-class baseline) trained on
`Churn_Dataset.csv` (5,000 telecom customers, 16 numeric features), used as a
vehicle to exercise the full AI-infrastructure lifecycle: immutable container
environments, reusable Kubernetes and SLURM orchestration, a staged
storage/ingestion tier, and a hardware scaling matrix measured on **four
architectures across three sites**: Apple M4 (CPU + GPU), arm64 Grace-class and
x86 Xeon nodes on the Stanford hpcc cluster (Kubernetes and SLURM), and a
Google Cloud **TPU v5e 2x2 slice** (GKE Autopilot). The measured conclusion:
at this dataset scale every backend is **launch-latency-bound**, not compute-,
memory-, or IO-bound. Extra CPU cores buy nothing (or less than nothing), and
accelerators only pay once the batch amortizes the fixed per-step dispatch
cost; the TPU's step time is *constant* from batch 512 to 16,384, which turns
linear batch growth into linear throughput growth up to a measured 3.22 M
samples/s.

**Stack elements used** (>= 3 required): Compute = CPU (M4, Grace, Xeon) +
GPU (M4/Metal; cluster NVIDIA job queued) + **TPU v5e**; Orchestration =
Docker/Apptainer + Kubernetes (on-prem + GKE Autopilot) + **SLURM** (installed
and configured single-node on `hpcc-cluster-49`); Compilation = JAX/XLA (CPU +
TPU) and PyTorch (eager + optional Inductor); Telemetry = native node metrics
(psutil sampler, per-step device-synced timers, nvidia-smi hook).

## Repository layout (the four phases)

```
docker/       Phase 1 - immutable envs: Dockerfile.cpu, Dockerfile.cuda, Dockerfile.arm64, pinned reqs
k8s/          Phase 1 - generic GPU/TPU/storage manifests; k8s/hpcc/ = the measured cluster jobs
slurm/        Phase 1 - generic sbatch templates; train_cpu_hpcc49.sbatch = the measured SLURM run
docs/         Phase 2 - INFRASTRUCTURE.md: node layout, VPC/subnets, NVMe/NFS/GCS ingestion tier
scripts/      Phase 2/3 - stage_data.sh, fetch_cluster_results.py, watch_gpu_job.sh
src/          Model code: data.py (pandas/numpy), train_jax.py (Flax/XLA), train_torch.py, telemetry.py
benchmarks/   Phase 3 - run_matrix.sh, results/*.json (19 measured configs), make_figures.py
notebooks/    Phase 3 - profiling.ipynb (executed, outputs committed)
reports/      Phase 4 - executive_report.pdf + figures/
slides/       Phase 4 - churn_hpc_slides.pdf (exactly 5 slides)
```

## 1. System Topology Diagram

```
                     ┌────────────────────────────────────────────────┐
                     │             Canonical data sources             │
                     │  NFS (cluster home)   ConfigMap   GCS/gcsfuse  │
                     └────────┬───────────────┬──────────────┬────────┘
                       staged once per job (scripts/stage_data.sh / K8s mounts)
                              │               │              │
  ┌───────────────────────────┼───────────────┼──────────────┼──────────────────┐
  │ MEASURED (local)          │ MEASURED (Stanford hpcc)     │ MEASURED (GCP)   │
  │ Apple M4                  │ K8s: hpcc-pilot              │ GKE Autopilot    │
  │ 10 cores / 16 GB          │  72c arm64 Grace + 1 NVIDIA  │ class-tpu-cluster│
  │ + M4 GPU (MPS)            │  GPU (GPU job queued)        │ TPU v5e 2x2      │
  │                           │ SLURM: hpcc-cluster-49       │  (4 chips,       │
  │ rungs: 1c / 10c CPU,      │  32c x86 Xeon head node      │  Autopilot-      │
  │ GPU batch sweep, bf16     │  slurmctld+slurmd+munge+     │  provisioned     │
  │                           │  Apptainer (pinned .sif)     │  on demand)      │
  └──────────────┬────────────┴──────────────────────────────┴──────────────────┘
                 │  CSV loaded to RAM once -> float32 matrix -> device arrays
                 ▼
      benchmarks/results/*.json  ->  reports/figures  ->  report + 5-slide deck
```

Storage path on every target: **shared tier (NFS / ConfigMap / GCS) ->
node-local scratch -> host RAM -> device memory**. The training loop never
touches shared storage; the IO bus sees the 341 KB CSV exactly once per job
(verified: zero input-pipeline stalls in every profile).

## 2. Performance Delta Analysis

MLP 128-64-32, Adam, 400 epochs, float32. Full 19-config table in the
notebook; figures in `reports/figures/`.

| Config | Site | p50 step (ms) | Samples/s | Test acc |
|---|---|---|---|---|
| CPU 1-core, JAX/XLA | M4 local | 0.395 | 897 K | 0.916 |
| CPU 10-core, JAX/XLA | M4 local | 0.395 | 898 K | 0.916 |
| CPU 10-core, PyTorch | M4 local | 0.549 | 938 K | 0.917 |
| GPU (MPS) bs512, PyTorch | M4 local | 1.237 | 394 K | 0.917 |
| GPU (MPS) bs4000, PyTorch | M4 local | 1.739 | 2,013 K | 0.908 |
| CPU Grace 1-core bs512 | hpcc K8s | 2.314 | 196 K | 0.921 |
| CPU Grace 32-core bs512 | hpcc K8s | 17.188 | 27.8 K | 0.921 |
| CPU Grace 32-core bs4000 | hpcc K8s | 34.811 | 111 K | 0.919 |
| CPU Xeon 1-core bs512 | hpcc SLURM | 4.463 | 114 K | 0.913 |
| CPU Xeon 8-core bs512 | hpcc SLURM | 3.631 | 140 K | 0.914 |
| CPU Xeon 32-core bs512 | hpcc SLURM | 4.513 | 107 K | 0.918 |
| CPU Xeon 8-core bs4000 | hpcc SLURM | 7.193 | 540 K | 0.916 |
| **TPU v5e bs512, JAX/XLA** | GKE | 2.088 | 189 K | 0.917 |
| **TPU v5e bs4000, JAX/XLA** | GKE | 2.144 | 789 K | 0.919 |
| **TPU v5e bs16384, JAX/XLA** | GKE | 2.148 | **3,216 K** | 0.919 |

Headline deltas:

- **Core scaling is dead on every architecture.** M4: 10c/1c = 1.00x.
  Xeon (SLURM): 32c/1c = 0.94x at **99.7% node utilization** (all 32 cores
  spinning, zero gain). Grace (K8s): 32c/1c = **0.14x**, a 7.1x slowdown.
- **Accelerators lose at small batch, win at large batch.** M4 GPU at bs512 is
  0.42x the CPU; at bs4000 it is 2.1x. TPU at bs512 is 0.2x the M4 CPU; at
  bs16384 it is 3.4x (3.22 M samples/s, best measured).
- **TPU step time is flat: 2.09 -> 2.14 -> 2.15 ms across a 32x batch range.**
  Pure fixed-cost dominance; throughput is simply batch/(fixed step time).
- **Compile overhead:** JAX jit 0.16 s (CPU) / 0.41 s (TPU); torch eager 0.01 s.
  Fixed cost, amortizes to noise over real workloads.
- **bfloat16 on M4 CPU: 0.75x** (no native bf16 path; cast overhead only).
  Kept for TPU/A100 where the hardware supports it.
- **Environment bootstrap:** 36 s (Grace, cpu wheels) / 25 s (TPU, jax[tpu])
  in ephemeral containers; 3.2 GB pinned `.sif` pulled once for SLURM.

## 3. The Infrastructure Bottleneck Diagnosis

**Primary bottleneck: fixed per-step host-side dispatch/launch latency.**
Six independent lines of evidence, three architectures:

1. Core-count indifference on M4 (1.00x for 10x cores): per-step compute
   (~0.1 ms of GEMMs on a 512x16 batch) sits below the threshold where
   thread-pool synchronization pays.
2. Negative scaling on Grace (0.14x for 32x cores) and Xeon (0.94x at 100%
   utilization): OpenMP fork/join and barrier costs on sub-millisecond GEMMs
   grow with thread count. **Utilization is not throughput** - the Xeon burns
   32 cores at 99.7% to go slower than 1 core.
3. GPU p50 step (1.24 ms) is ~3x CPU (0.40 ms) for identical math: the Metal
   kernel-launch + host-device sync floor.
4. TPU step time is constant across 512 -> 16,384 batch (2.1 ms): the step is
   pure fixed cost until enormous batches; throughput scales exactly linearly.
5. Throughput on every backend scales near-linearly with batch size while step
   time barely moves: the signature of per-step (not per-sample) cost.
6. No resource saturates: node CPU <= 25% (except the pathological Xeon 32c
   case), GPU memory ~19 MB, TPU HBM a fraction of 16 GB/chip, RSS 0.44 GB,
   IO one 341 KB read per job.

Not the bottleneck: FLOPs, memory bandwidth/HBM, IO (the ingestion tier keeps
the dataset device-resident; input stalls are zero by construction).

## 4. Engineering Mitigations

Applied and measured:

- **Batch scaling** - GPU 512->4000: 5.15x; TPU 512->16384: **17.0x** (the
  fixed step time makes this free until HBM or convergence pushes back). The
  large-batch accuracy dip (0.908 at bs4000 on GPU) is recoverable with LR
  warmup; TPU at bs16384 held 0.919.
- **Device/RAM-resident dataset** - zero input stalls on all backends.
- **XLA jit fusion** - fwd+bwd+Adam in one compiled computation (0.4 ms M4
  steps; the TPU's flat 2.1 ms includes the whole update).
- **bfloat16** - measured 0.75x on M4 CPU and rejected there; retained for
  TPU/A100 targets.
- **Right-sized parallelism** - the measured answer to "how many cores?" is
  1-8; more is negative value at this scale.

Recommended next:

- **Epoch-level fusion** (`jax.lax.scan` over minibatches): collapses ~9
  dispatches/epoch into 1; directly attacks the diagnosed bottleneck on CPU
  and would pull the TPU's small-batch throughput toward its large-batch line.
- **`torch.compile` (Inductor)** on the accelerator path (flag already wired).
- **Spend accelerators on parallel sweeps, not single small jobs**: the TPU
  wins 3.4x on throughput but costs ~10x a CPU node-hour; below ~1 M rows a
  single CPU node with XLA is cost-optimal, and the manifests here are best
  used to fan out hyperparameter sweeps.

## Reproduce

```bash
# local (uv, Python 3.12)
uv venv .venv && uv pip install -r docker/requirements-cpu.txt
./benchmarks/run_matrix.sh && .venv/bin/python benchmarks/make_figures.py

# Stanford hpcc K8s (arm64 Grace node)
kubectl create configmap churn-src --from-file=src/
kubectl create configmap churn-data --from-file=data/Churn_Dataset.csv
kubectl apply -f k8s/hpcc/train-cpu-job.yaml -f k8s/hpcc/train-gpu-job.yaml

# GKE Autopilot TPU v5e (soe-hpccenter class-tpu-cluster)
kubectl apply -f k8s/hpcc/train-tpu-job.yaml   # Autopilot provisions the slice

# SLURM on hpcc-cluster-49 (single node, see slurm/)
apptainer pull containers/churn-cpu.sif docker://pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
sbatch slurm/train_cpu_hpcc49.sbatch
```

Status: 15 of 16 planned configs measured. The Stanford cluster NVIDIA GPU job
(`churn-train-gpu`) remains queued behind another tenant's allocation
(single shared GPU); `scripts/watch_gpu_job.sh` auto-collects it when free.

Deliverables: `reports/executive_report.pdf` · `slides/churn_hpc_slides.pdf` ·
`notebooks/profiling.ipynb` (executed) · `docs/INFRASTRUCTURE.md`.
