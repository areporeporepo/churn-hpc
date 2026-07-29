# churn-hpc

**We benchmarked a churn classifier on four architectures across three sites, and the hardware kept losing to a stopwatch.**

Ten Apple M4 cores train this model no faster than one. Thirty-two Xeon cores under SLURM run at 99.7% utilization and finish *slower* than a single core. A 72-core Grace node drops to one seventh of its own single-core speed the moment you hand it 32 threads. And a TPU v5e slice, the most expensive chip in the lineup, takes exactly 2.1 milliseconds per step whether the batch holds 512 rows or 16,384.

That last number is the whole story. When step time refuses to move while batch size grows 32x, you are not compute-bound, memory-bound, or IO-bound. You are paying a fixed toll per step, and every piece of silicon in this repo is idling behind the toll booth.

The workload: a 128-64-32 MLP predicting customer churn from `Churn_Dataset.csv` (5,000 telecom customers, 16 numeric features, 341 KB, 14% churn rate). It hits 91-94% test accuracy against an 86% majority baseline, in under a second, on basically anything. Which is precisely why it makes a good probe: with the arithmetic this small, all you can see is the infrastructure. Fixed costs (kernel dispatch, thread barriers, compile time, staging) are usually noise at the bottom of a profile. Here they *are* the profile.

**The stack** (Compute / Orchestration / Compilation / Telemetry): CPU on M4, Grace, and Xeon, plus M4 GPU and TPU v5e (cluster NVIDIA GPU queued behind another tenant) · Docker + Apptainer, on-prem Kubernetes, GKE Autopilot, and a SLURM controller we installed on the head node ourselves · JAX/XLA on CPU and TPU, PyTorch (eager, Inductor wired) on GPU · psutil node sampler, device-synced step timers, nvidia-smi hooks, and one JSON record per run so every chart regenerates from committed data.

## Repository layout (the four phases)

```
docker/       Phase 1: pinned images (CPU, CUDA 12.4, arm64/cu128). Nodes install nothing, ever.
k8s/          Phase 1: generic GPU/TPU/storage manifests; k8s/hpcc/ holds the jobs that actually ran
slurm/        Phase 1: sbatch templates; train_cpu_hpcc49.sbatch is the one with real numbers behind it
docs/         Phase 2: INFRASTRUCTURE.md (node maps, VPC subnets, the NVMe/NFS/GCS ingestion tier)
scripts/      Phase 2/3: stage_data.sh, fetch_cluster_results.py, watch_gpu_job.sh (watch-and-heal)
src/          data.py, train_jax.py (Flax, jit-fused), train_torch.py (cpu/mps/cuda), telemetry.py
benchmarks/   Phase 3: run_matrix.sh, results/*.json (19 runs), make_figures.py
notebooks/    Phase 3: profiling.ipynb, executed, outputs committed
reports/      Phase 4: executive_report.pdf + figures
slides/       Phase 4: churn_hpc_slides.pdf, exactly five slides
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
  │ rungs: 1c / 10c CPU,      │  32c x86 Xeon head node      │  provisioned on  │
  │ GPU batch sweep, bf16     │  slurmctld+slurmd+munge+     │  demand from a   │
  │                           │  Apptainer (pinned .sif)     │  nodeSelector)   │
  └──────────────┬────────────┴──────────────────────────────┴──────────────────┘
                 │  CSV read once -> float32 matrix in RAM -> device arrays
                 ▼
      benchmarks/results/*.json  ->  reports/figures  ->  report + 5-slide deck
```

Every target funnels data the same way: shared tier, then node-local scratch, then RAM, then device memory. The training loop never touches shared storage. The IO bus sees this dataset exactly once per job, which is why IO never once appears in a profile below.

## 2. Performance Delta Analysis

MLP 128-64-32, Adam, 400 epochs, float32. Full 19-config table in the notebook; the interesting fifteen:

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
| TPU v5e bs512, JAX/XLA | GKE | 2.088 | 189 K | 0.917 |
| TPU v5e bs4000, JAX/XLA | GKE | 2.144 | 789 K | 0.919 |
| TPU v5e bs16384, JAX/XLA | GKE | 2.148 | **3,216 K** | 0.919 |

Read it as three experiments:

**Experiment 1: buy more cores.** M4, 1 to 10 cores: 1.00x. Xeon, 1 to 32 cores: 0.94x, with the node pinned at 99.7% utilization the entire time. Grace, 1 to 32 cores: 0.14x. The best core count on any machine here was between one and eight. Everything past that paid OpenMP barrier costs on GEMMs that finish in microseconds.

**Experiment 2: buy an accelerator.** At the default batch of 512, the M4 GPU is 0.42x its own CPU and the TPU is 0.2x. Accelerators at small batch are dispatch-latency amplifiers: same math, longer launch queue.

**Experiment 3: grow the batch instead.** Same GPU, batch 512 to 4000: 5.15x. Same TPU, batch 512 to 16,384: 17.0x, landing at 3.22 M samples/s, the fastest number in the matrix, at unchanged accuracy. Nobody bought new hardware between those rows. We changed one integer.

Also measured, so you don't have to: bfloat16 on M4 CPU is a 0.75x *regression* (no native bf16 units, pure cast overhead; the flag stays for TPU/A100 where it belongs). Compile overhead: 0.16 s of XLA jit on CPU, 0.41 s on TPU, versus 0.01 s for eager PyTorch. Environment bootstrap: 25-36 s ephemeral, or a 3.2 GB pinned `.sif` pulled once.

## 3. The Infrastructure Bottleneck Diagnosis

**Fixed per-step host-side dispatch latency. Nothing else survives contact with the data.**

Six lines of evidence, each independent, three architectures deep:

1. Core-count indifference on M4: 10x cores, 1.00x throughput.
2. Negative core scaling on both cluster CPUs, including the Xeon's 99.7%-utilization-yet-slower-than-one-core run. **Utilization is not throughput.** A dashboard showing every core busy told us the machine was working hard at synchronizing, not training.
3. The M4 GPU's step floor (1.24 ms) sits 3x above the CPU's (0.40 ms) for identical math. That gap is Metal launch plus host-device sync, not arithmetic.
4. The TPU's step time is *flat* across a 32x batch range: 2.09, 2.14, 2.15 ms. A step that costs the same regardless of work per step is, by definition, all fixed cost.
5. Throughput on every backend scales near-linearly with batch while step time barely moves. Per-step cost, not per-sample cost.
6. Nothing saturates: node CPU under 25% (outside the pathological Xeon case), 19 MB of GPU memory, a sliver of TPU HBM, 0.44 GB RSS, one 341 KB read per job.

FLOPs, memory bandwidth, and the input pipeline are acquitted. The ingestion tier kept the dataset device-resident, so input stalls were zero by construction, and the profiles confirm it.

## 4. Engineering Mitigations

What we applied, with receipts:

- **Batch scaling.** GPU 5.15x, TPU 17.0x, for free, because the step time was fixed cost all along. The GPU's accuracy dip at bs4000 (0.908) is the classic large-batch effect and yields to LR warmup; the TPU held 0.919 at bs16384.
- **XLA train-step fusion.** Forward, backward, and the Adam update compile into one computation. That is what a 0.4 ms full training step on a laptop CPU looks like.
- **Device-resident data.** The entire dataset lives in device memory. IO does not appear in any profile in this repository.
- **bfloat16, measured and rejected** for the M4 CPU (0.75x). A mitigation that ships without a benchmark is a rumor.
- **Right-sized parallelism.** The measured optimum is 1-8 cores. We now say no to free cores.

What we would do next: fuse whole epochs with `jax.lax.scan` (nine dispatches per epoch become one, attacking the diagnosis directly), turn on `torch.compile` for the accelerator path (already wired behind `--compile`), and spend the accelerator manifests on what accelerators are actually for at this scale: fanning out hyperparameter sweeps, where 3.22 M samples/s converts into wall-clock instead of idle toll booths. The TPU wins throughput 3.4x over the best CPU and costs roughly 10x per node-hour; below about a million rows, one CPU node with XLA is the cost-optimal home for a single job.

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
kubectl apply -f k8s/hpcc/train-tpu-job.yaml   # Autopilot provisions the slice itself

# SLURM on hpcc-cluster-49
apptainer pull containers/churn-cpu.sif docker://pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
sbatch slurm/train_cpu_hpcc49.sbatch
```

Status: complete. All three backend classes (CPU, GPU, TPU) are measured across 15 configs; the comparison the analysis rests on is closed. One opportunistic extra remains queued: the Stanford cluster's time-shared NVIDIA GPU, currently held by another tenant with no guaranteed release. `scripts/watch_gpu_job.sh` watches for it, and if it ever runs, its results drop into the same tables as a 16th row. Nothing in the analysis waits on it.

Deliverables: `reports/executive_report.pdf` · `slides/churn_hpc_slides.pdf` · `notebooks/profiling.ipynb` (executed) · `docs/INFRASTRUCTURE.md`.
