"""Build notebooks/profiling.ipynb (executed separately via nbconvert)."""
import pathlib

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
md("""# Churn-HPC - Dataset & Performance Profiling Notebook

Two profiles in one place:

1. **Data profile** of `Churn_Dataset.csv` (5,000 telecom customers, 16 numeric features, boolean `churned` label).
2. **Hardware profile**: the Phase-3 scaling matrix results (single-core CPU vs full-node CPU vs Apple M4 GPU), loaded from `benchmarks/results/*.json`.

All benchmark numbers here were measured on this machine (Apple M4, 10 cores, 16 GB) with JAX/XLA 0.11 and PyTorch 2.13."""),

code("""import json, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
df = pd.read_csv(ROOT / "data" / "Churn_Dataset.csv")
print(df.shape)
df.head()"""),

md("## 1. Data profile"),

code("""summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "missing_pct": (100 * df.isna().mean()).round(2),
    "mean": df.select_dtypes("number").mean().round(2),
    "std": df.select_dtypes("number").std().round(2),
})
summary"""),

code("""churn_rate = (df["churned"].astype(str).str.upper() == "TRUE").mean()
print(f"churn rate: {churn_rate:.1%}  (class imbalance -> accuracy baseline {1-churn_rate:.1%})")"""),

code("""y = (df["churned"].astype(str).str.upper() == "TRUE").astype(int)
num = df.drop(columns=["churned"]).apply(pd.to_numeric, errors="coerce")
corr = num.corrwith(y).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
corr.plot.barh(color=["#D55E00" if v > 0 else "#0072B2" for v in corr], ax=ax)
ax.set_title("Feature correlation with churn")
ax.set_xlabel("Pearson r")
plt.tight_layout()"""),

md("""**Data takeaways:** ~14% churn rate, so the ~92% test accuracy of the MLP must be judged against an ~86% majority-class baseline. `total_day_minutes`/`total_day_charge` (perfectly collinear) and `number_customer_service_calls` carry most of the signal. A few columns have sub-1% missingness, imputed with medians in `src/data.py`. The whole dataset is ~600 KB dense float32: it fits in L2 cache slices, which drives everything in the hardware profile below."""),

md("## 2. Hardware profile - the scaling matrix"),

code("""RES = ROOT / "benchmarks" / "results"
rows = []
for p in sorted(RES.glob("*.json")):
    r = json.load(open(p))
    rows.append({
        "config": r["label"], "framework": r["framework"], "batch": r["batch_size"],
        "compile_s": r["compile_overhead_s"], "p50_step_ms": r["p50_step_ms"],
        "steps/s": r["steps_per_sec"], "samples/s": r["samples_per_sec"],
        "cpu_util_%": r["telemetry"].get("cpu_util_mean_pct"),
        "peak_rss_MB": r["telemetry"].get("rss_peak_mb"),
        "test_acc": r["test_accuracy"],
    })
matrix = pd.DataFrame(rows).sort_values("samples/s", ascending=False)
matrix"""),

code("""from IPython.display import Image, display
for f in ["fig1_throughput", "fig2_batch_sweep", "fig3_step_compile", "fig4_telemetry"]:
    display(Image(str(ROOT / "reports" / "figures" / (f + ".png"))))"""),

md("""## 3. Bottleneck diagnosis

The matrix is unambiguous:

- **10 CPU cores are no faster than 1 core** (0.90M samples/s both): the per-step compute (~0.3 ms of GEMMs on a 512x16 batch) is far below the threshold where XLA's intra-op parallelism pays for its thread-pool synchronization.
- **The GPU *loses* to the CPU at batch 512** (0.39M vs 0.94M samples/s): every step pays a fixed Metal kernel-launch + host-device sync cost (~1.2 ms) that dwarfs the arithmetic.
- **The GPU only wins when the batch grows** (2.01M samples/s at batch 4000, 5.2x its own small-batch rate): larger batches amortize the launch overhead.
- **CPU utilization never exceeded ~20% of the node** and peak RSS was ~0.4 GB: neither compute nor memory saturates.

**Primary bottleneck: per-step dispatch/launch overhead (host-side), not FLOPs, not memory bandwidth, not IO.** The workload is *launch-latency-bound*. Mitigations (validated or recommended) are in the README: larger batches (validated, 5.2x), epoch-level jit/`jax.lax.scan` to fuse the whole epoch into one XLA computation, bfloat16 (measured a *slowdown* on M4 CPU: no native bf16 units, cast overhead only), and keeping this workload class on CPU unless batches are >=2K.

**Cluster replication (rows `cpu-cluster-*` above):** the same trainer on the Stanford hpcc K8s cluster's 72-core arm64 Grace node reproduces the diagnosis in extreme form: 32 threads are 7.1x *slower* than 1 thread at batch 512 (27.8K vs 196K samples/s): OpenMP barrier costs on sub-millisecond GEMMs make parallelism negative-value. See `reports/figures/fig5_cluster.png`."""),
]

out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "profiling.ipynb"
nbf.write(nb, out)
print("wrote", out)
