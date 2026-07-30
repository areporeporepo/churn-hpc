"""Generate profiling figures from benchmarks/results/*.json into reports/figures/."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "benchmarks" / "results"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito subset, validated CVD-safe (dataviz six-checks, light mode).
BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.axisbelow": True,
})


def load(name):
    with open(RES / f"{name}.json") as f:
        return json.load(f)


def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")


# ── Fig 1: scaling matrix throughput ─────────────────────────────────────────
steps = [
    ("CPU 1-core\nJAX/XLA", load("cpu_1core_jax"), BLUE),
    ("CPU 10-core\nJAX/XLA", load("cpu_10core_jax"), BLUE),
    ("CPU 10-core\nPyTorch", load("cpu_10core_torch"), ORANGE),
    ("M4 GPU (MPS)\nPyTorch bs=512", load("gpu_mps_torch"), GREEN),
    ("M4 GPU (MPS)\nPyTorch bs=4000", load("gpu_mps_bs4000"), GREEN),
]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
bars = ax.bar([s[0] for s in steps], [s[1]["samples_per_sec"] / 1e6 for s in steps],
              color=[s[2] for s in steps], width=0.62)
bar_labels(ax, bars, lambda v: f"{v:.2f}M")
ax.set_ylabel("Throughput (M samples/sec)")
ax.set_title("Scaling matrix: training throughput by backend (400 epochs, MLP 128-64-32)")
fig.tight_layout(); fig.savefig(FIG / "fig1_throughput.png"); plt.close(fig)

# ── Fig 2: batch-size saturation sweep ───────────────────────────────────────
bss = [128, 512, 2048, 4000]
gpu = [load(f"gpu_mps_bs{b}")["samples_per_sec"] / 1e6 for b in bss]
cpu = [load("cpu_10core_jax_bs128")["samples_per_sec"] / 1e6,
       load("cpu_10core_jax")["samples_per_sec"] / 1e6,
       load("cpu_10core_jax_bs2048")["samples_per_sec"] / 1e6,
       load("cpu_10core_jax_bs4000")["samples_per_sec"] / 1e6]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(bss, gpu, "-o", color=GREEN, lw=2, ms=7, label="M4 GPU (PyTorch/MPS)")
ax.plot(bss, cpu, "-o", color=BLUE, lw=2, ms=7, label="CPU 10-core (JAX/XLA)")
for x, y in zip(bss, gpu):
    ax.annotate(f"{y:.2f}M", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color=GREEN, fontweight="bold")
ax.set_xscale("log"); ax.set_xticks(bss); ax.set_xticklabels(bss)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("Batch size"); ax.set_ylabel("Throughput (M samples/sec)")
ax.set_title("Accelerator saturation: throughput vs batch size")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "fig2_batch_sweep.png"); plt.close(fig)

# ── Fig 3: step time vs compile/dispatch overhead ────────────────────────────
names = ["cpu_1core_jax", "cpu_10core_jax", "cpu_10core_torch", "gpu_mps_torch"]
labels = ["CPU 1c\nJAX", "CPU 10c\nJAX", "CPU 10c\nTorch", "M4 GPU\nTorch"]
p50 = [load(n)["p50_step_ms"] for n in names]
comp = [load(n)["compile_overhead_s"] * 1e3 for n in names]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
b1 = a1.bar(labels, p50, color=BLUE, width=0.6)
bar_labels(a1, b1, lambda v: f"{v:.2f}")
a1.set_ylabel("p50 step time (ms)"); a1.set_title("Steady-state step time")
b2 = a2.bar(labels, comp, color=VERM, width=0.6)
bar_labels(a2, b2, lambda v: f"{v:.0f}")
a2.set_ylabel("first-step / compile (ms)"); a2.set_title("Compile + warmup overhead")
fig.tight_layout(); fig.savefig(FIG / "fig3_step_compile.png"); plt.close(fig)

# ── Fig 4: node telemetry (CPU util + RSS) ───────────────────────────────────
runs = [("CPU 1-core JAX", "cpu_1core_jax", BLUE),
        ("CPU 10-core JAX", "cpu_10core_jax", ORANGE),
        ("M4 GPU Torch bs=512", "gpu_mps_torch", GREEN)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
for label, name, c in runs:
    s = load(name)["telemetry"]["series"]
    a1.plot([p[0] for p in s], [p[1] for p in s], color=c, lw=1.8, label=label)
a1.set_xlabel("wall time (s)"); a1.set_ylabel("node CPU util (%)")
a1.set_ylim(0, 100); a1.set_title("CPU utilization during training")
a1.legend(frameon=False, fontsize=8)
rss = [load(n)["telemetry"]["rss_peak_mb"] for _, n, _ in runs]
b = a2.bar(["CPU\n1-core", "CPU\n10-core", "M4 GPU\nbs=512"], rss,
           color=[r[2] for r in runs], width=0.6)
bar_labels(a2, b, lambda v: f"{v:.0f}")
a2.set_ylabel("peak RSS (MB)"); a2.set_title("Peak process memory")
fig.tight_layout(); fig.savefig(FIG / "fig4_telemetry.png"); plt.close(fig)

# ── Fig 5: cluster CPU rungs — thread scaling futility on two architectures ──
def maybe(name):
    p = RES / f"{name}.json"
    return json.load(open(p)) if p.exists() else None

cluster = [("Grace 1c\n(K8s)", maybe("cpu_cluster_1c_bs512"), ORANGE),
           ("Grace 32c\n(K8s)", maybe("cpu_cluster_32c_bs512"), ORANGE),
           ("Xeon 1c\n(SLURM)", maybe("slurm_1c_bs512"), VERM),
           ("Xeon 8c\n(SLURM)", maybe("slurm_8c_bs512"), VERM),
           ("Xeon 32c\n(SLURM)", maybe("slurm_32c_bs512"), VERM)]
cluster = [(l, r, c) for l, r, c in cluster if r]
if cluster:
    m4_best = load("cpu_10core_torch")["samples_per_sec"] / 1e6
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar([c[0] for c in cluster], [c[1]["samples_per_sec"] / 1e6 for c in cluster],
                  color=[c[2] for c in cluster], width=0.6)
    for b, (_, r, _) in zip(bars, cluster):
        v = r["samples_per_sec"] / 1e6
        cpu = r["telemetry"].get("cpu_util_mean_pct")
        ax.annotate(f"{v*1e3:.0f}K\n({cpu:.0f}% util)", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.axhline(m4_best, color=BLUE, lw=1.5, ls="--")
    ax.annotate(f"local M4 best (torch CPU): {m4_best:.2f}M", (0.02, m4_best),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", fontsize=9, color=BLUE)
    ax.set_ylim(0, m4_best * 1.18)
    ax.set_ylabel("Throughput (M samples/sec)")
    ax.set_title("Cluster CPU rungs at batch 512: more cores, same or less throughput")
    fig.tight_layout(); fig.savefig(FIG / "fig5_cluster.png"); plt.close(fig)

# ── Fig 6: the backend comparison (CPU vs GPU vs TPU peak + step-time flatness)
tpu512, tpu4k, tpu16k = maybe("tpu_v5e_bs512"), maybe("tpu_v5e_bs4000"), maybe("tpu_v5e_bs16384")
if tpu16k:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    peaks = [("CPU\nM4 10c\nbs512", load("cpu_10core_torch"), BLUE),
             ("CPU\nXeon SLURM\n8c bs4000", maybe("slurm_8c_bs4000"), BLUE),
             ("GPU\nM4 MPS\nbs4000", load("gpu_mps_bs4000"), GREEN),
             ("GPU\nGH200\nbs4000", maybe("gpu_cluster_bs4000"), GREEN),
             ("TPU\nv5e slice\nbs16384", tpu16k, ORANGE)]
    peaks = [(l, r, c) for l, r, c in peaks if r]
    bars = a1.bar([p[0] for p in peaks], [p[1]["samples_per_sec"] / 1e6 for p in peaks],
                  color=[p[2] for p in peaks], width=0.6)
    bar_labels(a1, bars, lambda v: f"{v:.2f}M")
    a1.set_ylabel("Peak throughput (M samples/sec)")
    a1.set_title("Best measured config per backend")

    bss = [512, 4000, 16384]
    tpu_p50 = [tpu512["p50_step_ms"], tpu4k["p50_step_ms"], tpu16k["p50_step_ms"]]
    cpu_p50 = [load("cpu_10core_jax")["p50_step_ms"], load("cpu_10core_jax_bs4000")["p50_step_ms"], None]
    gpu_p50 = [load("gpu_mps_bs512")["p50_step_ms"], load("gpu_mps_bs4000")["p50_step_ms"], None]
    a2.plot(bss, tpu_p50, "-o", color=ORANGE, lw=2, ms=7, label="TPU v5e (XLA)")
    a2.plot(bss[:2], cpu_p50[:2], "-o", color=BLUE, lw=2, ms=7, label="CPU M4 (XLA)")
    a2.plot(bss[:2], gpu_p50[:2], "-o", color=GREEN, lw=2, ms=7, label="GPU M4 (MPS)")
    gh512, gh4k = maybe("gpu_cluster_bs512"), maybe("gpu_cluster_bs4000")
    if gh4k:
        a2.plot(bss[:2], [gh512["p50_step_ms"], gh4k["p50_step_ms"]], "-o",
                color=VERM, lw=2, ms=7, label="GPU GH200 (CUDA)")
    a2.set_xscale("log"); a2.set_xticks(bss); a2.set_xticklabels(bss)
    a2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a2.set_ylim(0)
    a2.set_xlabel("Batch size"); a2.set_ylabel("p50 step time (ms)")
    a2.set_title("Step time vs batch: TPU is flat (latency-bound)")
    a2.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig6_backends.png"); plt.close(fig)

print("figures written to", FIG)
