"""Build slides/churn_hpc_slides.pdf - exactly 5 slides, 16:9 landscape."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "slides" / "churn_hpc_slides.pdf"

BLUE, ORANGE, GREEN, VERM, INK, MUT = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#1a1a2e", "#555"
W, H = 13.333, 7.5  # 16:9 inches


def new_slide(pdf_num, title, kicker):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, color=BLUE))
    ax.text(0.045, 0.895, kicker, fontsize=13, color=BLUE, fontweight="bold", va="top")
    ax.text(0.045, 0.86, title, fontsize=26, color=INK, fontweight="bold", va="top")
    ax.text(0.968, 0.025, f"{pdf_num} / 5", fontsize=10, color=MUT, ha="right")
    ax.text(0.045, 0.025, "churn-hpc · Anh Quang Nguyen · July 2026", fontsize=10, color=MUT)
    return fig, ax


def bullets(ax, items, x=0.05, y=0.74, dy=0.072, fs=14.5):
    for head, rest in items:
        ax.text(x, y, "▪", fontsize=fs, color=BLUE, va="top")
        ax.text(x + 0.022, y, head, fontsize=fs, color=INK, fontweight="bold", va="top")
        ax.text(x + 0.022, y - 0.034, rest, fontsize=12.5, color=MUT, va="top", wrap=True)
        y -= dy + 0.028
    return y


with PdfPages(OUT) as pdf:
    # ── Slide 1: Problem ────────────────────────────────────────────────────
    fig, ax = new_slide(1, "Predicting Customer Churn at Infrastructure Scale", "PROBLEM")
    bullets(ax, [
        ("Business problem:", "telecom churn costs 5-25x more to replace than retain; predict the 14% of customers who will leave from usage behavior."),
        ("Dataset:", "Churn_Dataset.csv: 5,000 customers, 16 numeric features, 341 KB. Model: MLP 128-64-32 (JAX/Flax + PyTorch), 91-94% test accuracy vs 86% majority baseline."),
        ("Resource challenge:", "the dataset is tiny but the pipeline must be production-shaped: every hardware tier is oversized, so FIXED costs (compile, kernel dispatch, staging) dominate."),
        ("Question:", "which rung of the hardware ladder (1 CPU core → many cores → GPU → TPU slice) does this workload actually need, and what is the real bottleneck?"),
        ("Scope delivered:", "15 measured configs across 4 architectures (Apple M4, arm64 Grace, x86 Xeon, TPU v5e) and 3 sites (local, Stanford hpcc cluster, Google Cloud)."),
    ], y=0.76)
    pdf.savefig(fig); plt.close(fig)

    # ── Slide 2: Proposal & Solutions ───────────────────────────────────────
    fig, ax = new_slide(2, "Immutable Containers, Declarative Orchestration, XLA", "PROPOSAL & SOLUTIONS")
    bullets(ax, [
        ("Environment (Docker/Apptainer):", "pinned images (CPU, CUDA 12.4, arm64/cu128); zero installs on nodes. SLURM runs the same pinned image as a read-only 3.2 GB .sif; K8s jobs bootstrap pinned wheels in ephemeral containers."),
        ("Orchestration (K8s + SLURM):", "on-prem K8s Jobs with exact requests/limits (32c Grace slice, 1×NVIDIA GPU) · GKE Autopilot TPU v5e 2x2 provisioned on demand from a nodeSelector · single-node SLURM (slurmctld+slurmd+munge) installed and configured on hpcc-cluster-49."),
        ("Storage tier:", "NFS / ConfigMap / GCS canonical copy → staged once to node-local scratch → RAM → device memory. The training loop never touches shared storage; the IO bus sees the CSV once per job."),
        ("Compilation (JAX/XLA):", "entire train step (fwd + bwd + Adam) jit-fused into one XLA computation on CPU and TPU; PyTorch eager (+ optional Inductor) on the GPU path; bf16 as a measured ablation."),
    ], y=0.76)
    pdf.savefig(fig); plt.close(fig)

    # ── Slide 3: Measurements ───────────────────────────────────────────────
    fig, ax = new_slide(3, "Telemetry: Honest Step Times + Node Metrics", "MEASUREMENTS")
    bullets(ax, [
        ("Scaling matrix:", "core sweeps (1/8/10/32) on three CPU architectures, batch sweeps (128→16,384) on GPU and TPU, bf16 ablation: 15 measured configs, 400 epochs each, one JSON record per run."),
        ("Step timing:", "per-step wall clock with forced device sync (jax.block_until_ready / torch.mps.synchronize / TPU); first traced+compiled step timed separately = compile overhead."),
        ("Node telemetry:", "background psutil sampler (whole-node CPU %, RSS); GPU memory via driver; nvidia-smi 1 Hz hook in the cluster GPU job; environment bootstrap timed (25-36 s ephemeral, 3.2 GB .sif once)."),
        ("IO metrics:", "staged + device-resident by design → zero steady-state input stalls, verified on every backend (one 341 KB read per job)."),
        ("Automation:", "benchmarks/run_matrix.sh, watch-and-heal job loops, fetch_cluster_results.py: every number in this deck regenerates from committed JSON."),
    ], y=0.76)
    pdf.savefig(fig); plt.close(fig)

    # ── Slide 4: Results (CPU vs GPU vs TPU) ────────────────────────────────
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, color=BLUE))
    ax.text(0.045, 0.895, "RESULTS - CPU vs GPU vs TPU", fontsize=13, color=BLUE, fontweight="bold", va="top")
    ax.text(0.045, 0.86, "Cores Buy Nothing; Batch Size Buys Everything", fontsize=26, color=INK, fontweight="bold", va="top")
    ax.text(0.968, 0.025, "4 / 5", fontsize=10, color=MUT, ha="right")
    ax.text(0.045, 0.025, "churn-hpc · Anh Quang Nguyen · July 2026", fontsize=10, color=MUT)

    # Panel 1: peak throughput per backend (best measured config)
    a1 = fig.add_axes([0.055, 0.22, 0.27, 0.52])
    peaks = [("CPU\nM4 10c", 0.938, BLUE), ("CPU\nXeon 8c\nbs4000", 0.540, BLUE),
             ("GPU\nM4 bs4000", 2.013, GREEN), ("TPU v5e\nbs16384", 3.216, ORANGE)]
    bars = a1.bar([p[0] for p in peaks], [p[1] for p in peaks], color=[p[2] for p in peaks], width=0.62)
    for b, (_, v, _) in zip(bars, peaks):
        a1.annotate(f"{v:.2f}M", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=10.5, fontweight="bold")
    a1.set_ylabel("peak samples/sec (millions)", fontsize=10)
    a1.set_title("Peak throughput per backend", fontsize=12)
    a1.tick_params(labelsize=8.5)
    a1.spines[["top", "right"]].set_visible(False); a1.grid(axis="y", alpha=0.25); a1.set_axisbelow(True)

    # Panel 2: step time vs batch (TPU flat)
    a2 = fig.add_axes([0.385, 0.22, 0.27, 0.52])
    bss = [512, 4000, 16384]
    a2.plot(bss, [2.088, 2.144, 2.148], "-o", color=ORANGE, lw=2, ms=6, label="TPU v5e")
    a2.plot(bss[:2], [0.395, 1.526], "-o", color=BLUE, lw=2, ms=6, label="CPU M4 (XLA)")
    a2.plot(bss[:2], [1.244, 1.739], "-o", color=GREEN, lw=2, ms=6, label="GPU M4 (MPS)")
    a2.set_xscale("log"); a2.set_xticks(bss); a2.set_xticklabels(bss, fontsize=8.5)
    a2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a2.set_ylim(0, 2.5); a2.tick_params(labelsize=8.5)
    a2.set_xlabel("batch size", fontsize=10); a2.set_ylabel("p50 step time (ms)", fontsize=10)
    a2.set_title("Step time vs batch: TPU is flat", fontsize=12)
    a2.legend(frameon=False, fontsize=8.5)
    a2.spines[["top", "right"]].set_visible(False); a2.grid(alpha=0.25); a2.set_axisbelow(True)

    # Panel 3: core-scaling futility (throughput vs cores, bs512)
    a3 = fig.add_axes([0.715, 0.22, 0.265, 0.52])
    scaling = [("M4\n1c", 0.897, BLUE), ("M4\n10c", 0.898, BLUE),
               ("Xeon\n1c", 0.114, VERM), ("Xeon\n32c", 0.107, VERM),
               ("Grace\n1c", 0.196, ORANGE), ("Grace\n32c", 0.028, ORANGE)]
    bars = a3.bar([s[0] for s in scaling], [s[1] for s in scaling], color=[s[2] for s in scaling], width=0.62)
    for b, (_, v, _) in zip(bars, scaling):
        a3.annotate(f"{v*1e3:.0f}K", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
    a3.set_ylabel("samples/sec (millions)", fontsize=10)
    a3.set_title("Adding cores: flat or negative", fontsize=12)
    a3.tick_params(labelsize=8.5)
    a3.spines[["top", "right"]].set_visible(False); a3.grid(axis="y", alpha=0.25); a3.set_axisbelow(True)

    ax.text(0.05, 0.115, "Speedups (bs512 → best): GPU 5.15x via batch alone · TPU 17.0x via batch alone (constant 2.1 ms step) · 10x/32x cores: 1.00x / 0.94x / 0.14x.\n"
                         "Xeon 32-core run: 99.7% node utilization, less throughput than 1 core. Utilization is not throughput.",
            fontsize=11, color=MUT, va="top")
    pdf.savefig(fig); plt.close(fig)

    # ── Slide 5: Conclusion ─────────────────────────────────────────────────
    fig, ax = new_slide(5, "The Bottleneck Is Dispatch Latency, Not Hardware", "CONCLUSION")
    bullets(ax, [
        ("Bottleneck identified:", "fixed per-step host-side launch cost, confirmed on four architectures: core-count indifference (M4), negative core scaling (Grace 0.14x, Xeon at 100% util), and a TPU step time constant across a 32x batch range."),
        ("What worked:", "batch scaling (GPU 5.15x, TPU 17.0x), XLA train-step fusion, device-resident data (zero IO stalls). What didn't: more cores (≤1.00x everywhere), bf16 on CPU (0.75x), accelerators at default batch (0.2-0.4x)."),
        ("Cost trade-off:", "TPU v5e wins throughput 3.4x over the best CPU but costs ~10x a CPU node-hour and 7 min of provisioning; below ~1M rows a single CPU node with XLA is cost-optimal for one job."),
        ("Scaling recommendation:", "grow batch (with LR warmup) before growing hardware; fuse epochs with jax.lax.scan; spend accelerator manifests on parallel hyperparameter sweeps, where the TPU's 3.22M samples/s actually converts to wall-clock wins."),
        ("Lesson:", "measure fixed costs before buying parallelism: the scaling matrix falsified three 'obvious' upgrades (more cores, GPU at default batch, bf16-everywhere) with under a minute of total compute."),
    ], y=0.76)
    pdf.savefig(fig); plt.close(fig)

print("wrote", OUT)
