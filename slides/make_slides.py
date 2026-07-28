"""Build slides/churn_hpc_slides.pdf - exactly 5 slides, 16:9 landscape."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
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


def bullets(ax, items, x=0.05, y=0.74, dy=0.072, fs=14.5, wrapcolor=INK):
    for head, rest in items:
        ax.text(x, y, "▪", fontsize=fs, color=BLUE, va="top")
        ax.text(x + 0.022, y, f"$\\bf{{{head}}}$ {rest}" if False else "", va="top")
        ax.annotate("", (0, 0))
        ax.text(x + 0.022, y, head, fontsize=fs, color=INK, fontweight="bold", va="top")
        ax.text(x + 0.022, y - 0.034, rest, fontsize=12.5, color=MUT, va="top", wrap=True)
        y -= dy + 0.028
    return y


with PdfPages(OUT) as pdf:
    # ── Slide 1: Problem ────────────────────────────────────────────────────
    fig, ax = new_slide(1, "Predicting Customer Churn at Infrastructure Scale", "PROBLEM")
    bullets(ax, [
        ("Business problem:", "telecom churn costs 5-25x more to replace than retain; predict the 14% of customers who will leave from usage behavior."),
        ("Dataset:", "Churn_Dataset.csv - 5,000 customers, 16 numeric features, 341 KB. Model: MLP 128-64-32 (JAX/Flax + PyTorch), 92-94% test accuracy vs 86% majority baseline."),
        ("Resource challenge:", "the dataset is tiny but the pipeline must be production-shaped: every hardware tier is oversized, so FIXED costs (compile, kernel dispatch, staging) dominate."),
        ("Question:", "which rung of the hardware ladder (1 CPU core → 10 cores → GPU → cluster GPU/TPU) does this workload actually need, and what is the real bottleneck?"),
        ("Constraint:", "Stanford SLURM cluster (hpcc-cluster-49) is VPN-gated and was unreachable; cluster rungs are provisioned as manifests, local rungs are measured."),
    ], y=0.76)
    pdf.savefig(fig); plt.close(fig)

    # ── Slide 2: Proposal & Solutions ───────────────────────────────────────
    fig, ax = new_slide(2, "Immutable Containers, Declarative Orchestration, XLA", "PROPOSAL & SOLUTIONS")
    bullets(ax, [
        ("Environment (Docker):", "two pinned images - python:3.12-slim CPU and CUDA 12.4+cuDNN - all deps baked at build; nothing installed on nodes; Apptainer .sif on SLURM."),
        ("Orchestration (K8s + SLURM):", "GKE Jobs with exact requests/limits (8 CPU / 24 GiB / 1×L4; TPU v5e 2×2 = 4 chips) · sbatch scripts with --cpus-per-task/--mem/--gres · 2-node jax.distributed."),
        ("Storage tier:", "NFS / GCS canonical copy → staged once to node-local NVMe scratch ($SLURM_TMPDIR, local-SSD emptyDir, gcsfuse cache) → RAM → device memory. Training loop never touches shared storage."),
        ("Compilation (JAX/XLA):", "entire train step (fwd + bwd + Adam) jit-fused into one XLA computation; torch.compile (Inductor) wired for the accelerator path; bf16 flag for A100/TPU rungs."),
    ], y=0.76)
    pdf.savefig(fig); plt.close(fig)

    # ── Slide 3: Measurements ───────────────────────────────────────────────
    fig, ax = new_slide(3, "Telemetry: Honest Step Times + Node Metrics", "MEASUREMENTS")
    bullets(ax, [
        ("Scaling matrix:", "3 discrete scale steps - single-core CPU (JAX/XLA), full 10-core node (JAX/XLA), Apple M4 GPU (PyTorch/MPS) - plus batch sweeps and a bf16 ablation: 12 configs × 400 epochs."),
        ("Step timing:", "per-step wall clock with forced device sync (jax.block_until_ready / torch.mps.synchronize); first traced+compiled step timed separately = compile overhead."),
        ("Node telemetry:", "background psutil sampler at 5 Hz: whole-node CPU %, process RSS; GPU driver memory via torch.mps; nvidia-smi 1 Hz hook baked into the SLURM GPU job."),
        ("IO metrics:", "ingestion tier is staged + device-resident by design → input pipeline contributes zero steady-state stalls (verified: one 341 KB read per job)."),
        ("Output:", "every run emits one JSON record → pandas matrix → figures → this deck. Fully scripted: benchmarks/run_matrix.sh."),
    ], y=0.76)
    pdf.savefig(fig); plt.close(fig)

    # ── Slide 4: Results (hardware comparison) ──────────────────────────────
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, color=BLUE))
    ax.text(0.045, 0.895, "RESULTS - HARDWARE COMPARISON", fontsize=13, color=BLUE, fontweight="bold", va="top")
    ax.text(0.045, 0.86, "10 Cores Buy Nothing; Batch Size Buys 5.2x", fontsize=26, color=INK, fontweight="bold", va="top")
    ax.text(0.968, 0.025, "4 / 5", fontsize=10, color=MUT, ha="right")
    ax.text(0.045, 0.025, "churn-hpc · Anh Quang Nguyen · July 2026", fontsize=10, color=MUT)

    a1 = fig.add_axes([0.06, 0.22, 0.42, 0.52])
    configs = ["CPU 1c\nJAX", "CPU 10c\nJAX", "CPU 10c\nTorch", "M4 GPU\nbs512", "M4 GPU\nbs4000"]
    vals = [0.897, 0.898, 0.938, 0.394, 2.013]
    cols = [BLUE, BLUE, ORANGE, GREEN, GREEN]
    bars = a1.bar(configs, vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        a1.annotate(f"{v:.2f}M", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=11, fontweight="bold")
    a1.set_ylabel("samples / sec (millions)")
    a1.set_title("Training throughput (400 epochs)", fontsize=13)
    a1.spines[["top", "right"]].set_visible(False); a1.grid(axis="y", alpha=0.25)
    a1.set_axisbelow(True)

    a2 = fig.add_axes([0.56, 0.22, 0.40, 0.52])
    speedup = {"10 cores\nvs 1 core": 1.00, "GPU vs CPU\n(bs 512)": 0.42,
               "GPU bs 4000\nvs bs 512": 5.15, "GPU bs 4000\nvs best CPU": 2.14}
    cols2 = [BLUE, VERM, GREEN, GREEN]
    bars = a2.bar(list(speedup), list(speedup.values()), color=cols2, width=0.62)
    for b, v in zip(bars, speedup.values()):
        a2.annotate(f"{v:.2f}x", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=11, fontweight="bold")
    a2.axhline(1.0, color=MUT, lw=1, ls="--")
    a2.set_ylabel("speedup factor")
    a2.set_title("Speedup factors (dashed = parity)", fontsize=13)
    a2.spines[["top", "right"]].set_visible(False); a2.grid(axis="y", alpha=0.25)
    a2.set_axisbelow(True)

    ax.text(0.05, 0.095, "Utilization never saturated: node CPU ≤ 25%, GPU memory 19 MB, peak RSS 0.44 GB / 16 GB.\n"
                         "Compile overhead 0.16 s (JAX) vs 0.01 s (eager torch). bf16 on M4 CPU: 0.75x, a measured regression.",
            fontsize=11, color=MUT, va="top")
    pdf.savefig(fig); plt.close(fig)

    # ── Slide 5: Conclusion ─────────────────────────────────────────────────
    fig, ax = new_slide(5, "The Bottleneck Is Dispatch Latency, Not Hardware", "CONCLUSION")
    bullets(ax, [
        ("Bottleneck identified:", "fixed per-step host-side launch cost. Evidence: core-count indifference (1.00x), GPU step 3x CPU step for identical math, throughput ~linear in batch size, no resource saturated."),
        ("What worked:", "batch 512→4000 amortized launch cost for 5.15x; XLA fusion gave 0.4 ms CPU steps; device-resident data removed IO entirely. What didn't: bf16 on CPU (0.75x), more cores (1.00x)."),
        ("Cost trade-off:", "GPU saves ~0.8 s per run but costs ~4x more per hour than a CPU node - accelerators earn their premium here only for parallel sweeps, not for shrinking one small job."),
        ("Scaling recommendation:", "< ~1M rows: one CPU node with XLA. Grow batch (with LR warmup) before growing hardware; fuse epochs with jax.lax.scan next. GPU/TPU + multinode manifests are provisioned for the real rung: cluster-scale sweeps once VPN access returns."),
        ("Lesson:", "measure fixed costs before buying parallelism - the scaling matrix falsified two 'obvious' upgrades (more cores, an accelerator at default batch) in under 30 s of compute."),
    ], y=0.76)
    pdf.savefig(fig); plt.close(fig)

print("wrote", OUT)
