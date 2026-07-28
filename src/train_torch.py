"""MLP churn classifier in PyTorch for CPU / Apple MPS / CUDA backends.

Same architecture and telemetry as the JAX trainer so results are comparable
across the scaling matrix.

Example:
  python src/train_torch.py --data data/Churn_Dataset.csv --device mps
"""
import argparse
import json
import os
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--compile", action="store_true", help="torch.compile (Inductor)")
    p.add_argument("--out", default="")
    p.add_argument("--label", default="")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from torch import nn

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    from data import load_churn
    from telemetry import NodeSampler

    (Xtr, ytr), (Xte, yte) = load_churn(args.data)
    dev = torch.device(args.device)

    layers, d = [], Xtr.shape[1]
    for h in args.hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 2))
    model = nn.Sequential(*layers).to(dev)
    if args.compile:
        model = torch.compile(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).long().to(dev)
    n = len(Xtr)
    steps_per_epoch = max(1, n // args.batch_size)

    def sync():
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()

    # Warmup step (kernel/graph compilation, MPS shader build).
    t0 = time.perf_counter()
    xb, yb = Xtr_t[: args.batch_size], ytr_t[: args.batch_size]
    opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step(); sync()
    compile_s = time.perf_counter() - t0

    step_times = []
    g = torch.Generator().manual_seed(1)
    with NodeSampler() as sampler:
        t_train0 = time.perf_counter()
        for epoch in range(args.epochs):
            perm = torch.randperm(n, generator=g).to(dev)
            for s in range(steps_per_epoch):
                idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
                ts = time.perf_counter()
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                opt.step()
                sync()
                step_times.append(time.perf_counter() - ts)
        train_s = time.perf_counter() - t_train0

    with torch.no_grad():
        pred = model(torch.from_numpy(Xte).to(dev)).argmax(-1).cpu().numpy()
    acc = float((pred == yte).mean())

    mps_peak = None
    if args.device == "mps":
        mps_peak = round(torch.mps.driver_allocated_memory() / 2**20, 1)

    total_steps = len(step_times)
    result = {
        "label": args.label or f"torch-{args.device}",
        "framework": f"torch {torch.__version__}" + (" (inductor)" if args.compile else ""),
        "backend": args.device,
        "device": args.device,
        "threads": args.threads,
        "dtype": "float32",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps": total_steps,
        "compile_overhead_s": round(compile_s, 4),
        "train_wall_s": round(train_s, 3),
        "mean_step_ms": round(1e3 * sum(step_times) / total_steps, 4),
        "p50_step_ms": round(1e3 * sorted(step_times)[total_steps // 2], 4),
        "steps_per_sec": round(total_steps / train_s, 1),
        "samples_per_sec": round(total_steps * args.batch_size / train_s, 0),
        "test_accuracy": round(acc, 4),
        "gpu_mem_peak_mb": mps_peak,
        "telemetry": sampler.summary(),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "telemetry"}, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
