"""MLP churn classifier in JAX/Flax, jit-compiled through XLA.

Measures compile overhead (first-step time), per-step time, and epoch
throughput, alongside node telemetry. Results are written as JSON for the
benchmark matrix.

Examples:
  python src/train_jax.py --data data/Churn_Dataset.csv --backend cpu --threads 1
  python src/train_jax.py --data data/Churn_Dataset.csv --backend cpu --threads 10
"""
import argparse
import json
import os
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "tpu"])
    p.add_argument("--threads", type=int, default=0, help="XLA intra-op threads (0 = all)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bf16", action="store_true", help="compute in bfloat16")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--coordinator", default="")
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--process-id", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--label", default="")
    return p.parse_args()


def main():
    args = parse_args()

    # Backend/thread pinning must happen before importing jax.
    if args.threads > 0:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_cpu_multi_thread_eigen={'true' if args.threads > 1 else 'false'}"
            f" intra_op_parallelism_threads={args.threads}"
        )
    os.environ.setdefault("JAX_PLATFORMS", args.backend if args.backend != "gpu" else "cuda,cpu")

    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn

    if args.distributed:
        jax.distributed.initialize(args.coordinator, args.num_processes, args.process_id)

    from data import load_churn
    from telemetry import NodeSampler

    (Xtr, ytr), (Xte, yte) = load_churn(args.data)
    dtype = jnp.bfloat16 if args.bf16 else jnp.float32

    class MLP(nn.Module):
        hidden: tuple

        @nn.compact
        def __call__(self, x):
            x = x.astype(dtype)
            for h in self.hidden:
                x = nn.relu(nn.Dense(h, dtype=dtype)(x))
            return nn.Dense(2, dtype=dtype)(x).astype(jnp.float32)

    model = MLP(tuple(args.hidden))
    params = model.init(jax.random.PRNGKey(0), Xtr[:1])
    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, xb, yb):
        def loss_fn(p):
            logits = model.apply(p, xb)
            return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def accuracy(params, x, y):
        return (model.apply(params, x).argmax(-1) == y).mean()

    # Device-resident dataset (fits trivially in HBM/RAM).
    Xtr_d, ytr_d = jnp.asarray(Xtr), jnp.asarray(ytr)
    n = len(Xtr)
    steps_per_epoch = max(1, n // args.batch_size)

    # Compile overhead: time the first (traced + compiled) step separately.
    t0 = time.perf_counter()
    params, opt_state, _ = train_step(params, opt_state,
                                      Xtr_d[: args.batch_size], ytr_d[: args.batch_size])
    jax.block_until_ready(params)
    compile_s = time.perf_counter() - t0

    step_times = []
    rng = jax.random.PRNGKey(1)
    with NodeSampler() as sampler:
        t_train0 = time.perf_counter()
        for epoch in range(args.epochs):
            rng, k = jax.random.split(rng)
            perm = jax.random.permutation(k, n)
            for s in range(steps_per_epoch):
                idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
                ts = time.perf_counter()
                params, opt_state, loss = train_step(params, opt_state, Xtr_d[idx], ytr_d[idx])
                jax.block_until_ready(loss)
                step_times.append(time.perf_counter() - ts)
        train_s = time.perf_counter() - t_train0

    acc = float(accuracy(params, jnp.asarray(Xte), jnp.asarray(yte)))
    total_steps = len(step_times)
    result = {
        "label": args.label or f"jax-{args.backend}-t{args.threads}",
        "framework": f"jax {jax.__version__} (XLA)",
        "backend": args.backend,
        "device": str(jax.devices()[0]),
        "threads": args.threads,
        "dtype": "bfloat16" if args.bf16 else "float32",
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
