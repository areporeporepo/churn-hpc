"""Fetch churn-hpc K8s job logs from the hpcc head node and extract result JSONs.

Usage: python scripts/fetch_cluster_results.py [job ...]   (default: both jobs)
Writes benchmarks/results/<label>.json and telemetry/<job>_smi.csv (GPU job).
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEAD = "admin@hpcc-cluster-49.stanford.edu"


def fetch(job):
    cmd = ["ssh", "-o", "BatchMode=yes", HEAD,
           f"export KUBECONFIG=~/student49.kubeconfig; kubectl logs job/{job} --tail=-1"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        print(f"{job}: log fetch failed: {out.stderr.strip()[:200]}")
        return
    text = out.stdout
    n = 0
    for chunk in text.split("===RESULT ")[1:]:
        body = chunk.split("===", 1)[1]
        payload = body[: body.find("\n===")] if "\n===" in body else body
        rec = json.loads(payload.strip().splitlines()[0])
        dest = ROOT / "benchmarks" / "results" / f"{rec['label'].replace('-', '_')}.json"
        dest.write_text(json.dumps(rec))
        print(f"{job}: wrote {dest.name}  samples/s={rec['samples_per_sec']:.0f} "
              f"p50={rec['p50_step_ms']}ms acc={rec['test_accuracy']}")
        n += 1
    if "===SMI===" in text:
        smi = text.split("===SMI===", 1)[1].strip()
        (ROOT / "telemetry" / f"{job}_smi.csv").write_text(smi + "\n")
        print(f"{job}: wrote telemetry/{job}_smi.csv ({len(smi.splitlines())} samples)")
    if "===GPU===" in text:
        print(f"{job}: GPU = {text.split('===GPU===',1)[1].splitlines()[2]}")
    if not n:
        print(f"{job}: no RESULT blocks yet")


for job in sys.argv[1:] or ["churn-train-cpu", "churn-train-gpu"]:
    fetch(job)
