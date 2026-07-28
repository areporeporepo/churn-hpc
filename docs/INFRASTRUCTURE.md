# Phase 2 - Cluster Provisioning & Storage Orchestration

## Infrastructure mapping

Three provisioning targets are mapped; the same container images and manifests
run on all of them.

| Target | Nodes | Interconnect | Network policy |
|---|---|---|---|
| Local dev (Apple M4) | 1 node, 10 cores / 16 GB / MPS GPU | n/a | localhost only |
| Stanford ME344 SLURM (`hpcc-cluster-49`) | head + compute nodes, Rocky 9 | cluster-private subnet | reachable only via Stanford VPN; no inbound from internet; jobs submitted from head node |
| GKE (`soe-hpccenter` project) | GPU pool (L4) + TPU v5e 2x2 pool | GCP VPC-native | private cluster: nodes on a dedicated subnet with secondary ranges for pods/services; no public node IPs; egress via Cloud NAT; NetworkPolicy default-deny between namespaces |

VPC notes for the GKE target:

- Cluster subnet `10.10.0.0/20`, pod range `10.64.0.0/14`, service range
  `10.80.0.0/20` (VPC-native / alias IPs, so pod-to-pod traffic rides the VPC
  fabric directly rather than an overlay).
- TPU slice workers communicate over the ICI (inter-chip interconnect) inside
  the slice; only the dataset read crosses the VPC.
- SLURM multi-node jobs pin the JAX coordinator to the head compute node
  (`slurm/train_multinode.sbatch`) so internode traffic stays on the cluster's
  private fabric.

## Data ingestion tier

`Churn_Dataset.csv` is 341 KB, which means the danger is not bandwidth but
*per-step filesystem latency* (reading from NFS inside the training loop) and
redundant reads when jobs fan out. The tier is arranged so the training loop
never touches shared storage:

```
 canonical copy                 staging (per job)               training loop
 ───────────────                ─────────────────               ─────────────
 NFS  /home/.../data/  ──cp──►  node-local NVMe scratch  ──►  host RAM (numpy)
 GCS  gs://churn-hpc-datasets   $SLURM_TMPDIR / emptyDir        └─► device HBM
      (gcsfuse w/ file cache)   (local-SSD backed)
```

- **SLURM**: every sbatch script copies the CSV from the NFS submit dir to
  `$SLURM_TMPDIR` (node-local scratch) before training (`scripts/stage_data.sh`).
- **GKE GPU**: dataset PVC is Filestore (NFS, `ReadOnlyMany`); an `emptyDir` on
  local-SSD nodes provides NVMe scratch.
- **GKE TPU**: gcsfuse CSI mount with a 512 MB file cache, so the object-store
  read happens once per node.
- **In all cases** the trainer loads the CSV into RAM once, preprocesses to a
  dense `float32` matrix (~600 KB), and either feeds device-resident arrays
  (JAX: the whole dataset lives in HBM) or an in-memory dataloader (PyTorch).
  At this dataset size the IO bus is touched exactly once per job.
