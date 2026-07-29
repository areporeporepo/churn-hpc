#!/bin/bash
# Watch-and-heal loop for the cluster GPU job (churn-train-gpu).
# - Polls job state via the hpcc head node every INTERVAL seconds.
# - Emits one line per state change (Pending -> Running -> Completed).
# - If the pod fails (image flake, bootstrap error), resubmits up to MAX_RESUBMITS.
# - Exits 0 on Completed, 1 on giving up.
# Usage: scripts/watch_gpu_job.sh [interval_seconds]
set -uo pipefail

HEAD=admin@hpcc-cluster-49.stanford.edu
INTERVAL=${1:-60}
MAX_RESUBMITS=3
resubmits=0
prev=""

remote() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD" \
  "export KUBECONFIG=~/student49.kubeconfig; $*" 2>&1; }

while true; do
  raw=$(remote "kubectl get pods -l app=churn-hpc --no-headers")
  if echo "$raw" | grep -qiE "refused|unable to connect|timed out|error from server|no route"; then
    st="ApiDown"          # transient control-plane/VPN flake: never resubmit on this
    line="$raw"
  else
    line=$(echo "$raw" | grep churn-train-gpu || true)
    st=$(echo "$line" | awk '{print $3}')
    [ -z "$st" ] && st="NoPod"
  fi
  if [ "$st" != "$prev" ]; then
    echo "GPU JOB: $st"
    prev="$st"
  fi

  case "$st" in
    Completed)
      echo "GPU DONE: $line"
      exit 0
      ;;
    ApiDown)
      ;;
    Error|Failed|CrashLoopBackOff|ImagePullBackOff|NoPod)
      if [ "$resubmits" -ge "$MAX_RESUBMITS" ]; then
        echo "GPU GAVE UP after $resubmits resubmits: $line"
        exit 1
      fi
      resubmits=$((resubmits + 1))
      echo "GPU RESUBMIT #$resubmits (state was $st)"
      remote "kubectl logs job/churn-train-gpu --tail=5" | sed 's/^/  last-log: /'
      remote "kubectl delete job churn-train-gpu --ignore-not-found >/dev/null; kubectl apply -f /tmp/churn-k8s/train-gpu-job.yaml" >/dev/null
      prev=""
      ;;
  esac
  sleep "$INTERVAL"
done
