#!/usr/bin/env bash
# Start this node's raylet inside the primus container. Runs on the node.
#   02_ray_start_primus.sh head|worker
#
# The primus twin of the cluster's ray_start script: same shape, but it sources
# ray_env_primus.sh (RDMA on, shared NFS tree on PYTHONPATH) and defaults to the
# anp-primus container.
#
# ⚠️ Start the raylet explicitly rather than letting the driver's ray.init()
# create a local instance. Ray's AMD detection is run once at init and has been
# observed to come back empty on this image -- the raylet then comes up with no
# GPU in --static_resource_list and every num_gpus=1 actor hangs forever behind
# "No available node types can fulfill resource request {'GPU': 1.0}", with the
# driver still cheerfully logging "1 nodes x 8 GPUs" from the config. --num-gpus=8
# states it instead of detecting it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The per-allocation settings (JOBID / HEAD_NODE / HEAD_IP / NODES / NET_IF)
# change every job and are not in the repo. Point LUMEN_CLUSTER_ENV at wherever
# yours lives.
source "${LUMEN_CLUSTER_ENV:-/home/xysheng/4node/env.sh}"
ROLE="${1:?usage: ray_start_primus.sh head|worker}"
CONTAINER=${RL24_CONTAINER:-anp-primus}
MY_IP="$(ip -o -4 addr show "$NET_IF" | awk '{print $4}' | cut -d/ -f1)"

if [ "$ROLE" = head ]; then
  START="ray start --head --port=$RAY_PORT --node-ip-address=$MY_IP --num-gpus=8 \
    --dashboard-host=0.0.0.0 --disable-usage-stats"
else
  START="ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$MY_IP --num-gpus=8 \
    --disable-usage-stats"
fi

docker exec "$CONTAINER" bash -lc "
set -e
export RL_ROOT=$RL_ROOT DATA_ROOT=$DATA_ROOT SCRATCH_ROOT=$SCRATCH_ROOT
source $HERE/ray_env_primus.sh
ray stop --force >/dev/null 2>&1 || true
# ray stop reaps its own actors, but vLLM's EngineCore and its multiproc_executor
# workers are plain children and outlive it, holding GPU memory and the weight-sync
# IPC endpoint. Matched on the process NAME, not -f: this shell's own command line
# contains 'VLLM::', so pkill -f would kill the killer.
pkill -9 '^VLLM::' 2>/dev/null || true
sleep 3
$START
"
echo "ray $ROLE (primus env) started on $(hostname) ($MY_IP)"
