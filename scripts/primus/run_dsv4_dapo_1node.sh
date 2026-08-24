#!/usr/bin/env bash
# DeepSeek-V4 DAPO on the megatron_native backend, ONE node / 8 GPUs, on primus.
# Runs on the node (docker host side).
#
# The primus replacement for ~/4node/07_dsv4_megatron_1node.sh, which predates
# the primus migration and sources the 22.04 ray_env_dsv4_megatron.sh. Two lines
# in that file are fatal here: NCCL_IB_DISABLE=1 throws away the 26x RDMA fabric,
# and HSA_DISABLE_FRAGMENT_ALLOCATOR=1 breaks intra-node reduce-scatter, which is
# how Megatron's distributed optimizer reduces gradients. See
# docs/agent/06-primus-pitfalls.md.
#
#   RAY=1 DETACH=1 STEPS=3 \
#   CFG=examples/DAPO/configs/dapo_dsv4_flash_ray_vllm_megatron_1node_shortsmoke.yaml \
#     bash scripts/primus/run_dsv4_dapo_1node.sh
#
# DETACH=1 hands the driver to the docker daemon rather than holding it on this
# script's stdout: reached over `spur exec`, the driver is a child of a job step
# that tears its children down when it ends. run_dapo.sh still writes $LOG.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Per-allocation settings (JOBID / HEAD_NODE / HEAD_IP / NODES / NET_IF); these
# change every job and are not in the repo.
source "${LUMEN_CLUSTER_ENV:-/home/xysheng/4node/env.sh}"
mkdir -p "$LOG4N"

CONTAINER=${RL24_CONTAINER:-anp-primus}
# BF16 weights under the checkpoint's own tensor names (make_native_bf16.py):
# vLLM's DSv4 loader keys on those names, and a BF16 rollout takes the trainer's
# BF16 weight sync unquantized.
MODEL_PATH="${MODEL_PATH:-/mnt/m2m_nobackup/xysheng/models/dsv4-L4-bf16-native}"
CFG="${CFG:-examples/DAPO/configs/dapo_dsv4_flash_ray_vllm_megatron_1node_shortsmoke.yaml}"
STEPS="${STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG="${LOG:-$LOG4N/dsv4-megatron-1node-primus.log}"
MY_IP="$(ip -o -4 addr show "$NET_IF" | awk '{print $4}' | cut -d/ -f1)"

# Diagnostics that have to reach the ACTORS, not just the driver: the coverage
# assertion runs inside the vLLM worker, which inherits the raylet's environment.
# Setting these only around the driver has no effect on it.
PASSTHROUGH=""
for v in LUMENRL_WEIGHT_SYNC_CHECK LUMENRL_WEIGHT_SYNC_VERIFY LUMENRL_LOGGING_LEVEL; do
  [ -n "${!v:-}" ] && PASSTHROUGH+="export $v=${!v}; "
done

if [ "${RAY:-0}" = 1 ]; then
  RAY_EXTRA_EXPORTS="$PASSTHROUGH" RAY_ENV_SCRIPT="$HERE/ray_env_dsv4_primus.sh" \
    bash "$HERE/ray_start_primus.sh" head
fi

docker exec ${DETACH:+-d} "$CONTAINER" bash -lc "
set -uo pipefail
exec > >(tee -a '$LOG4N/driver-bootstrap-1node-primus.log') 2>&1
export RL_ROOT=$RL_ROOT DATA_ROOT=$DATA_ROOT
source $HERE/ray_env_dsv4_primus.sh
$PASSTHROUGH
# Swallows run_dapo.sh's opening 'ray stop --force', which would tear down the
# head this driver is about to connect to.
export PATH=/home/xysheng/4node/bin:\$PATH
export SCRATCH_ROOT=\"\${SCRATCH_ROOT:-\$DATA_ROOT}\"
# Both are '\${VAR-default}' inside run_dapo.sh, so an explicit empty value is
# the only way to turn them off. ROCm/HIP has no expandable_segments, and
# HSA_DISABLE_FRAGMENT_ALLOCATOR is the reduce-scatter killer described above --
# leaving it set fails far from its cause, in a one-element torch.zeros in
# clip_grads.py.
export PYTORCH_CUDA_ALLOC_CONF= HSA_DISABLE_FRAGMENT_ALLOCATOR=

CONFIG_OVERRIDE='$CFG' \
MODEL_PATH='$MODEL_PATH' \
TRAIN_FILE=\$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet \
VAL_FILE='' \
STEPS=$STEPS \
MODE=bf16 \
LOG='$LOG' \
EXTRA_OVERRIDE='cluster.num_nodes=1 cluster.gpus_per_node=8 cluster.ray_address=$MY_IP:$RAY_PORT controller.ray.actor.num_workers=$NUM_WORKERS ${EXTRA:-}' \
  bash \$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh
"
echo "=== log: $LOG ==="
