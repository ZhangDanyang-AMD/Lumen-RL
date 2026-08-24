#!/usr/bin/env bash
#SBATCH --job-name=k3-sddd-ray
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=k3-sddd-ray-%j.out
#SBATCH --error=k3-sddd-ray-%j.err

# Ray places four TP=8 ATOM teachers and one 8-rank draft torchrun. Teacher
# hidden states use pinned-host Mooncake RDMA; draft gradients use local RCCL.
#
# Submit on any five of the requested machines (first four become teachers,
# the final hostname in Ray's sorted topology becomes the draft node):
#   sbatch --nodelist=crsuse2-m2m-v2-[034-038] \
#     examples/Kimi_K3_SDDD_MI350_ATOM/run_multinode_slurm.sh
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch or run it inside an allocation.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPECTED_NODES="${SLURM_NNODES:-5}"
# Compute nodes have neither scontrol nor sinfo. Expand the bracketed SLURM
# nodelist locally so enumeration does not create a nested srun job step.
mapfile -t NODES < <(
    python3 - "${SLURM_JOB_NODELIST}" <<'PY'
import re
import sys

value = sys.argv[1]
match = re.fullmatch(r"([^\[]+)\[([^\]]+)\]", value)
if match is None:
    print(value)
    raise SystemExit

prefix, body = match.groups()
for item in body.split(","):
    if "-" not in item:
        print(prefix + item)
        continue
    start, end = item.split("-", 1)
    width = max(len(start), len(end))
    for number in range(int(start), int(end) + 1):
        print(prefix + str(number).zfill(width))
PY
)
NUM_NODES="${#NODES[@]}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
HEAD_NODE="${NODES[0]}"
HEAD_IP="$(srun --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" hostname -I | awk '{print $1}')"
RAY_PORT="${RAY_PORT:-6379}"
CONTAINER_NAME="${CONTAINER_NAME:-kimi-k3-sddd-ray-${SLURM_JOB_ID}}"

DATA_ROOT="${DATA_ROOT:-/mnt/m2m_nobackup/danyzhan}"
SHARED_ROOT="${SHARED_ROOT:-/shared_nfs/danyzhan/lumenrl}"
MODEL_PATH="${MODEL_PATH:-${DATA_ROOT}/models/Kimi-K3}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/datasets/ATOM_regen_seeklight_kimi_mtp/data/train.jsonl}"
CKPT_DIR="${CKPT_DIR:-${SHARED_ROOT}/checkpoints/kimi_k3_dspark_atom_ray_${NUM_NODES}n}"
CACHE_DIR="${CACHE_DIR:-${SHARED_ROOT}/cache/kimi_k3_teacher_ray_${NUM_NODES}n}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-${DATA_ROOT}/cache/lumenrl_tokenized}"
LOG_DIR="${LOG_DIR:-${SHARED_ROOT}/logs/kimi_k3_dspark_atom_ray_${SLURM_JOB_ID}}"
DOCKER_IMAGE="${DOCKER_IMAGE:-kimi_k3_dspark_atom:latest}"
CONFIG="${CONFIG:-${REPO_ROOT}/examples/Kimi_K3_SDDD_MI350_ATOM/configs/train.yaml}"
SMOKE_TEST="${SMOKE_TEST:-0}"
export MOONCAKE_DEVICE_NAME="${MOONCAKE_DEVICE_NAME:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-${MOONCAKE_DEVICE_NAME}}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-spur0}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-2GB}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-1GB}"
export LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE="${LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE:-2GB}"
export LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE="${LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE:-128}"
export LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE="${LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE:-2GB}"
export LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS="${LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS:-300}"

DRAFT_WORLD_SIZE="${GPUS_PER_NODE}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
if (( NUM_NODES != 5 )); then
    echo "This launcher requires exactly 5 nodes, got ${NUM_NODES}." >&2
    exit 2
fi
if (( GLOBAL_BATCH_SIZE % DRAFT_WORLD_SIZE != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by draft world size ${DRAFT_WORLD_SIZE}." >&2
    exit 2
fi
# Jim's branch aligns these with the TorchSpec reference as one recipe. Do not
# linearly rescale the learning rate from the previous bs=64 run.
LEARNING_RATE="${LEARNING_RATE:-5.0e-5}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    CONFIG="${REPO_ROOT}/examples/Kimi_K3_SDDD_MI350_ATOM/configs/smoke_test.yaml"
fi

for node in "${NODES[@]}"; do
    for required in "${MODEL_PATH}" "${DATASET_PATH}" "${REPO_ROOT}"; do
        if ! srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            test -e "${required}"; then
            echo "Required path does not exist on ${node}: ${required}" >&2
            exit 2
        fi
    done
    if ! srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
        docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
        echo "Docker image ${DOCKER_IMAGE} is missing on ${node}." >&2
        exit 2
    fi
done
srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
    mkdir -p "${CKPT_DIR}" "${CACHE_DIR}" "${TOKEN_CACHE_DIR}" "${LOG_DIR}"

OVERRIDES=(
    "cluster.num_nodes=1"
    "cluster.gpus_per_node=${GPUS_PER_NODE}"
    "controller.ray.enabled=false"
    "policy.model_name=${MODEL_PATH}"
    "policy.train_global_batch_size=${GLOBAL_BATCH_SIZE}"
    "policy.learning_rate=${LEARNING_RATE}"
    "algorithm.teacher.model_name=${MODEL_PATH}"
    "reward.dataset=${DATASET_PATH}"
    "dataset.cache_dir=${TOKEN_CACHE_DIR}"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
)
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    read -r -a USER_OVERRIDES <<< "${EXTRA_OVERRIDES}"
    OVERRIDES+=("${USER_OVERRIDES[@]}")
fi
printf -v OVERRIDE_STRING '%q ' "${OVERRIDES[@]}"

cleanup() {
    for node in "${NODES[@]}"; do
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
            docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT
cleanup

COMMON_DOCKER_ARGS=(
    --name "${CONTAINER_NAME}"
    --network host
    --ipc host
    --ulimit memlock=-1:-1
    --device /dev/kfd
    --device /dev/dri
    -v /dev/infiniband:/dev/infiniband
    -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro
    -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro
    --group-add video
    --group-add render
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    -v /dev/shm:/dev/shm
    -v "${DATA_ROOT}:${DATA_ROOT}"
    -v "${SHARED_ROOT}:${SHARED_ROOT}"
    -v "${REPO_ROOT}:/root/lumenrl"
    -w /root/lumenrl
    -e PYTHONUNBUFFERED=1
    -e NCCL_TIMEOUT=7200
    -e RAY_DEDUP_LOGS=0
)
for rdma_device in /dev/infiniband/*; do
    if [[ -c "${rdma_device}" ]]; then
        COMMON_DOCKER_ARGS+=(--device "${rdma_device}:${rdma_device}")
    fi
done
for name in HF_TOKEN NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX \
    NCCL_IB_DISABLE NCCL_NET_GDR_LEVEL NCCL_DMABUF_ENABLE \
    MOONCAKE_DEVICE_NAME MOONCAKE_GLOBAL_SEGMENT_SIZE \
    MOONCAKE_LOCAL_BUFFER_SIZE MOONCAKE_KV_LEASE_TTL_S \
    LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE \
    LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE \
    LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE \
    LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS; do
    if [[ -n "${!name:-}" ]]; then
        COMMON_DOCKER_ARGS+=(-e "${name}=${!name}")
    fi
done

echo "Starting Ray head ${HEAD_NODE} (${HEAD_IP}:${RAY_PORT})"
srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
    docker run -d "${COMMON_DOCKER_ARGS[@]}" "${DOCKER_IMAGE}" \
    ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
        --num-gpus="${GPUS_PER_NODE}" --block

for node in "${NODES[@]:1}"; do
    node_ip="$(srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" hostname -I | awk '{print $1}')"
    echo "Starting Ray worker ${node} (${node_ip})"
    srun --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
        docker run -d "${COMMON_DOCKER_ARGS[@]}" "${DOCKER_IMAGE}" \
        ray start --address="${HEAD_IP}:${RAY_PORT}" --node-ip-address="${node_ip}" \
            --num-gpus="${GPUS_PER_NODE}" --block
done

for _ in {1..60}; do
    active_nodes="$(srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
        docker exec "${CONTAINER_NAME}" python3 -c \
        'import ray; ray.init(address="auto"); print(sum(1 for n in ray.nodes() if n["Alive"]))' \
        2>/dev/null | awk 'NF {value=$NF} END {print value+0}')"
    [[ "${active_nodes}" == "${NUM_NODES}" ]] && break
    sleep 2
done
if [[ "${active_nodes}" != "${NUM_NODES}" ]]; then
    echo "Ray cluster has ${active_nodes}/${NUM_NODES} active nodes." >&2
    exit 1
fi

echo "Ray cluster ready: 4 teacher nodes + 1 draft node, $((NUM_NODES * GPUS_PER_NODE)) GPUs"
srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
    docker exec \
    -e LUMENRL_NUM_NODES="${NUM_NODES}" \
    -e LUMENRL_GPUS_PER_NODE="${GPUS_PER_NODE}" \
    -e LUMENRL_REPO_ROOT=/root/lumenrl \
    -e LUMENRL_CONFIG="/root/lumenrl/${CONFIG#${REPO_ROOT}/}" \
    -e LUMENRL_LOG_DIR="${LOG_DIR}" \
    -e LUMENRL_SHARED_CACHE_DIR="${CACHE_DIR}" \
    -e LUMENRL_OVERRIDES="${OVERRIDE_STRING}" \
    "${CONTAINER_NAME}" \
    python3 /root/lumenrl/examples/Kimi_K3_SDDD_MI350_ATOM/ray_multinode_launcher.py
