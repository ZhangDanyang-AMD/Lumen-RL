#!/usr/bin/env bash
#SBATCH --job-name=k3-sddd-ray
#SBATCH --nodes=6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=k3-sddd-ray-%j.out
#SBATCH --error=k3-sddd-ray-%j.err

# Ray places and supervises one 8-GPU torchrun launcher per node. The training
# ranks themselves communicate through RCCL, because FSDP2 collectives cannot be
# replaced by Ray object transfers.
#
# Submit for the requested machines:
#   sbatch --nodelist=crsuse2-m2m-v2-[034-039] \
#     examples/Kimi_K3_SDDD_MI350_ATOM/run_multinode_slurm.sh
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch or run it inside an allocation.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NUM_NODES="${#NODES[@]}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
HEAD_NODE="${NODES[0]}"
HEAD_IP="$(srun --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" hostname -I | awk '{print $1}')"
RAY_PORT="${RAY_PORT:-6379}"
CONTAINER_NAME="${CONTAINER_NAME:-kimi-k3-sddd-ray-${SLURM_JOB_ID}}"

DATA_ROOT="${DATA_ROOT:-/data}"
MODEL_PATH="${MODEL_PATH:-${DATA_ROOT}/models/Kimi-K3}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/datasets/kimi-mtp-dataset-full/train.jsonl}"
CKPT_DIR="${CKPT_DIR:-${DATA_ROOT}/checkpoints/kimi_k3_dspark_atom_ray_${NUM_NODES}n}"
CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/cache/kimi_k3_teacher_ray_${NUM_NODES}n}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-${DATA_ROOT}/cache/lumenrl_tokenized}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/kimi_k3_dspark_atom_ray_${SLURM_JOB_ID}}"
DOCKER_IMAGE="${DOCKER_IMAGE:-kimi_k3_dspark_atom:latest}"
CONFIG="${CONFIG:-${REPO_ROOT}/examples/Kimi_K3_SDDD_MI350_ATOM/configs/train.yaml}"
SMOKE_TEST="${SMOKE_TEST:-0}"

WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${WORLD_SIZE}}"
if (( GLOBAL_BATCH_SIZE % WORLD_SIZE != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by world size ${WORLD_SIZE}." >&2
    exit 2
fi
LEARNING_RATE="${LEARNING_RATE:-$(awk -v bs="${GLOBAL_BATCH_SIZE}" 'BEGIN {printf "%.12g", 7.5e-5 * bs / 64.0}')}"
CACHE_BATCHES="${CACHE_BATCHES:-50}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    CONFIG="${REPO_ROOT}/examples/Kimi_K3_SDDD_MI350_ATOM/configs/smoke_test.yaml"
    CACHE_BATCHES="${CACHE_BATCHES_SMOKE:-5}"
fi

for required in "${MODEL_PATH}" "${DATASET_PATH}" "${REPO_ROOT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required path does not exist on the submit node: ${required}" >&2
        exit 2
    fi
done
mkdir -p "${CKPT_DIR}" "${CACHE_DIR}" "${TOKEN_CACHE_DIR}" "${LOG_DIR}"

OVERRIDES=(
    "cluster.num_nodes=${NUM_NODES}"
    "cluster.gpus_per_node=${GPUS_PER_NODE}"
    "controller.ray.enabled=false"
    "policy.model_name=${MODEL_PATH}"
    "policy.train_global_batch_size=${GLOBAL_BATCH_SIZE}"
    "policy.learning_rate=${LEARNING_RATE}"
    "algorithm.teacher.model_name=${MODEL_PATH}"
    "algorithm.spec_distill.cache_dir=${CACHE_DIR}"
    "algorithm.spec_distill.cache_batches=${CACHE_BATCHES}"
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
    --device /dev/kfd
    --device /dev/dri
    --group-add video
    --group-add render
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    -v /dev/shm:/dev/shm
    -v "${DATA_ROOT}:${DATA_ROOT}"
    -v "${REPO_ROOT}:/root/lumenrl"
    -w /root/lumenrl
    -e PYTHONUNBUFFERED=1
    -e NCCL_TIMEOUT=7200
    -e RAY_DEDUP_LOGS=0
)
for name in HF_TOKEN NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX \
    NCCL_IB_DISABLE NCCL_NET_GDR_LEVEL NCCL_DMABUF_ENABLE; do
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

echo "Ray cluster ready: ${NUM_NODES} nodes, ${WORLD_SIZE} GPUs"
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
