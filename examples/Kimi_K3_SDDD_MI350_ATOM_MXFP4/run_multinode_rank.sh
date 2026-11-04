#!/usr/bin/env bash
# One copy of this script runs on each node in the Ray topology.
set -euo pipefail

: "${SLURM_PROCID:?Set one rank per node}"
: "${SLURM_NTASKS:?Set the topology node count}"
if (( SLURM_NTASKS < 2 )); then
    echo "Expected at least two tasks, got ${SLURM_NTASKS}." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RANK="${SLURM_PROCID}"
DRAFT_NODE_RANK="${LUMENRL_DRAFT_NODE_RANK:-$((SLURM_NTASKS - 1))}"
if (( DRAFT_NODE_RANK < 0 || DRAFT_NODE_RANK >= SLURM_NTASKS )); then
    echo "Invalid LUMENRL_DRAFT_NODE_RANK=${DRAFT_NODE_RANK} " \
        "for ${SLURM_NTASKS} nodes." >&2
    exit 2
fi
HOST="$(hostname)"
HOST_IP="$(hostname -I | awk '{print $1}')"
RUN_ID="${LUMENRL_RUN_ID:-${SLURM_JOB_ID:-manual}}"

DATA_ROOT="${DATA_ROOT:-/path/to/local_nvme}"
SHARED_ROOT="${SHARED_ROOT:-/path/to/shared_nfs/lumenrl}"
LOG_ROOT="${LOG_ROOT:-${SHARED_ROOT}/logs}"
MODEL_PATH="${MODEL_PATH:-${DATA_ROOT}/models/Kimi-K3}"
DATASET_PATH="${DATASET_PATH:-${SHARED_ROOT}/datasets/Kimi-K3-Draft-mxfp4-training-data/data/train.jsonl}"
CKPT_DIR="${CKPT_DIR:-${SHARED_ROOT}/checkpoints/kimi_k3_dspark_mxfp4_atom}"
CACHE_DIR="${CACHE_DIR:-${SHARED_ROOT}/cache/kimi_k3_dspark_mxfp4_atom}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-${DATA_ROOT}/cache/lumenrl}"
EVAL_FIXED_PATH="${EVAL_FIXED_PATH:-${SHARED_ROOT}/eval/kimi_k3_dspark_mxfp4_fixed_128.pt}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/kimi_k3_dspark_mxfp4_atom_${RUN_ID}}"
COORD_DIR="${SHARED_ROOT}/coord/kimi_k3_dspark_mxfp4_atom_${RUN_ID}"
DOCKER_IMAGE="${DOCKER_IMAGE:-kimi_k3_dspark_mxfp4_atom:latest}"
CONTAINER_NAME="kimi-k3-dspark-mxfp4-${RUN_ID}"
RAY_PORT="${RAY_PORT:-26379}"
# Containers run with --network host, so Ray's default auxiliary ports are shared
# with everything else on the node. 10001 (the Ray client server) was already
# taken by another tenant on one of these hosts, and `ray start --head` then
# fails with "Failed to bind to address 127.0.0.1:10001" and exits, which shows
# up only as "Ray cluster has 0/N active nodes". Nothing here uses a ray:// client
# or the dashboard, so move one and switch the other off.
RAY_CLIENT_PORT="${RAY_CLIENT_PORT:-26380}"

# Hidden states move over Mooncake TCP, not RDMA. On this cluster's Ionic HCAs a
# process can register one RDMA client per HCA; the second client on a device
# fails with EINVAL. Seven HCAs
# therefore cap a teacher's producer pool at 7 x 2 GiB = 14 GiB, while one batch
# of 128 sequences publishes about 24 GiB, so the producer stalls until
# extract_hidden times out at 600 s -- which is exactly how the 2026-08-24 runs
# died, zero steps completed. TCP has no memory-region limit and measured
# 2.0 GiB/s put plus 4.44 GiB/s cross-node get with eight fetchers, i.e. about
# 5 s per batch against a ~44 s draft step. Set MOONCAKE_PROTOCOL=rdma only if
# the HCA limitation is lifted, and then keep the pool at or below the HCA count.
export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
export MOONCAKE_DEVICE_NAME="${MOONCAKE_DEVICE_NAME:-}"
# NCCL only carries draft gradients inside the draft node, so it keeps using the
# RDMA fabric. ionic_7 does not exist on these hosts; the HCAs are ionic_0..6.
export NCCL_IB_HCA="${NCCL_IB_HCA:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-spur0}"
# One teacher segment must hold every 32768-window batch it has in flight.
# Memory is touched lazily; the formal value is 256 GB.
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-256GB}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-8GB}"
# Draft ranks only read, and all eight of them register a segment on one node.
export LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE="${LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE:-4GB}"
# The segmented producer pool exists to work around the per-MR size limit on
# RDMA. Over TCP a single store takes an arbitrarily large segment, so the pool
# collapses to one store and the whole failure mode above disappears.
export LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE="${LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE:-1}"
export LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE="${LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE:-256GB}"
export LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS="${LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS:-300}"
export LUMEN_MXFP4_AUTOTUNE="${LUMEN_MXFP4_AUTOTUNE:-0}"
export LUMEN_MXFP4_AUTOTUNE_CACHE="${LUMEN_MXFP4_AUTOTUNE_CACHE:-${SHARED_ROOT}/cache/dspark_mxfp4_autotune_gfx950.json}"
export LUMEN_MXFP4_GEMM_SHAPE_LOG="${LUMEN_MXFP4_GEMM_SHAPE_LOG:-}"
export LUMEN_MXFP4_WGRAD_TILE_PAD="${LUMEN_MXFP4_WGRAD_TILE_PAD:-1}"
export LUMEN_MXFP4_WGRAD_POWER2_BUCKET="${LUMEN_MXFP4_WGRAD_POWER2_BUCKET:-1}"
export LUMEN_MXFP4_FUSED_TARGET_ROWS="${LUMEN_MXFP4_FUSED_TARGET_ROWS:-1}"
export LUMENRL_BALANCE_DISAGG_ROWS="${LUMENRL_BALANCE_DISAGG_ROWS:-1}"

for required in "${MODEL_PATH}" "${DATASET_PATH}" "${EVAL_FIXED_PATH}" "${REPO_ROOT}"; do
    [[ -e "${required}" ]] || {
        echo "${HOST}: required path missing: ${required}" >&2
        exit 2
    }
done
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${DOCKER_IMAGE}")"

mkdir -p "${COORD_DIR}" "${LOG_DIR}" "${CKPT_DIR}" "${CACHE_DIR}" "${TOKEN_CACHE_DIR}"
# `spur exec` may replay a detached remote command while establishing the step.
# An atomic per-rank directory makes the launch idempotent so only one process
# can create the Ray container for this RUN_ID.
RANK_LAUNCH_LOCK="${COORD_DIR}/rank-${RANK}.launch-lock"
if ! mkdir "${RANK_LAUNCH_LOCK}" 2>/dev/null; then
    echo "${HOST}: duplicate rank launch ignored for run ${RUN_ID}"
    exit 0
fi
printf '%s %s\n' "${HOST}" "${HOST_IP}" >"${COORD_DIR}/node-${RANK}"
printf '%s\n' "${IMAGE_ID}" >"${COORD_DIR}/image-${RANK}"

cleanup() {
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
on_exit() {
    rc=$?
    if (( RANK == 0 )) && [[ ! -f "${COORD_DIR}/exit-code" ]]; then
        echo "${rc}" >"${COORD_DIR}/exit-code"
    fi
    cleanup
    rmdir "${RANK_LAUNCH_LOCK}" >/dev/null 2>&1 || true
}
trap on_exit EXIT
cleanup

exit_if_requested() {
    local requested_rc
    [[ -f "${COORD_DIR}/exit-code" ]] || return 0
    requested_rc="$(<"${COORD_DIR}/exit-code")"
    [[ "${requested_rc}" =~ ^[0-9]+$ ]] || requested_rc=2
    echo "${HOST}: startup abort requested (exit=${requested_rc})" >&2
    exit "${requested_rc}"
}

exit_if_requested

# 600, not 120. Under srun the tasks start together, but when the ranks are
# started through separate `spur exec` calls each call can take about 100 s to
# return, so the node
# files can be minutes apart. Waiting longer costs nothing; timing out here costs
# a full restart.
for _ in {1..600}; do
    exit_if_requested
    shopt -s nullglob
    node_files=("${COORD_DIR}"/node-*)
    image_files=("${COORD_DIR}"/image-*)
    (( ${#node_files[@]} == SLURM_NTASKS \
        && ${#image_files[@]} == SLURM_NTASKS )) && break
    sleep 1
done
if (( ${#node_files[@]} != SLURM_NTASKS \
    || ${#image_files[@]} != SLURM_NTASKS )); then
    echo "${HOST}: node/image discovery timed out " \
        "(${#node_files[@]}/${#image_files[@]}/${SLURM_NTASKS})." >&2
    exit 1
fi
exit_if_requested
read -r EXPECTED_IMAGE_ID <"${COORD_DIR}/image-0"
for image_file in "${image_files[@]}"; do
    read -r observed_image_id <"${image_file}"
    if [[ "${observed_image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
        echo "${HOST}: image mismatch: ${image_file} has " \
            "${observed_image_id}, expected ${EXPECTED_IMAGE_ID}" >&2
        exit 2
    fi
done
echo "${HOST}: image=${IMAGE_ID}"
read -r HEAD_HOST HEAD_IP <"${COORD_DIR}/node-0"
read -r DRAFT_HOST DRAFT_IP <"${COORD_DIR}/node-${DRAFT_NODE_RANK}"
if (( RANK == 0 )); then
    echo "Draft target: rank=${DRAFT_NODE_RANK} host=${DRAFT_HOST} ip=${DRAFT_IP}"
fi

COMMON_DOCKER_ARGS=(
    --name "${CONTAINER_NAME}"
    --network host
    --ipc host
    --tmpfs /tmp:exec
    --ulimit memlock=-1:-1
    --device /dev/kfd
    --device /dev/dri
    -v /dev/infiniband:/dev/infiniband
    # The ATOM image ships an older Ionic provider than the host kernel driver.
    # Inject the matching host userspace pieces so libibverbs accepts ionic_0..7.
    -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro
    -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro
    --group-add video
    --group-add render
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    -v /dev/shm:/dev/shm
    -v "${DATA_ROOT}:${DATA_ROOT}"
    -v "${SHARED_ROOT}:${SHARED_ROOT}"
    -w /root/lumenrl
    -e PYTHONUNBUFFERED=1
    -e NCCL_TIMEOUT=7200
    -e RAY_DEDUP_LOGS=0
    -e "LUMENRL_DRAFT_NODE_IP=${DRAFT_IP}"
    # Without a flight recorder a hung collective reports only "Stack trace of
    # the failed collective not found", which is what the first teacher deadlock
    # here left behind. It costs a small ring buffer per rank and only produces
    # output when a collective actually times out.
    -e TORCH_FR_BUFFER_SIZE=2048
    -e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
    -e "NCCL_IB_HCA=${NCCL_IB_HCA}"
    -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
    -e "NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE}"
    -e "MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL}"
    -e "MOONCAKE_DEVICE_NAME=${MOONCAKE_DEVICE_NAME}"
    -e "MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE}"
    -e "MOONCAKE_LOCAL_BUFFER_SIZE=${MOONCAKE_LOCAL_BUFFER_SIZE}"
    -e "LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE=${LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE}"
    -e "LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE=${LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE}"
    -e "LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE=${LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE}"
    -e "LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS=${LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS}"
    -e "LUMEN_MXFP4_AUTOTUNE=${LUMEN_MXFP4_AUTOTUNE}"
    -e "LUMEN_MXFP4_AUTOTUNE_CACHE=${LUMEN_MXFP4_AUTOTUNE_CACHE}"
    -e "LUMEN_MXFP4_GEMM_SHAPE_LOG=${LUMEN_MXFP4_GEMM_SHAPE_LOG}"
    -e "LUMEN_MXFP4_WGRAD_TILE_PAD=${LUMEN_MXFP4_WGRAD_TILE_PAD}"
    -e "LUMEN_MXFP4_WGRAD_POWER2_BUCKET=${LUMEN_MXFP4_WGRAD_POWER2_BUCKET}"
    -e "LUMEN_MXFP4_FUSED_TARGET_ROWS=${LUMEN_MXFP4_FUSED_TARGET_ROWS}"
    -e "LUMENRL_BALANCE_DISAGG_ROWS=${LUMENRL_BALANCE_DISAGG_ROWS}"
    -e "EVAL_FIXED_PATH=${EVAL_FIXED_PATH}"
    -e "DATASET_PATH=${DATASET_PATH}"
)
if [[ -n "${TRITON_CACHE_ROOT:-}" ]]; then
    TRITON_CACHE_DIR="${TRITON_CACHE_ROOT}/${HOST}"
    mkdir -p "${TRITON_CACHE_DIR}"
    COMMON_DOCKER_ARGS+=(
        -e "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
    )
fi
if [[ "${LOG_ROOT}" != "${SHARED_ROOT}" && "${LOG_ROOT}" != "${SHARED_ROOT}/"* ]]; then
    COMMON_DOCKER_ARGS+=(-v "${LOG_ROOT}:${LOG_ROOT}")
fi
for rdma_device in /dev/infiniband/*; do
    if [[ -c "${rdma_device}" ]]; then
        COMMON_DOCKER_ARGS+=(--device "${rdma_device}:${rdma_device}")
    fi
done

exit_if_requested
if (( RANK == 0 )); then
    docker run -d "${COMMON_DOCKER_ARGS[@]}" "${DOCKER_IMAGE}" \
        ray start --head --node-ip-address="${HOST_IP}" --port="${RAY_PORT}" \
        --ray-client-server-port="${RAY_CLIENT_PORT}" --include-dashboard=false \
        --num-gpus=8 --block
else
    docker run -d "${COMMON_DOCKER_ARGS[@]}" "${DOCKER_IMAGE}" \
        ray start --address="${HEAD_IP}:${RAY_PORT}" \
        --node-ip-address="${HOST_IP}" --num-gpus=8 --block
fi
touch "${COORD_DIR}/ray-${RANK}"

if (( RANK == 0 )); then
    for _ in {1..120}; do
        exit_if_requested
        ray_files=("${COORD_DIR}"/ray-*)
        (( ${#ray_files[@]} == SLURM_NTASKS )) && break
        sleep 1
    done
    if (( ${#ray_files[@]} != SLURM_NTASKS )); then
        echo "Ray process startup timed out (${#ray_files[@]}/${SLURM_NTASKS})." >&2
        echo 1 >"${COORD_DIR}/exit-code"
        exit 1
    fi

    active_nodes=0
    for _ in {1..120}; do
        exit_if_requested
        active_nodes="$(
            docker exec "${CONTAINER_NAME}" python3 -c \
                'import ray; ray.init(address="auto"); print(sum(1 for n in ray.nodes() if n["Alive"]))' \
                2>/dev/null | awk 'NF {value=$NF} END {print value+0}' || true
        )"
        [[ "${active_nodes}" == "${SLURM_NTASKS}" ]] && break
        sleep 2
    done
    if [[ "${active_nodes}" != "${SLURM_NTASKS}" ]]; then
        echo "Ray cluster has ${active_nodes}/${SLURM_NTASKS} active nodes." >&2
        # The probe above hides stderr, so on its own this message says nothing
        # about why. A head container that died during `ray start` is the usual
        # cause and its log names the reason outright.
        echo "--- head container state and log ---" >&2
        docker inspect -f 'running={{.State.Running}} exit={{.State.ExitCode}}' \
            "${CONTAINER_NAME}" >&2 2>/dev/null || true
        docker logs --tail 40 "${CONTAINER_NAME}" >&2 2>&1 || true
        echo 1 >"${COORD_DIR}/exit-code"
        exit 1
    fi

    ready_tmp="${COORD_DIR}/.cluster-ready-${RANK}-$$"
    printf '%s\n' "${SLURM_NTASKS}" >"${ready_tmp}"
    mv "${ready_tmp}" "${COORD_DIR}/cluster-ready"

    OVERRIDES=(
        "cluster.num_nodes=1"
        "cluster.gpus_per_node=8"
        "controller.ray.enabled=false"
        "policy.model_name=${MODEL_PATH}"
        "algorithm.teacher.model_name=${MODEL_PATH}"
        "reward.dataset=${DATASET_PATH}"
        "dataset.cache_dir=${TOKEN_CACHE_DIR}"
        "checkpointing.checkpoint_dir=${CKPT_DIR}"
    )
    if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
        read -r -a USER_OVERRIDES <<<"${EXTRA_OVERRIDES}"
        OVERRIDES+=("${USER_OVERRIDES[@]}")
    fi
    printf -v OVERRIDE_STRING '%q ' "${OVERRIDES[@]}"

    set +e
    docker exec \
        -e "LUMENRL_NUM_NODES=${SLURM_NTASKS}" \
        -e LUMENRL_GPUS_PER_NODE=8 \
        -e LUMENRL_REPO_ROOT=/root/lumenrl \
        -e "LUMENRL_CONFIG=/root/lumenrl/examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/configs/${CONFIG_NAME:-train_from_scratch.yaml}" \
        -e "LUMENRL_LOG_DIR=${LOG_DIR}" \
        -e "LUMENRL_SHARED_CACHE_DIR=${CACHE_DIR}" \
        -e "LUMENRL_OVERRIDES=${OVERRIDE_STRING}" \
        "${CONTAINER_NAME}" \
        python3 /root/lumenrl/examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/ray_multinode_launcher.py
    rc=$?
    set -e
    echo "${rc}" >"${COORD_DIR}/exit-code"
else
    while [[ ! -f "${COORD_DIR}/exit-code" ]]; do
        sleep 2
    done
    rc="$(<"${COORD_DIR}/exit-code")"
fi

exit "${rc}"
