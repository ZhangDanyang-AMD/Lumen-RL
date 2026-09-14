#!/usr/bin/env bash
# One copy of this script runs on each node of a top-level five-task srun.
set -euo pipefail

: "${SLURM_PROCID:?Run with srun --nodes=5 --ntasks=5}"
: "${SLURM_NTASKS:?Run with srun --nodes=5 --ntasks=5}"
if (( SLURM_NTASKS != 5 )); then
    echo "Expected five tasks, got ${SLURM_NTASKS}." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RANK="${SLURM_PROCID}"
HOST="$(hostname)"
HOST_IP="$(hostname -I | awk '{print $1}')"
RUN_ID="${LUMENRL_RUN_ID:-${SLURM_JOB_ID:-manual}}"

DATA_ROOT="${DATA_ROOT:-/mnt/m2m_nobackup/danyzhan}"
SHARED_ROOT="${SHARED_ROOT:-/shared_nfs/danyzhan/lumenrl}"
MODEL_PATH="${MODEL_PATH:-${DATA_ROOT}/models/Kimi-K3}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/datasets/ATOM_regen_seeklight_kimi_mtp/data/train.jsonl}"
CKPT_DIR="${CKPT_DIR:-${SHARED_ROOT}/checkpoints/kimi_k3_dspark_atom}"
CACHE_DIR="${CACHE_DIR:-${SHARED_ROOT}/cache/kimi_k3_dspark_atom}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-${DATA_ROOT}/cache/lumenrl}"
LOG_DIR="${LOG_DIR:-${SHARED_ROOT}/logs/kimi_k3_dspark_atom_${RUN_ID}}"
COORD_DIR="${SHARED_ROOT}/coord/kimi_k3_dspark_atom_${RUN_ID}"
DOCKER_IMAGE="${DOCKER_IMAGE:-kimi_k3_dspark_atom:latest}"
CONTAINER_NAME="kimi-k3-sddd-${RUN_ID}"
RAY_PORT="${RAY_PORT:-26379}"
# Containers run with --network host, so Ray's default auxiliary ports are shared
# with everything else on the node. 10001 (the Ray client server) was already
# taken by another tenant on one of these hosts, and `ray start --head` then
# fails with "Failed to bind to address 127.0.0.1:10001" and exits, which shows
# up only as "Ray cluster has 0/5 active nodes". Nothing here uses a ray:// client
# or the dashboard, so move one and switch the other off.
RAY_CLIENT_PORT="${RAY_CLIENT_PORT:-26380}"
SMOKE_TEST="${SMOKE_TEST:-0}"

# Hidden states move over Mooncake TCP, not RDMA. On this cluster's Ionic HCAs a
# process can register one RDMA client per HCA; the second client on a device
# fails with EINVAL (measured with selfcheck/time_mooncake_pool.py). Seven HCAs
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
# One teacher segment must hold every batch it has in flight. At batch 128 and
# the nine-category set's mean length that is about 24 GiB per batch, 88 GiB in
# the 8192-token worst case, and stream_prefetch_batches=8 leaves two batches per
# replica. 128 GB is host RAM on a 2.8 TB node and is only touched lazily.
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-128GB}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-8GB}"
# Draft ranks only read, and all eight of them register a segment on one node.
export LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE="${LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE:-4GB}"
# The segmented producer pool exists to work around the per-MR size limit on
# RDMA. Over TCP a single store takes an arbitrarily large segment, so the pool
# collapses to one store and the whole failure mode above disappears.
export LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE="${LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE:-1}"
export LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE="${LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE:-128GB}"
export LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS="${LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS:-300}"

# 用 CONFIG_NAME 换配置文件（默认还是 train.yaml，不改变原有行为）。
# gen-6 的续训用 train_stage2.yaml：CONFIG_NAME=train_stage2.yaml
CONFIG="${REPO_ROOT}/examples/Kimi_K3_SDDD_MI350_ATOM/configs/${CONFIG_NAME:-train.yaml}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    CONFIG="${REPO_ROOT}/examples/Kimi_K3_SDDD_MI350_ATOM/configs/${SMOKE_CONFIG_NAME:-smoke_test.yaml}"
fi

for required in "${MODEL_PATH}" "${DATASET_PATH}" "${REPO_ROOT}"; do
    [[ -e "${required}" ]] || {
        echo "${HOST}: required path missing: ${required}" >&2
        exit 2
    }
done
docker image inspect "${DOCKER_IMAGE}" >/dev/null

mkdir -p "${COORD_DIR}" "${LOG_DIR}" "${CKPT_DIR}" "${CACHE_DIR}" "${TOKEN_CACHE_DIR}"
printf '%s %s\n' "${HOST}" "${HOST_IP}" >"${COORD_DIR}/node-${RANK}"

cleanup() {
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
on_exit() {
    rc=$?
    if (( RANK == 0 )) && [[ ! -f "${COORD_DIR}/exit-code" ]]; then
        echo "${rc}" >"${COORD_DIR}/exit-code"
    fi
    cleanup
}
trap on_exit EXIT
cleanup

# 600, not 120. Under srun the five tasks start together, but when the ranks are
# started through five `spur exec` calls (five separate single-node jobs rather
# than one -N5 allocation) each call can take about 100 s to return, so the node
# files can be minutes apart. Waiting longer costs nothing; timing out here costs
# a full restart.
for _ in {1..600}; do
    shopt -s nullglob
    node_files=("${COORD_DIR}"/node-*)
    (( ${#node_files[@]} == 5 )) && break
    sleep 1
done
if (( ${#node_files[@]} != 5 )); then
    echo "${HOST}: node discovery timed out (${#node_files[@]}/5)." >&2
    exit 1
fi
read -r HEAD_HOST HEAD_IP <"${COORD_DIR}/node-0"

COMMON_DOCKER_ARGS=(
    --name "${CONTAINER_NAME}"
    --network host
    --ipc host
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
    -v "${REPO_ROOT}:/root/lumenrl"
    -w /root/lumenrl
    -e PYTHONUNBUFFERED=1
    -e NCCL_TIMEOUT=7200
    -e RAY_DEDUP_LOGS=0
    # Without a flight recorder a hung collective reports only "Stack trace of
    # the failed collective not found", which is what the first teacher deadlock
    # here left behind. It costs a small ring buffer per rank and only produces
    # output when a collective actually times out.
    -e TORCH_NCCL_TRACE_BUFFER_SIZE=2048
    -e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
    -e "NCCL_IB_HCA=${NCCL_IB_HCA}"
    -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
    -e "NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE}"
    -e "MOONCAKE_DEVICE_NAME=${MOONCAKE_DEVICE_NAME}"
    -e "MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE}"
    -e "MOONCAKE_LOCAL_BUFFER_SIZE=${MOONCAKE_LOCAL_BUFFER_SIZE}"
    -e "LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE=${LUMENRL_DRAFT_MOONCAKE_SEGMENT_SIZE}"
    -e "LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE=${LUMENRL_TEACHER_MOONCAKE_SEGMENT_POOL_SIZE}"
    -e "LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE=${LUMENRL_TEACHER_MOONCAKE_SEGMENT_SIZE}"
    -e "LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS=${LUMENRL_TEACHER_MOONCAKE_POOL_WAIT_SECONDS}"
)
for rdma_device in /dev/infiniband/*; do
    if [[ -c "${rdma_device}" ]]; then
        COMMON_DOCKER_ARGS+=(--device "${rdma_device}:${rdma_device}")
    fi
done

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
        ray_files=("${COORD_DIR}"/ray-*)
        (( ${#ray_files[@]} == 5 )) && break
        sleep 1
    done
    if (( ${#ray_files[@]} != 5 )); then
        echo "Ray process startup timed out (${#ray_files[@]}/5)." >&2
        echo 1 >"${COORD_DIR}/exit-code"
        exit 1
    fi

    active_nodes=0
    for _ in {1..120}; do
        active_nodes="$(
            docker exec "${CONTAINER_NAME}" python3 -c \
                'import ray; ray.init(address="auto"); print(sum(1 for n in ray.nodes() if n["Alive"]))' \
                2>/dev/null | awk 'NF {value=$NF} END {print value+0}' || true
        )"
        [[ "${active_nodes}" == "5" ]] && break
        sleep 2
    done
    if [[ "${active_nodes}" != "5" ]]; then
        echo "Ray cluster has ${active_nodes}/5 active nodes." >&2
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
        -e LUMENRL_NUM_NODES=5 \
        -e LUMENRL_GPUS_PER_NODE=8 \
        -e LUMENRL_REPO_ROOT=/root/lumenrl \
        -e "LUMENRL_CONFIG=/root/lumenrl/${CONFIG#${REPO_ROOT}/}" \
        -e "LUMENRL_LOG_DIR=${LOG_DIR}" \
        -e "LUMENRL_SHARED_CACHE_DIR=${CACHE_DIR}" \
        -e "LUMENRL_OVERRIDES=${OVERRIDE_STRING}" \
        "${CONTAINER_NAME}" \
        python3 /root/lumenrl/examples/Kimi_K3_SDDD_MI350_ATOM/ray_multinode_launcher.py
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
