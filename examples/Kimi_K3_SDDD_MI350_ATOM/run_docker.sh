#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K3 DSpark SDDD (ATOM teacher) — Docker launch
#
# Storage layout this expects, because K3 is large enough that where things live
# is a correctness concern rather than a preference:
#   - Teacher weights are ~1.5 TB, so they belong on a real filesystem, not on
#     tmpfs. DATA_ROOT points at one; MODEL_PATH and CKPT_DIR default under it.
#   - Hidden-state cache stays on /dev/shm, and cache_batches is sized so one
#     round fits (~11.3 GB per batch at B=64).
#   - Checkpoints go to DATA_ROOT, not /dev/shm, which is wiped when the job
#     ends on most schedulers.
#   - The dataset is read by 96 preprocessing workers, so it is staged onto
#     /dev/shm and re-staged if tmpfs has been cleared.
#
# Everything is overridable; nothing is specific to one cluster or user:
#   DATA_ROOT   MODEL_PATH  DATASET_SRC  DATASET_DST  CKPT_DIR  CACHE_DIR
#   DOCKER_IMAGE  LUMENRL_DIR  CONTAINER_NAME  EXTRA_MOUNTS  EXTRA_OVERRIDES
#
# Usage:
#   bash examples/Kimi_K3_SDDD_MI350_ATOM/run_docker.sh --smoke-test
#   DETACH=1 bash examples/Kimi_K3_SDDD_MI350_ATOM/run_docker.sh
#   DETACH=1 EXTRA_OVERRIDES="checkpointing.resume=true" \
#       bash examples/Kimi_K3_SDDD_MI350_ATOM/run_docker.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where the large, persistent artifacts live. Anything on tmpfs disappears with
# the allocation, so this should be a real filesystem with room for the weights.
DATA_ROOT="${DATA_ROOT:-/data}"

# Built from docker/Dockerfile: rocm/atom-dev plus LumenRL, Lumen and the
# torchspec shim. Contains no vLLM by design.
DOCKER_IMAGE="${DOCKER_IMAGE:-kimi_k3_dspark_atom:latest}"
LUMENRL_DIR="${LUMENRL_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-${DATA_ROOT}/models/Kimi-K3}"
DATASET_SRC="${DATASET_SRC:-${DATA_ROOT}/datasets/kimi-mtp-dataset-full}"
DATASET_DST="${DATASET_DST:-/dev/shm/kimi-mtp-dataset-full}"

# Extra bind mounts, e.g. a shared home or a second data volume:
#   EXTRA_MOUNTS="-v /nfs/home:/nfs/home -v /scratch:/scratch"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"

if [ "${SMOKE_TEST}" = true ]; then
    CONTAINER_NAME="${CONTAINER_NAME:-kimi_k3_dspark_atom_smoke}"
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/kimi_k3_dspark_atom_smoke}"
    CACHE_DIR="${CACHE_DIR:-/dev/shm/teacher_cache_atom_smoke}"
    CACHE_BATCHES="${CACHE_BATCHES:-5}"
    RUN_CMD="bash examples/Kimi_K3_SDDD_MI350_ATOM/run_kimi_k3.sh --smoke-test"
else
    CONTAINER_NAME="${CONTAINER_NAME:-kimi_k3_dspark_atom}"
    CKPT_DIR="${CKPT_DIR:-${DATA_ROOT}/checkpoints/kimi_k3_dspark_atom}"
    CACHE_DIR="${CACHE_DIR:-/dev/shm/teacher_cache_atom}"
    # At bs=64 a cached batch is 11.3 GB, so 50 rounds needs ~565 GB of tmpfs.
    # Override only after redoing that arithmetic.
    CACHE_BATCHES="${CACHE_BATCHES:-50}"
    RUN_CMD="bash examples/Kimi_K3_SDDD_MI350_ATOM/run_kimi_k3.sh"
fi

EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
EXTRA_OVERRIDES="algorithm.spec_distill.cache_dir=${CACHE_DIR} algorithm.spec_distill.cache_batches=${CACHE_BATCHES} ${EXTRA_OVERRIDES}"

# Lumen's kernels are built against its own aiter fork, while this image ships
# the AMD aiter that ATOM's K3 kernels need. Keep them off until the draft path
# is proven on this backend, then flip to 1 to measure the speedup.
if [ "${LUMEN_KERNELS:-0}" != "1" ]; then
    EXTRA_OVERRIDES="${EXTRA_OVERRIDES} quantization.training.lumen_norm=false quantization.training.lumen_linear=false quantization.training.hf_attn_patch=false"
fi

# ---- Stage the dataset onto tmpfs -------------------------------------------
# Both the check and the copy run inside a container that bind-mounts the same
# /dev/shm the training container gets. Running them directly on the host is
# wrong whenever this script is itself invoked from inside a container with a
# private /dev/shm, as cluster `exec` wrappers commonly give you: the copy would
# report success against the wrong tmpfs, and the trainer would then fail on a
# missing dataset only after the teacher had finished loading.
_shm_stage() {
    docker run --rm \
        -v /dev/shm:/dev/shm \
        -v "${DATA_ROOT}:${DATA_ROOT}" \
        ${EXTRA_MOUNTS} \
        --entrypoint bash "${DOCKER_IMAGE}" -c "$1"
}

if ! _shm_stage "test -f '${DATASET_DST}/train.jsonl'"; then
    if [ ! -f "${DATASET_SRC}/train.jsonl" ]; then
        echo ">>> FATAL: dataset not found at ${DATASET_SRC}/train.jsonl" >&2
        exit 1
    fi
    echo ">>> Staging dataset to tmpfs: ${DATASET_SRC} -> ${DATASET_DST}"
    _shm_stage "mkdir -p '${DATASET_DST}' && cp '${DATASET_SRC}/train.jsonl' '${DATASET_DST}/train.jsonl' && du -sh '${DATASET_DST}' | cut -f1 | xargs echo '>>> Dataset staged'"
else
    echo ">>> Dataset already staged at ${DATASET_DST}"
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo ">>> FATAL: teacher weights not found at ${MODEL_PATH}" >&2
    exit 1
fi

# A foreground `docker run` forwards SIGTERM into the container, so whatever
# reaps the launching shell's process tree takes the training run down with it,
# silently and without an exit code. Detached containers are owned by the docker
# daemon instead, so they survive their launching session.
#
# The crashes this recipe hits are transient and recoverable — a long batch
# pushes one GPU over the edge during an RCCL all-reduce, a rank aborts,
# torchrun tears the job down — and resuming costs a single round. Noticing that
# by hand has cost as much as 8 idle hours, so the daemon's restart policy
# carries the retry. A supervisor process cannot, because anything started from
# an interactive cluster session dies with that session, detached or not.
# Exit 0 (training reached num_training_steps) does not trigger a restart, and a
# manual `docker stop` suspends the policy until the container starts again.
DETACH="${DETACH:-0}"
if [ "${DETACH}" = "1" ]; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    # Sized to cover genuinely rare crashes (~1 per 830 steps). A low cap also
    # stops a systematic failure from spinning hundreds of times unnoticed.
    RUN_MODE_ARGS="-d --restart=on-failure:${MAX_RESTARTS:-40}"
else
    RUN_MODE_ARGS="--rm"
fi

# PYTORCH_CUDA_ALLOC_CONF is deliberately not forwarded from the caller.
#
# expandable_segments was tried against the RCCL all-reduce OOM and broke the
# round transition instead: the driver counts the allocator's whole virtual
# reservation as used while torch reports only the mapped part, so free memory
# between rounds fell from 265 GiB to 175 GiB (reserved simultaneously dropped
# 16.5 -> 4.9 GiB) and the teacher engine, which needs ~253 GiB, could no
# longer restart for the next Phase A. Every round transition then failed.
#
# Backing it out by deleting the export did not work, because the line here
# read "${PYTORCH_CUDA_ALLOC_CONF:-}" and inherited the setting from whatever
# shell launched the script — including the one that had exported it. Six
# hours went into "proving" the setting was innocent on numbers taken from
# containers that still had it. Nothing is passed now, so the container's
# allocator cannot depend on the launching environment. The trainer also
# refuses to start batch-alternating training if it sees the setting.

mkdir -p "${CKPT_DIR%/*}" 2>/dev/null || true

cat <<EOF
═══════════════════════════════════════════════════════════════
  Kimi K3 DSpark (ATOM) $([ "${SMOKE_TEST}" = true ] && echo "— SMOKE TEST")
  Image:          ${DOCKER_IMAGE}
  Container:      ${CONTAINER_NAME}
  LumenRL:        ${LUMENRL_DIR}
  Data root:      ${DATA_ROOT}
  Teacher model:  ${MODEL_PATH}
  Dataset:        ${DATASET_DST}
  Checkpoints:    ${CKPT_DIR}
  Teacher cache:  ${CACHE_DIR}  (cache_batches=${CACHE_BATCHES})
  Overrides:      ${EXTRA_OVERRIDES}
═══════════════════════════════════════════════════════════════
EOF

# aiter and hiprtc deadlock on their first compile if a previous run left a
# stale JIT lock behind.
find /tmp -name '*.lock' -path '*aiter*' -delete 2>/dev/null || true
find /tmp -name '*.lock' -path '*hiprtc*' -delete 2>/dev/null || true

docker run ${RUN_MODE_ARGS} \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    --shm-size 64G \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e PYTORCH_ROCM_ARCH=gfx950 \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e PYTHONUNBUFFERED=1 \
    -e LUMENRL_LOG_LEVEL="${LUMENRL_LOG_LEVEL:-INFO}" \
    -e NCCL_TIMEOUT=7200 \
    -e LUMENRL_TEACHER_READY_TIMEOUT_SECONDS=7200 \
    -e LUMENRL_TEACHER_GENERATE_TIMEOUT_SECONDS="${LUMENRL_TEACHER_GENERATE_TIMEOUT_SECONDS:-}" \
    -e GLOG_minloglevel=3 \
    -e GLOG_v=0 \
    -e GLOG_logtostderr=1 \
    -e MOONCAKE_LOG_LEVEL=FATAL \
    -e MOONCAKE_VLOG_LEVEL=-1 \
    -e MOONCAKE_PROTOCOL=tcp \
    -e WANDB_MODE=disabled \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HSA_COREDUMP_FILE=/dev/null \
    -e HSA_ENABLE_COREDUMP=0 \
    -e MODEL_PATH="${MODEL_PATH}" \
    -e CKPT_DIR="${CKPT_DIR}" \
    -e EXTRA_OVERRIDES="${EXTRA_OVERRIDES}" \
    --log-opt max-size=500m \
    --log-opt max-file=2 \
    -v /dev/shm:/dev/shm \
    -v "${DATA_ROOT}:${DATA_ROOT}" \
    -v "${LUMENRL_DIR}/lumenrl:/root/lumenrl/lumenrl" \
    -v "${LUMENRL_DIR}/examples:/root/lumenrl/examples" \
    -v "${LUMENRL_DIR}/output:/root/lumenrl/output" \
    ${EXTRA_MOUNTS} \
    -w /root/lumenrl \
    "${DOCKER_IMAGE}" \
    bash -c "${RUN_CMD}"

EXIT_CODE=$?

if [ "${DETACH}" = "1" ]; then
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo ">>> Container ${CONTAINER_NAME} started detached."
        # Read back from the container rather than trusting this shell: the
        # setting that broke the round transition survived a "revert" because
        # it was being inherited, and nobody checked what actually landed.
        ALLOC_IN_CONTAINER=$(docker inspect "${CONTAINER_NAME}" \
            --format '{{range .Config.Env}}{{println .}}{{end}}' \
            | grep '^PYTORCH_CUDA_ALLOC_CONF=' || true)
        echo ">>> Allocator: ${ALLOC_IN_CONTAINER:-PYTORCH_CUDA_ALLOC_CONF unset (expected)}"
        echo ">>> Follow:  docker logs -f ${CONTAINER_NAME}"
        echo ">>> Stop:    docker stop ${CONTAINER_NAME}"
    else
        echo ">>> Failed to start ${CONTAINER_NAME} (exit ${EXIT_CODE})." >&2
    fi
    echo ">>> Logs: ${LUMENRL_DIR}/output/Kimi_K3_SDDD/LumenRL/"
    exit ${EXIT_CODE}
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Kimi K3 DSpark (ATOM) completed successfully."
else
    echo ">>> Kimi K3 DSpark (ATOM) failed with exit code ${EXIT_CODE}." >&2
fi
echo ">>> Logs: ${LUMENRL_DIR}/output/Kimi_K3_SDDD/LumenRL/"
exit ${EXIT_CODE}
