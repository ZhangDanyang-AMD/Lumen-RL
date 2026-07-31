#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K3 DSpark SDDD — Docker Launch on the Spur cluster (crsuse2-m2m-*)
#
# Differences from run_docker.sh, all forced by the Spur node layout:
#   - Teacher weights are 1.56 TB, so they live on /mnt/m2m_nobackup (28 TB),
#     not /dev/shm (1.3 TB). That path must be bind-mounted explicitly.
#   - The NFS home is only mountable as /home/<user>:/home/<user> (spur-authz).
#   - Teacher hidden-state cache stays on /dev/shm; cache_batches is sized so
#     one round fits (~5.6 GB per batch at B=8, T=8192).
#   - Dataset download is skipped: it is pre-staged on the host /dev/shm, which
#     is a different tmpfs from the one seen inside `spur exec`.
#
# Usage:
#   bash examples/Kimi_K3_SDDD_MI350_vllm/run_docker_spur.sh --smoke-test
#   CACHE_BATCHES=140 bash examples/Kimi_K3_SDDD_MI350_vllm/run_docker_spur.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

USER_NAME="${USER_NAME:-jimguo12}"
NFS_HOME="/home/${USER_NAME}"
NVME_ROOT="/mnt/m2m_nobackup/${USER_NAME}"

# Built from Dockerfile.train.k3img: stable vLLM releases do not know the
# KimiK3ForConditionalGeneration architecture, only the kimi-k3 pre-release does.
DOCKER_IMAGE="${DOCKER_IMAGE:-kimi_k3_dspark_k3img:latest}"
LUMENRL_DIR="${LUMENRL_DIR:-${NFS_HOME}/Lumen-RL}"
MODEL_PATH="${MODEL_PATH:-${NVME_ROOT}/models/Kimi-K3}"

export HOME="${NFS_HOME}"
export DOCKER_CONFIG="${NFS_HOME}/.docker"

if [ "${SMOKE_TEST}" = true ]; then
    CONTAINER_NAME="${CONTAINER_NAME:-kimi_k3_dspark_smoke}"
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/kimi_k3_dspark_smoke_test}"
    CACHE_DIR="${CACHE_DIR:-/dev/shm/teacher_cache_smoke}"
    CACHE_BATCHES="${CACHE_BATCHES:-5}"
    RUN_CMD="bash examples/Kimi_K3_SDDD_MI350_vllm/run_kimi_k3.sh --smoke-test"
else
    CONTAINER_NAME="${CONTAINER_NAME:-kimi_k3_dspark_v1}"
    # Checkpoints go to NVMe: /dev/shm is wiped when the allocation ends.
    CKPT_DIR="${CKPT_DIR:-${NVME_ROOT}/checkpoints/kimi_k3_dspark_vllm}"
    CACHE_DIR="${CACHE_DIR:-/dev/shm/teacher_cache}"
    CACHE_BATCHES="${CACHE_BATCHES:-140}"
    RUN_CMD="bash examples/Kimi_K3_SDDD_MI350_vllm/run_kimi_k3.sh"
fi

EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
EXTRA_OVERRIDES="algorithm.spec_distill.cache_dir=${CACHE_DIR} algorithm.spec_distill.cache_batches=${CACHE_BATCHES} ${EXTRA_OVERRIDES}"

# Lumen's kernels are built against its own aiter fork; the kimi-k3 base ships
# the AMD aiter that the K3 inference kernels need. Keep them off until the
# draft-training path is proven, then flip to 1 to measure the speedup.
if [ "${LUMEN_KERNELS:-0}" != "1" ]; then
    EXTRA_OVERRIDES="${EXTRA_OVERRIDES} quantization.training.lumen_norm=false quantization.training.lumen_linear=false quantization.training.hf_attn_patch=false"
fi

# A foreground `docker run` forwards SIGTERM into the container, so whatever
# reaps the launching shell's process tree (spur exec tearing down its session)
# takes the training run down with it, silently and without an exit code.
# Detached containers are owned by the docker daemon instead, so they survive.
DETACH="${DETACH:-0}"
if [ "${DETACH}" = "1" ]; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    RUN_MODE_ARGS="-d"
else
    RUN_MODE_ARGS="--rm"
fi

mkdir -p "${CKPT_DIR%/*}" 2>/dev/null || true

cat <<EOF
═══════════════════════════════════════════════════════════════
  Kimi K3 DSpark — Spur launch $([ "${SMOKE_TEST}" = true ] && echo "(SMOKE TEST)")
  Image:          ${DOCKER_IMAGE}
  Container:      ${CONTAINER_NAME}
  LumenRL:        ${LUMENRL_DIR}
  Teacher model:  ${MODEL_PATH}
  Checkpoints:    ${CKPT_DIR}
  Teacher cache:  ${CACHE_DIR}  (cache_batches=${CACHE_BATCHES})
  Overrides:      ${EXTRA_OVERRIDES}
═══════════════════════════════════════════════════════════════
EOF

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
    -e VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}" \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}" \
    -e LUMENRL_LOG_LEVEL="${LUMENRL_LOG_LEVEL:-INFO}" \
    -e NCCL_TIMEOUT=7200 \
    -e LUMENRL_TEACHER_READY_TIMEOUT_SECONDS=7200 \
    -e LUMENRL_TEACHER_GPU_MEM_UTIL="${LUMENRL_TEACHER_GPU_MEM_UTIL:-0.85}" \
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
    -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
    -v "${NFS_HOME}:${NFS_HOME}" \
    -v "${LUMENRL_DIR}/lumenrl:/root/lumenrl/lumenrl" \
    -v "${LUMENRL_DIR}/examples:/root/lumenrl/examples" \
    -v "${LUMENRL_DIR}/output:/root/lumenrl/output" \
    -v "${LUMENRL_DIR}/third_party/Lumen/lumen:/root/Lumen/lumen" \
    -w /root/lumenrl \
    "${DOCKER_IMAGE}" \
    bash -c "find /root/aiter -name 'amd_buffer_addressing_builtins.hpp' -exec sed -i 's/#if __clang_major__ >= 21 \&\& __clang_major__ < 23/#if 0/' {} \; 2>/dev/null; rm -rf /root/aiter/aiter/jit/build/module_moe_ck2stages* 2>/dev/null; ${RUN_CMD}"

EXIT_CODE=$?

if [ "${DETACH}" = "1" ]; then
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo ">>> Container ${CONTAINER_NAME} started detached."
        echo ">>> Follow:  docker logs -f ${CONTAINER_NAME}"
        echo ">>> Stop:    docker stop ${CONTAINER_NAME}"
    else
        echo ">>> Failed to start ${CONTAINER_NAME} (exit ${EXIT_CODE})." >&2
    fi
    echo ">>> Logs: ${LUMENRL_DIR}/output/Kimi_K3_SDDD/LumenRL/"
    exit ${EXIT_CODE}
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Kimi K3 DSpark (Spur) completed successfully."
else
    echo ">>> Kimi K3 DSpark (Spur) failed with exit code ${EXIT_CODE}." >&2
fi
echo ">>> Logs: ${LUMENRL_DIR}/output/Kimi_K3_SDDD/LumenRL/"
exit ${EXIT_CODE}
