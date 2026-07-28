#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K3 DSpark SDDD — Docker Launch (vLLM, 8-GPU Sequential, MI350)
#
# Step 0: Prepare dataset (split kimi-mtp-dataset into phases)
# Step 1: Launch sequential DSpark training in Docker
#
# 8-GPU sequential mode (all GPUs for infer, then all for train):
#   Phase A: GPUs 0-7 — vLLM teacher inference (TP=8, MXFP4 MoE)
#   Phase B: GPUs 0-7 — torchrun FSDP2 DSpark training (BF16)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

DOCKER_IMAGE="${DOCKER_IMAGE:-kimi_k3_dspark_vllm_train:latest}"
CONTAINER_NAME="kimi_k3_dspark_vllm_mi350"
LUMENRL_DIR="${LUMENRL_DIR:-/home/danyzhan/Lumen-RL}"

# Step 0: Download dataset if not present
DATASET_DIR="/dev/shm/kimi-mtp-dataset"
if [ ! -d "${DATASET_DIR}" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Step 0: Downloading kimi-mtp-dataset"
    echo "═══════════════════════════════════════════════════════════════"
    docker run --rm \
        --name "${CONTAINER_NAME}_data" \
        --network host \
        --ipc host \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e PYTHONUNBUFFERED=1 \
        -v /dev/shm:/dev/shm \
        "${DOCKER_IMAGE}" \
        huggingface-cli download lightseekorg/kimi-mtp-dataset \
            --local-dir "${DATASET_DIR}"
    if [ $? -ne 0 ]; then
        echo ">>> Dataset download failed." >&2
        exit 1
    fi
    echo ">>> Dataset ready: ${DATASET_DIR}"
fi

# Step 1: Launch training
if [ "${SMOKE_TEST}" = true ]; then
    RUN_CMD="bash examples/Kimi_K3_SDDD_MI350_vllm/run_kimi_k3.sh --smoke-test"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Kimi K3 DSpark Smoke Test (Docker) — vLLM TP=8, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  GPUs:     8× MI350 (sequential: infer → train)"
    echo "  Teacher:  Kimi-K3 (vLLM, TP=8, MXFP4 MoE)"
    echo "  Draft:    DSpark (5-layer, Markov+confidence)"
    echo "═══════════════════════════════════════════════════════════════"
else
    RUN_CMD="bash examples/Kimi_K3_SDDD_MI350_vllm/run_kimi_k3.sh"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Kimi K3 DSpark Training (Docker) — vLLM TP=8, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  Dataset:  kimi-mtp-dataset (~477K perfectblend)"
    echo "  GPUs:     8× MI350 (sequential: infer → train)"
    echo "═══════════════════════════════════════════════════════════════"
fi

# Clean stale lock files and compile caches
find /tmp -name '*.lock' -path '*aiter*' -delete 2>/dev/null || true
find /tmp -name '*.lock' -path '*hiprtc*' -delete 2>/dev/null || true

docker run --rm \
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
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e LUMENRL_LOG_LEVEL=INFO \
    -e NCCL_TIMEOUT=7200 \
    -e LUMENRL_TEACHER_READY_TIMEOUT_SECONDS=3600 \
    -e GLOG_minloglevel=3 \
    -e GLOG_v=0 \
    -e GLOG_logtostderr=1 \
    -e MOONCAKE_LOG_LEVEL=FATAL \
    -e MOONCAKE_VLOG_LEVEL=-1 \
    -e WANDB_MODE=disabled \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    --log-opt max-size=500m \
    --log-opt max-file=2 \
    -e MODEL_PATH="${MODEL_PATH:-}" \
    -e CKPT_DIR="${CKPT_DIR:-}" \
    -e HSA_COREDUMP_FILE=/dev/null \
    -e HSA_ENABLE_COREDUMP=0 \
    -v /dev/shm:/dev/shm \
    -v "${LUMENRL_DIR}/lumenrl:/root/lumenrl/lumenrl" \
    -v "${LUMENRL_DIR}/examples:/root/lumenrl/examples" \
    -v "${LUMENRL_DIR}/output:/root/lumenrl/output" \
    -v "${LUMENRL_DIR}/third_party/Lumen/lumen:/root/Lumen/lumen" \
    -v "${LUMENRL_DIR}/third_party/ATOM/atom:/root/ATOM/atom" \
    -v "${LUMENRL_DIR}/third_party/triton_kernels:/root/triton_kernels" \
    -w /root/lumenrl \
    "${DOCKER_IMAGE}" \
    bash -c "pip install -e /root/triton_kernels 2>/dev/null; find /root/aiter -name 'amd_buffer_addressing_builtins.hpp' -exec sed -i 's/#if __clang_major__ >= 21 && __clang_major__ < 23/#if 0/' {} \; 2>/dev/null; rm -rf /root/aiter/aiter/jit/build/module_moe_ck2stages* 2>/dev/null; ${RUN_CMD}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Docker run (Kimi K3 DSpark, MI350) completed successfully."
    echo ">>> Logs: ${LUMENRL_DIR}/output/Kimi_K3_SDDD/LumenRL/"
else
    echo ">>> Docker run (Kimi K3 DSpark, MI350) failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
