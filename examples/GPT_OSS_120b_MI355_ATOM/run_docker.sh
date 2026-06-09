#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# GPT-OSS-120B Eagle3 SDDD — Docker Launch (vLLM 0.11 + Mooncake TCP, MI350)
#
# Step 0: Prepare combined dataset (UltraChat + Magpie → single JSONL)
# Step 1: Launch single-pass Eagle3 training in Docker
#
# GPU split (8x MI350):
#   GPUs 0-3: FSDP2 draft model training (BF16, LumenRL + aiter)
#   GPUs 4-7: vLLM teacher inference (TP=4, native MXFP4 gpt-oss-120b)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

DOCKER_IMAGE="${DOCKER_IMAGE:-lumenrl-vllm-mi350:latest}"
CONTAINER_NAME="gpt_oss_120b_eagle3_mi350"
LUMENRL_DIR="${LUMENRL_DIR:-/home/danyzhan/Lumen-RL}"

# Step 0: Prepare combined dataset (skip for smoke test)
DATASET_DIR="/dev/shm/gpt_oss_120b_dataset"
if [ "${SMOKE_TEST}" = false ] && [ ! -f "${DATASET_DIR}/train.jsonl" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Step 0: Preparing combined dataset (UltraChat + Magpie)"
    echo "═══════════════════════════════════════════════════════════════"
    docker run --rm \
        --name "${CONTAINER_NAME}_data" \
        --network host \
        --ipc host \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e PYTHONUNBUFFERED=1 \
        -v /dev/shm:/dev/shm \
        -v "${LUMENRL_DIR}/examples:/root/lumenrl/examples" \
        -w /root/lumenrl \
        "${DOCKER_IMAGE}" \
        python3 examples/GPT_OSS_120b_MI355_ATOM/make_dataset.py \
            --output-dir "${DATASET_DIR}" --skip-existing
    if [ $? -ne 0 ]; then
        echo ">>> Dataset preparation failed." >&2
        exit 1
    fi
    echo ">>> Dataset ready: ${DATASET_DIR}/train.jsonl"
fi

# Step 1: Launch training
if [ "${SMOKE_TEST}" = true ]; then
    RUN_CMD="bash examples/GPT_OSS_120b_MI355_ATOM/run_gpt_oss_120b.sh --smoke-test"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  GPT-OSS-120B Eagle3 Smoke Test (Docker) — vLLM, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  GPUs:     8x MI350 (0-3 training, 4-7 inference)"
    echo "  Transfer: Mooncake TCP"
    echo "═══════════════════════════════════════════════════════════════"
else
    RUN_CMD="bash examples/GPT_OSS_120b_MI355_ATOM/run_gpt_oss_120b.sh"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  GPT-OSS-120B Eagle3 Training (Docker) — vLLM, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  Dataset:  Combined UltraChat + Magpie (~503K samples)"
    echo "  GPUs:     8x MI350 (0-3 training, 4-7 inference)"
    echo "═══════════════════════════════════════════════════════════════"
fi

# Clean stale lock files and compile caches to prevent AITER JIT deadlocks
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
    -e LUMENRL_TEACHER_READY_TIMEOUT_SECONDS=1800 \
    -e ATOM_WARMUP_MAX_TOKENS=8192 \
    -e ATOM_USE_TRITON_MOE=1 \
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
    -e LUMENRL_DRAFT_FLASH_BACKEND="${LUMENRL_DRAFT_FLASH_BACKEND:-}" \
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
    bash -c "pip install -e /root/triton_kernels 2>/dev/null; ${RUN_CMD}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Docker run (vLLM, MI350) completed successfully."
    echo ">>> Logs: ${LUMENRL_DIR}/output/GPT_OSS_120b_SDDD/LumenRL/"
else
    echo ">>> Docker run (vLLM, MI350) failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
