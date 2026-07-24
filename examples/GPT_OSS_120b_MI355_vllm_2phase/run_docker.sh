#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# GPT-OSS-120B Eagle3 SDDD v3 — Docker Launch (Two-Phase, vLLM + Mooncake TCP)
#
# Step 0: Prepare Nemotron PTv3 dataset (phase1_short.jsonl + phase2_long.jsonl)
# Step 1: Launch training for the specified phase
#
# GPU split (8x MI355):
#   GPUs 0-3: FSDP2 draft model training (BF16, LumenRL + aiter)
#   GPUs 4-7: vLLM teacher inference (TP=4, BF16, FP8 KV cache)
#
# Usage:
#   PHASE=1 bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_docker.sh
#   PHASE=2 bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_docker.sh
#   bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_docker.sh --smoke-test
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
PHASE="${PHASE:-1}"
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
        --phase=*) PHASE="${arg#*=}" ;;
    esac
done

DOCKER_IMAGE="${DOCKER_IMAGE:-gpt_oss_eagle3_vllm_train:latest}"
CONTAINER_NAME="gpt_oss_120b_eagle3_v3_phase${PHASE}_mi355"
LUMENRL_DIR="${LUMENRL_DIR:-/home/danyzhan/Lumen-RL}"

# Step 0: Prepare Nemotron PTv3 dataset (skip for smoke test)
DATASET_DIR="/dev/shm/gpt_oss_120b_dataset_v3"
if [ "${SMOKE_TEST}" = false ] && [ ! -f "${DATASET_DIR}/phase1_short.jsonl" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Step 0: Preparing Nemotron PTv3 dataset (two-phase split)"
    echo "═══════════════════════════════════════════════════════════════"
    docker run --rm \
        --name "${CONTAINER_NAME}_data" \
        --network host \
        --ipc host \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e PYTHONUNBUFFERED=1 \
        -v /dev/shm:/dev/shm \
        -v "${LUMENRL_DIR}:/root/lumenrl" \
        -w /root/lumenrl \
        "${DOCKER_IMAGE}" \
        python3 examples/GPT_OSS_120b_MI355_vllm_2phase/make_dataset_nemotron_ptv3.py \
            --output-dir "${DATASET_DIR}" \
            --tokenizer /dev/shm/gpt-oss-120b \
            --skip-existing
    if [ $? -ne 0 ]; then
        echo ">>> Dataset preparation failed." >&2
        exit 1
    fi
    echo ">>> Dataset ready: ${DATASET_DIR}/"
fi

# Step 1: Launch training
if [ "${SMOKE_TEST}" = true ]; then
    RUN_CMD="bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_gpt_oss_120b.sh --smoke-test"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  GPT-OSS-120B Eagle3 v3 Smoke Test (Docker)"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "═══════════════════════════════════════════════════════════════"
else
    RUN_CMD="PHASE=${PHASE} bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_gpt_oss_120b.sh --phase=${PHASE}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  GPT-OSS-120B Eagle3 v3 Phase ${PHASE} Training (Docker)"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  Dataset:  Nemotron PTv3 (phase${PHASE})"
    echo "  GPUs:     8x MI355 (0-3 training, 4-7 inference)"
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
    -e PHASE="${PHASE}" \
    -e LUMENRL_DRAFT_FLASH_BACKEND="${LUMENRL_DRAFT_FLASH_BACKEND:-}" \
    -e HSA_COREDUMP_FILE=/dev/null \
    -e HSA_ENABLE_COREDUMP=0 \
    -v /dev/shm:/dev/shm \
    -v "${LUMENRL_DIR}/lumenrl:/root/lumenrl/lumenrl" \
    -v "${LUMENRL_DIR}/examples:/root/lumenrl/examples" \
    -v "${LUMENRL_DIR}/output:/root/lumenrl/output" \
    -v "${LUMENRL_DIR}/third_party/Lumen/lumen:/root/Lumen/lumen" \
    -w /root/lumenrl \
    "${DOCKER_IMAGE}" \
    bash -c "${RUN_CMD}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Docker Phase ${PHASE} (vLLM, MI355) completed successfully."
    echo ">>> Logs: ${LUMENRL_DIR}/output/GPT_OSS_120b_SDDD_v3/LumenRL/"
else
    echo ">>> Docker Phase ${PHASE} (vLLM, MI355) failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
