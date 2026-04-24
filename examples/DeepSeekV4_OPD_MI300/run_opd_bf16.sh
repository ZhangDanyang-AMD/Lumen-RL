#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# DeepSeek-V4 On-Policy Distillation (OPD)
#
# On-policy distillation: student generates sequences, teacher provides
# logits, student minimises KL(student || teacher). Replaces the mixed RL
# stage from DeepSeek-V4's training pipeline.
#
# Data flow:
#   Student Rollout (no grad) → sequences
#     → Teacher Forward (no grad) → logits [B,T,V]
#     → Student Forward (with grad) → student logits [B,T,V]
#     → KL(student || teacher) → Backward → Update Student
#
# Usage:
#   bash examples/DeepSeekV4_OPD_MI300/run_opd_bf16.sh
#   MODEL_PATH=/path/to/model bash examples/DeepSeekV4_OPD_MI300/run_opd_bf16.sh
#   bash examples/DeepSeekV4_OPD_MI300/run_opd_bf16.sh --smoke-test
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXP_NAME="deepseekv4-opd-bf16"
OUTPUT_DIR="${REPO_ROOT}/output/DeepSeekV4_OPD_MI300/${EXP_NAME}"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"
NUM_GPUS="${NUM_GPUS:-8}"

# Model / data paths
MODEL_PATH="${MODEL_PATH:-/dev/shm/model/deepseek-v4-pro}"
TEACHER_PATH="${TEACHER_PATH:-${MODEL_PATH}}"
DATA_DIR="${DATA_DIR:-/dev/shm/data}"
CKPT_DIR="${CKPT_DIR:-/dev/shm/ckpts/opd/bf16}"

# Config
if [ "${SMOKE_TEST}" = true ]; then
    CONFIG="${SCRIPT_DIR}/configs/smoke_test.yaml"
    echo ">>> SMOKE TEST: 2-step OPD on 8x MI300X"
else
    CONFIG="${SCRIPT_DIR}/configs/opd_bf16.yaml"
fi

# Preflight
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Student model not found at ${MODEL_PATH}" >&2
    echo "  Set MODEL_PATH or download: huggingface-cli download deepseek-ai/DeepSeek-V4-Pro --local-dir ${MODEL_PATH}" >&2
    exit 1
fi

# ROCm environment
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_DISABLE_FRAGMENT_ALLOCATOR=1
export LUMENRL_LOG_LEVEL=INFO
export NCCL_TIMEOUT=7200

mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  DeepSeek-V4 On-Policy Distillation (OPD)"
echo "  Student: ${MODEL_PATH}"
echo "  Teacher: ${TEACHER_PATH}"
echo "  Config:  ${CONFIG}"
echo "  GPUs:    ${NUM_GPUS}"
echo "  Output:  ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# Build overrides
OVERRIDES=()
if [ "${SMOKE_TEST}" = false ]; then
    OVERRIDES+=(
        "policy.model_name=${MODEL_PATH}"
        "algorithm.teacher.model_name=${TEACHER_PATH}"
        "checkpointing.checkpoint_dir=${CKPT_DIR}"
    )
    if [ -f "${DATA_DIR}/dapo-math-17k.parquet" ]; then
        OVERRIDES+=("reward.dataset=${DATA_DIR}/dapo-math-17k.parquet")
    fi
fi

# Auto-restart loop for crash recovery (ROCm page faults)
MAX_RETRIES=10
for attempt in $(seq 1 ${MAX_RETRIES}); do
    echo ">>> Attempt ${attempt}/${MAX_RETRIES}"

    if [ "${NUM_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${NUM_GPUS}" \
            -m lumenrl.trainer.main \
            --config "${CONFIG}" \
            "${OVERRIDES[@]}" \
            2>&1 | tee -a "${LOG_FILE}"
    else
        python -m lumenrl.trainer.main \
            --config "${CONFIG}" \
            "${OVERRIDES[@]}" \
            2>&1 | tee -a "${LOG_FILE}"
    fi

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo ">>> OPD training completed successfully."
        exit 0
    fi

    echo ">>> Attempt ${attempt} failed (exit ${EXIT_CODE}). Retrying in 10s..." >&2
    sleep 10
done

echo ">>> OPD training failed after ${MAX_RETRIES} attempts." >&2
exit 1
