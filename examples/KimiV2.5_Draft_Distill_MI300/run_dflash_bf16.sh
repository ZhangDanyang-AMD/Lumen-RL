#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# KimiV2.5 DFlash Draft Model Distillation (4+4 GPU split)
#
# Off-policy DFlash draft model training with block-causal masking and
# cross-entropy loss with exponential decay weighting.
#
# GPU split strategy (8x MI300X):
#   GPUs 0-3: ATOM teacher inference (TP=4, MXFP4 quantization)
#   GPUs 4-7: torchrun FSDP2 draft model training (BF16)
#
# Data flow:
#   Dataset Prompts → ATOM Teacher Forward (GPUs 0-3) → hidden states [B,T,D]
#     → shared memory → DFlash Draft Forward (GPUs 4-7, with grad)
#     → CE loss with exp(-i/gamma) weighting
#     → Backward → Update Draft Model
#
# Usage:
#   bash examples/KimiV2.5_Draft_Distill_MI300/run_dflash_bf16.sh
#   bash examples/KimiV2.5_Draft_Distill_MI300/run_dflash_bf16.sh --smoke-test
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

EXP_NAME="kimiv2.5-dflash-bf16"
OUTPUT_DIR="${REPO_ROOT}/output/KimiV2.5_Draft_Distill_MI300/${EXP_NAME}"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"

# GPU configuration
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"

MODEL_PATH="${MODEL_PATH:-/dev/shm/model/kimi-k2.5}"
DATA_DIR="${DATA_DIR:-/dev/shm/data}"
CKPT_DIR="${CKPT_DIR:-/dev/shm/ckpts/spec-distill/dflash-bf16}"
CONFIG="${SCRIPT_DIR}/configs/dflash_bf16.yaml"

if [ "${SMOKE_TEST}" = true ]; then
    CONFIG="${SCRIPT_DIR}/configs/smoke_test.yaml"
    echo ">>> SMOKE TEST: 2-step DFlash distillation (4+4 GPU split)"
fi

# Preflight
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}" >&2
    echo "  Set MODEL_PATH or download: huggingface-cli download moonshotai/Kimi-K2.5 --local-dir ${MODEL_PATH}" >&2
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
echo "  KimiV2.5 DFlash Draft Model Distillation"
echo "  Teacher:     ${MODEL_PATH} (ATOM, GPUs 0-3, TP=4)"
echo "  Draft:       DFlash (from scratch, 2 Transformer blocks, BF16)"
echo "  Train GPUs:  ${TRAIN_GPUS} (${NUM_TRAIN_GPUS} GPUs)"
echo "  Config:      ${CONFIG}"
echo "═══════════════════════════════════════════════════════════════"

OVERRIDES=()
if [ "${SMOKE_TEST}" = false ]; then
    OVERRIDES+=(
        "policy.model_name=${MODEL_PATH}"
        "algorithm.teacher.model_name=${MODEL_PATH}"
        "algorithm.spec_distill.draft_type=dflash"
        "checkpointing.checkpoint_dir=${CKPT_DIR}"
    )
    if [ -f "${DATA_DIR}/dapo-math-17k.parquet" ]; then
        OVERRIDES+=("reward.dataset=${DATA_DIR}/dapo-math-17k.parquet")
    fi
fi

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" \
        -m lumenrl.trainer.main \
        --config "${CONFIG}" \
        "${OVERRIDES[@]}" \
        2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> DFlash distillation completed successfully."
else
    echo ">>> DFlash distillation failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
