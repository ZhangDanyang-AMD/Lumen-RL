#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K2.5 Eagle3 Draft Model Distillation (SGLang + ATOM + MXFP4)
#
# Off-policy speculative decoding draft model training using teacher hidden
# states. SGLang Engine with ATOM plugin provides MXFP4-quantized inference
# on ROCm/AMD GPUs; Mooncake RDMA transfers hidden states to training GPUs.
#
# GPU split strategy (8x GPU):
#   GPUs 0-3: torchrun FSDP2 draft model training (BF16)
#   GPUs 4-7: SGLang + ATOM teacher inference (TP=4, MXFP4)
#
# Data flow:
#   Dataset Prompts -> SGLang+ATOM Forward (GPUs 4-7) -> hidden states [B,T,D]
#     -> Mooncake RDMA -> Eagle3 Draft Forward (GPUs 0-3, with grad)
#     -> Forward KL loss with position decay (0.8^i) -> Backward -> Update
#
# Usage:
#   bash examples/Kimi_K25_SDDD/run_kimi_k25.sh
#   bash examples/Kimi_K25_SDDD/run_kimi_k25.sh --smoke-test
#   MODEL_PATH=/path/to/model bash examples/Kimi_K25_SDDD/run_kimi_k25.sh
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

EXP_NAME="kimi-k25-eagle3-sglang-mxfp4"
OUTPUT_DIR="${REPO_ROOT}/output/Kimi_K25_SDDD/${EXP_NAME}"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"

# GPU split: 0-3 training, 4-7 inference (managed by SGLang subprocess)
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"

# Model / data paths
MODEL_PATH="${MODEL_PATH:-/datasets/Kimi-K2.5-BF16}"
CKPT_DIR="${CKPT_DIR:-results/kimi_k25_eagle3_sglang_mxfp4}"

# Config selection
if [ "${SMOKE_TEST}" = true ]; then
    CONFIG="${SCRIPT_DIR}/configs/smoke_test.yaml"
    echo ">>> SMOKE TEST: 2-step Eagle3 validation (no ATOM/SGLang/Mooncake)"
else
    CONFIG="${SCRIPT_DIR}/configs/phase1_foundation.yaml"
fi

# Environment
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LUMENRL_LOG_LEVEL=INFO
export NCCL_TIMEOUT=7200

# SGLang-specific environment
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1

# ATOM + SGLang plugin
ATOM_REPO="${ATOM_REPO:-/home/danyzhan/ATOM_repo}"
if [ -d "${ATOM_REPO}" ]; then
    export PYTHONPATH="${ATOM_REPO}:${PYTHONPATH:-}"
fi

mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Kimi K2.5 Eagle3 Draft Model Distillation (SGLang+ATOM)"
echo "  Teacher:     ${MODEL_PATH} (SGLang+ATOM, GPUs 4-7, TP=4, MXFP4)"
echo "  Draft:       Eagle3 (from scratch, 1 Transformer block, BF16)"
echo "  Train GPUs:  ${TRAIN_GPUS} (${NUM_TRAIN_GPUS} GPUs, FSDP2)"
echo "  Transfer:    Mooncake RDMA"
echo "  Config:      ${CONFIG}"
echo "  Output:      ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# Build overrides
OVERRIDES=()
if [ "${SMOKE_TEST}" = false ]; then
    OVERRIDES+=(
        "policy.model_name=${MODEL_PATH}"
        "algorithm.teacher.model_name=${MODEL_PATH}"
        "checkpointing.checkpoint_dir=${CKPT_DIR}"
    )
fi

# Launch training on GPUs 0-3 (SGLang subprocess manages GPUs 4-7 internally)
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" \
        -m lumenrl.trainer.main \
        --config "${CONFIG}" \
        ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
        2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Kimi K2.5 Eagle3 distillation completed successfully."
else
    echo ">>> Kimi K2.5 Eagle3 distillation failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
