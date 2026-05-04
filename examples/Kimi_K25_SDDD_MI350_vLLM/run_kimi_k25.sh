#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K2.5 Eagle3 Draft Model Distillation (vLLM + ATOM + MXFP4) — MI350
#
# Off-policy speculative decoding draft model training using teacher hidden
# states. vLLM with ATOM plugin provides MXFP4-quantized inference on
# ROCm/AMD MI350 GPUs via AITER kernels; Mooncake TCP transfers hidden states
# to training GPUs running LumenRL's aiter-patched FSDP2.
#
# Both inference (ATOM+vLLM) and training (LumenRL FSDP2) use the same
# AITER version from third_party/aiter.
#
# GPU split strategy (8x MI350):
#   GPUs 0-3: torchrun FSDP2 draft model training (BF16, LumenRL + aiter)
#   GPUs 4-7: vLLM + ATOM teacher inference (TP=4, MXFP4, aiter)
#
# Data flow:
#   Dataset Prompts -> vLLM+ATOM Forward (GPUs 4-7) -> hidden states [B,T,D]
#     -> Mooncake TCP -> Eagle3 Draft Forward (GPUs 0-3, with grad)
#     -> Forward KL loss with position decay (0.8^i) -> Backward -> Update
#
# Usage:
#   bash examples/Kimi_K25_SDDD_MI350_vLLM/run_kimi_k25.sh
#   bash examples/Kimi_K25_SDDD_MI350_vLLM/run_kimi_k25.sh --smoke-test
#   MODEL_PATH=/path/to/model bash examples/Kimi_K25_SDDD_MI350_vLLM/run_kimi_k25.sh
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

EXP_NAME="kimi-k25-eagle3-sglang-mxfp4-mi350"
OUTPUT_DIR="${REPO_ROOT}/output/Kimi_K25_SDDD_MI350/${EXP_NAME}"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"

# GPU split: 0-3 training, 4-7 inference (managed by VllmTeacherEngine subprocess)
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"

# Model / data paths (default to /dev/shm)
if [ "${SMOKE_TEST}" = true ]; then
    MODEL_PATH="${MODEL_PATH:-/dev/shm/Qwen3-8B}"
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/kimi_k25_smoke_test_vllm}"
else
    MODEL_PATH="${MODEL_PATH:-/dev/shm/Kimi-K2.5-BF16}"
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/kimi_k25_eagle3_vllm_mxfp4}"
fi

# Config selection
if [ "${SMOKE_TEST}" = true ]; then
    CONFIG="${SCRIPT_DIR}/configs/smoke_test.yaml"
    echo ">>> SMOKE TEST: 2-step Eagle3 validation (vLLM+ATOM+MXFP4+Mooncake TCP, Qwen3-8B)"
else
    CONFIG="${SCRIPT_DIR}/configs/phase1_foundation.yaml"
fi

# Environment
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LUMENRL_LOG_LEVEL=INFO
export NCCL_TIMEOUT=7200

mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Kimi K2.5 Eagle3 Draft Model Distillation (vLLM+ATOM) — MI350"
echo "  Teacher:     ${MODEL_PATH} (vLLM+ATOM, GPUs 4-7, TP=4, MXFP4)"
echo "  Draft:       Eagle3 (from scratch, 1 Transformer block, BF16)"
echo "  Train GPUs:  ${TRAIN_GPUS} (${NUM_TRAIN_GPUS} GPUs, FSDP2+aiter)"
echo "  Transfer:    Mooncake TCP"
echo "  Config:      ${CONFIG}"
echo "  Output:      ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# Build overrides
OVERRIDES=(
    "policy.model_name=${MODEL_PATH}"
    "algorithm.teacher.model_name=${MODEL_PATH}"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
)

# Launch training on GPUs 0-3 (VllmTeacherEngine subprocess manages GPUs 4-7 internally)
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" \
        -m lumenrl.trainer.main \
        --config "${CONFIG}" \
        ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
        2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Kimi K2.5 Eagle3 distillation (vLLM+ATOM, MI350) completed successfully."
else
    echo ">>> Kimi K2.5 Eagle3 distillation (vLLM+ATOM, MI350) failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
