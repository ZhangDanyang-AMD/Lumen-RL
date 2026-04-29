#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3-8B Eagle3 Speculative Decoding Draft Distillation
#
# Disaggregated architecture (matches TorchSpec qwen3-8b-single-node):
#   - 2 GPUs for inference: SGLang Engine + ATOM plugin (Aiter kernels)
#   - 2 GPUs for training:  FSDP2 Eagle3 draft model + Aiter CK kernels
#   - Mooncake RDMA for hidden state transfer between inference and training
#
# Data flow:
#   Dataset -> Teacher Forward (SGLang, GPU 2-3)
#     -> hidden_states via Mooncake RDMA -> Training GPUs (GPU 0-1)
#     -> Eagle3 draft model training (position-weighted CE loss)
#
# Usage:
#   bash examples/Qwen3_8B_SDDD/run_opd.sh
#   bash examples/Qwen3_8B_SDDD/run_opd.sh --smoke-test
#   MODEL_PATH=/path/to/model bash examples/Qwen3_8B_SDDD/run_opd.sh
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

EXP_NAME="qwen3-8b-spec-distill"
OUTPUT_DIR="${REPO_ROOT}/output/Qwen3_8B_SDDD/${EXP_NAME}"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"

# GPU allocation: 2 training + 2 inference = 4 total
TRAIN_GPUS="${TRAIN_GPUS:-2}"
TOTAL_GPUS="${TOTAL_GPUS:-4}"

# Model paths
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
TEACHER_PATH="${TEACHER_PATH:-${MODEL_PATH}}"
CKPT_DIR="${CKPT_DIR:-results/qwen3_8b_spec_distill}"

# Config
if [ "${SMOKE_TEST}" = true ]; then
    CONFIG="${SCRIPT_DIR}/configs/smoke_test.yaml"
    echo ">>> SMOKE TEST: 2-step Eagle3 distillation with Qwen3-8B"
else
    CONFIG="${SCRIPT_DIR}/configs/opd_qwen3_8b.yaml"
fi

# ROCm environment
export PYTHONUNBUFFERED=1
# TORCHDYNAMO_DISABLE=1  # Disabled: FlexAttention requires torch.compile
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_DISABLE_FRAGMENT_ALLOCATOR=1
export LUMENRL_LOG_LEVEL=INFO
export NCCL_TIMEOUT=7200
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# ATOM + SGLang plugin (Aiter-optimized inference kernels)
ATOM_REPO="${ATOM_REPO:-/home/danyzhan/ATOM_repo}"
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export PYTHONPATH="${ATOM_REPO}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Qwen3-8B Eagle3 Speculative Decoding Draft Distillation"
echo "  Teacher: ${TEACHER_PATH}"
echo "  Config:  ${CONFIG}"
echo "  Training GPUs: ${TRAIN_GPUS} (torchrun)"
echo "  Inference GPUs: dedicated (configured in YAML)"
echo "  Total GPUs: ${TOTAL_GPUS}"
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
fi

# Auto-restart loop for crash recovery
MAX_RETRIES=10
for attempt in $(seq 1 ${MAX_RETRIES}); do
    echo ">>> Attempt ${attempt}/${MAX_RETRIES}"

    if [ "${TRAIN_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${TRAIN_GPUS}" \
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
        echo ">>> Training completed successfully."
        exit 0
    fi

    echo ">>> Attempt ${attempt} failed (exit ${EXIT_CODE}). Retrying in 10s..." >&2
    sleep 10
done

echo ">>> Training failed after ${MAX_RETRIES} attempts." >&2
exit 1
