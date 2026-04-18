#!/usr/bin/env bash
# Experiment 1A — BF16 Baseline on 8×GPU (Native LumenRL + FSDP2)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"
CONFIG="${SCRIPT_DIR}/configs/1a_bf16_baseline.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/DAPO/1a-bf16-8gpu}"
NUM_GPUS="${NUM_GPUS:-8}"
STEPS="${STEPS:-275}"

errors=0
[ ! -d "${MODEL_PATH}" ] && echo "ERROR: Model not found: ${MODEL_PATH}" >&2 && errors=1
[ ! -f "${CONFIG}" ]     && echo "ERROR: Config not found: ${CONFIG}" >&2 && errors=1
python3 -c "import lumenrl" 2>/dev/null || { echo "ERROR: lumenrl not importable" >&2; errors=1; }
[ "${errors}" -ne 0 ] && exit 1

mkdir -p "${OUTPUT_DIR}"

export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LUMENRL_LOG_LEVEL=INFO

echo ""
echo "========================================================"
echo "  LumenRL Native — Exp 1A: BF16 Baseline (${NUM_GPUS} GPU)"
echo "  Model:   ${MODEL_PATH}"
echo "  Config:  ${CONFIG}"
echo "  Steps:   ${STEPS}"
echo "  Output:  ${OUTPUT_DIR}"
echo "========================================================"
echo ""

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29500 \
    -m lumenrl.trainer.main \
    --config "${CONFIG}" \
    policy.model_name="${MODEL_PATH}" \
    checkpointing.checkpoint_dir="${OUTPUT_DIR}/ckpts" \
    num_training_steps="${STEPS}" \
    2>&1 | tee "${OUTPUT_DIR}/1a-bf16-8gpu.log"
