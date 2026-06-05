#!/usr/bin/env bash
# Experiment 1A-Async: Qwen3-8B-Base — BF16 training + BF16 rollout (Async mode)
#
# Uses LumenRL native AsyncRLTrainer with decoupled rollout and training threads.
# Same hyperparameters as 1A baseline for direct comparison.
#
# ROCm adaptations:
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   - TORCHDYNAMO_DISABLE=1
#   - VLLM_USE_V1=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXP_NAME="1a-bf16-async"
CONFIG="${SCRIPT_DIR}/configs/1a_bf16_async.yaml"
OUTPUT_DIR="${REPO_ROOT}/output/DAPO/${EXP_NAME}"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"
NUM_GPUS="${NUM_GPUS:-8}"

# Preflight
if [ ! -d "/dev/shm/model/qwen3-8b-base" ]; then
    echo "ERROR: Model not found at /dev/shm/model/qwen3-8b-base" >&2
    exit 1
fi
if [ ! -f "/dev/shm/data/dapo-math-17k.parquet" ]; then
    echo "ERROR: Training data not found" >&2
    exit 1
fi

# ROCm environment
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export LUMENRL_LOG_LEVEL=INFO

# Kill any stale Python/torchrun processes
pkill -f "lumenrl.trainer.main" 2>/dev/null || true
pkill -f "pt_elastic" 2>/dev/null || true
sleep 2

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LumenRL 1A-Async — BF16 Fully Async Training              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:    /dev/shm/model/qwen3-8b-base                    ║"
echo "║  GPUs:     ${NUM_GPUS}                                             ║"
echo "║  Mode:     AsyncRLTrainer (decoupled rollout+train)        ║"
echo "║  Config:   ${CONFIG}"
echo "║  Log:      ${LOG_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

OVERRIDES=()
if [ -n "${TOTAL_STEPS:-}" ]; then
    OVERRIDES+=("num_training_steps=${TOTAL_STEPS}")
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT:-29500}" \
    -m lumenrl.trainer.main \
    --config "${CONFIG}" \
    "${OVERRIDES[@]}" \
    2>&1 | tee "${LOG_FILE}"
