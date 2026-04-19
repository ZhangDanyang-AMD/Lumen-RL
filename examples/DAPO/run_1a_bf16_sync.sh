#!/usr/bin/env bash
# Experiment 1A-Sync: Qwen3-8B-Base — BF16 training + BF16 rollout (Sync on-policy)
#
# Uses LumenRL native RLTrainer (sync) with ATOM for generation.
# On-policy: generates with current weights via weight sync, trains on that data.
# FSDP2 param_offload + optimizer offload to free GPU memory for ATOM rollout.
#
# Offload targets /dev/shm (tmpfs) for fast CPU↔GPU transfer.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXP_NAME="1a-bf16-sync"
CONFIG="${SCRIPT_DIR}/configs/1a_bf16_sync.yaml"
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
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_DISABLE_FRAGMENT_ALLOCATOR=1
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export LUMENRL_LOG_LEVEL=INFO
export LUMENRL_WEIGHT_SYNC_DIR=/dev/shm/lumenrl_weight_sync
export NCCL_TIMEOUT=7200

# Kill any stale processes
pkill -f "lumenrl.trainer.main" 2>/dev/null || true
pkill -f "pt_elastic" 2>/dev/null || true
sleep 2

mkdir -p "${OUTPUT_DIR}"
mkdir -p /dev/shm/lumenrl_weight_sync
mkdir -p /dev/shm/ckpts/lumenrl-dapo/1a-bf16-sync

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LumenRL 1A-Sync — BF16 On-Policy Training                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:    /dev/shm/model/qwen3-8b-base                    ║"
echo "║  GPUs:     ${NUM_GPUS}                                             ║"
echo "║  Mode:     RLTrainer (sync on-policy + ATOM rollout)       ║"
echo "║  Offload:  /dev/shm (optimizer + FSDP2 param + weights)    ║"
echo "║  Config:   ${CONFIG}"
echo "║  Log:      ${LOG_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

OVERRIDES=()
if [ -n "${TOTAL_STEPS:-}" ]; then
    OVERRIDES+=("num_training_steps=${TOTAL_STEPS}")
fi

MAX_RETRIES="${MAX_RETRIES:-50}"
RETRY_DELAY="${RETRY_DELAY:-10}"

for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo ""
    echo "=== Attempt ${attempt}/${MAX_RETRIES} ($(date)) ==="
    echo ""

    pkill -9 -f "python.*vllm" 2>/dev/null || true
    sleep 2

    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT:-29500}" \
        -m lumenrl.trainer.main \
        --config "${CONFIG}" \
        "${OVERRIDES[@]}" \
        >> "${LOG_FILE}" 2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "Training completed successfully on attempt ${attempt}."
        exit 0
    fi

    echo ""
    echo "*** Training crashed (exit=${EXIT_CODE}). Cleaning up GPU state... ***"
    pkill -9 -f "python.*lumenrl" 2>/dev/null || true
    pkill -9 -f "python.*vllm" 2>/dev/null || true
    sleep "${RETRY_DELAY}"
done

echo "ERROR: Training failed after ${MAX_RETRIES} attempts." >&2
exit 1
