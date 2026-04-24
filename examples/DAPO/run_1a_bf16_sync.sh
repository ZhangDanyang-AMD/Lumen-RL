#!/usr/bin/env bash
# Experiment 1A-Sync: Qwen3-8B-Base — BF16 training + BF16 rollout (Sync on-policy)
#
# Uses LumenRL native RLTrainer (sync) with ATOM for generation.
# On-policy: generates with current weights via weight sync, trains on that data.
# FSDP2 param_offload + optimizer offload to free GPU memory for ATOM rollout.
#
# Paths default to /home/danyzhan for MI300X bare-metal setups.
# Override MODEL_PATH / DATA_DIR / CKPT_DIR to use different locations.
set -uo pipefail

# ─── CLI flags ───────────────────────────────────────────────────────────────
SMOKE_TEST=false
DRY_RUN=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
        --dry-run)    DRY_RUN=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXP_NAME="1a-bf16-sync"
CONFIG="${SCRIPT_DIR}/configs/1a_bf16_sync.yaml"
OUTPUT_DIR="${REPO_ROOT}/output/DAPO/${EXP_NAME}"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"
NUM_GPUS="${NUM_GPUS:-8}"

# ─── Data / model paths ─────────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-/home/danyzhan/model}"
DATA_DIR="${DATA_DIR:-/home/danyzhan/data}"
CKPT_DIR="${CKPT_DIR:-/home/danyzhan/ckpts/lumenrl-dapo/1a-bf16-sync}"

# Preflight
if [ ! -d "${MODEL_DIR}/qwen3-8b-base" ]; then
    echo "ERROR: Model not found at ${MODEL_DIR}/qwen3-8b-base" >&2
    exit 1
fi
if [ ! -f "${DATA_DIR}/dapo-math-17k.parquet" ]; then
    echo "ERROR: Training data not found at ${DATA_DIR}/dapo-math-17k.parquet" >&2
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

# ─── ATOM/vLLM attention backend ────────────────────────────────────────────
# Try ROCM_AITER_FA (CK/ASM Flash Attention) first.
# If it fails at runtime (e.g. gfx950 fmha_fwd_v3 symbol undefined on MI300X),
# the retry loop will fall back to ROCM_AITER_UNIFIED_ATTN.
ATTN_BACKEND="${VLLM_ROCM_ATTN_BACKEND:-ROCM_AITER_FA}"
ATTN_BACKEND_FALLBACK="${VLLM_ROCM_ATTN_BACKEND_FALLBACK:-ROCM_AITER_UNIFIED_ATTN}"
export VLLM_ROCM_ATTN_BACKEND="${ATTN_BACKEND}"

# Kill any stale processes
pkill -f "lumenrl.trainer.main" 2>/dev/null || true
pkill -f "pt_elastic" 2>/dev/null || true
sleep 2

mkdir -p "${OUTPUT_DIR}"
mkdir -p /dev/shm/lumenrl_weight_sync
mkdir -p "${CKPT_DIR}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LumenRL 1A-Sync — BF16 On-Policy Training                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:    ${MODEL_DIR}/qwen3-8b-base"
echo "║  GPUs:     ${NUM_GPUS}"
echo "║  Mode:     RLTrainer (sync on-policy + ATOM rollout)       ║"
echo "║  Attn:     ${VLLM_ROCM_ATTN_BACKEND} (fallback: ${ATTN_BACKEND_FALLBACK})"
echo "║  Config:   ${CONFIG}"
echo "║  Log:      ${LOG_FILE}"
echo "║  Smoke:    ${SMOKE_TEST}   Dry-run: ${DRY_RUN}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

OVERRIDES=()
OVERRIDES+=("policy.model_name=${MODEL_DIR}/qwen3-8b-base")
OVERRIDES+=("reward.dataset=${DATA_DIR}/dapo-math-17k.parquet")
OVERRIDES+=("checkpointing.checkpoint_dir=${CKPT_DIR}")
if [ -n "${TOTAL_STEPS:-}" ]; then
    OVERRIDES+=("num_training_steps=${TOTAL_STEPS}")
fi

# ─── Smoke-test: tiny config for 2-step pipeline validation (~5-10 min) ──────
if [ "${SMOKE_TEST}" = true ]; then
    EXP_NAME="1a-bf16-sync-smoke"
    OUTPUT_DIR="${REPO_ROOT}/output/DAPO/${EXP_NAME}"
    LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"
    mkdir -p "${OUTPUT_DIR}"
    OVERRIDES+=("num_training_steps=2")
    OVERRIDES+=("policy.train_global_batch_size=16")
    OVERRIDES+=("policy.max_response_length=512")
    OVERRIDES+=("policy.max_total_sequence_length=1024")
    OVERRIDES+=("policy.max_token_len_per_gpu=1024")
    OVERRIDES+=("policy.generation.atom_cfg.max_model_len=1024")
    OVERRIDES+=("algorithm.dapo.num_generations=8")
    echo "*** SMOKE-TEST MODE: 2 steps, batch=16, max_resp=512 ***"
fi

# ─── Dry-run: skip ATOM, use mock generation for training-only debugging ─────
if [ "${DRY_RUN}" = true ]; then
    export LUMENRL_DRY_RUN=1
    echo "*** DRY-RUN MODE: ATOM disabled, using mock generation ***"
fi

MAX_RETRIES="${MAX_RETRIES:-50}"
RETRY_DELAY="${RETRY_DELAY:-10}"
_fell_back=false

for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo ""
    echo "=== Attempt ${attempt}/${MAX_RETRIES} ($(date)) ==="
    echo "    VLLM_ROCM_ATTN_BACKEND=${VLLM_ROCM_ATTN_BACKEND}"
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

    # If ROCM_AITER_FA crashed and we haven't fallen back yet, switch backend
    if [ "${_fell_back}" = false ] && [ "${VLLM_ROCM_ATTN_BACKEND}" = "ROCM_AITER_FA" ]; then
        if grep -q "fmha_fwd_v3\|undefined symbol.*aiter.*mha_fwd\|ROCM_AITER_FA.*fail" "${LOG_FILE}" 2>/dev/null; then
            echo "*** ROCM_AITER_FA backend failed — falling back to ${ATTN_BACKEND_FALLBACK} ***"
            export VLLM_ROCM_ATTN_BACKEND="${ATTN_BACKEND_FALLBACK}"
            _fell_back=true
        fi
    fi

    sleep "${RETRY_DELAY}"
done

echo "ERROR: Training failed after ${MAX_RETRIES} attempts." >&2
exit 1
