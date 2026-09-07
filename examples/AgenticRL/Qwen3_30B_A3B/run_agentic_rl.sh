#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3-30B-A3B Agentic RL — GSPO + GEAK sandbox + TRLOO
#
# AMD-specific kernel agent training for HIP/Triton/FlyDSL kernel optimisation.
#
# Usage:
#   MODE=smoke   STEPS=3    bash examples/AgenticRL/Qwen3_30B_A3B/run_agentic_rl.sh
#   MODE=longrun STEPS=200  bash examples/AgenticRL/Qwen3_30B_A3B/run_agentic_rl.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
ulimit -n 524288 2>/dev/null || true
: "${RL_ROOT:?需要设置 RL_ROOT}"; : "${DATA_ROOT:?需要设置 DATA_ROOT}"

MODE="${MODE:-smoke}"; STEPS="${STEPS:-3}"
LUMENRL_DIR="${LUMENRL_DIR:-$RL_ROOT/Lumen-RL}"
ATOM_DIR="${ATOM_DIR:-$RL_ROOT/ATOM}"
AITER_DIR="${AITER_DIR:-$RL_ROOT/aiter-lumen}"
LUMEN_DIR="${LUMEN_DIR:-$RL_ROOT/Lumen}"

# Repository fallback
if [ ! -f "$LUMEN_DIR/lumen/config.py" ]; then
  LUMEN_DIR="$LUMENRL_DIR/third_party/Lumen"
fi
if [ ! -d "$AITER_DIR/aiter" ]; then
  AITER_DIR="$LUMENRL_DIR/third_party/aiter"
fi
if [ ! -f "$ATOM_DIR/atom/rollout/async_engine.py" ]; then
  ATOM_DIR="$LUMENRL_DIR/third_party/ATOM"
fi

MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/Qwen3-30B-A3B}"
GEAK_ROOT="${GEAK_ROOT:-$DATA_ROOT/geak}"
CASES_PATH="${CASES_PATH:-$GEAK_ROOT/cases/phase1-pilot-gfx942-generation.yaml}"

RUN_ID="${RUN_ID:-qwen3-30b-a3b-agentic-${MODE}-$(date +%Y%m%d-%H%M%S)}"
LOG="${LOG:-$DATA_ROOT/logs/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-$DATA_ROOT/ckpts/qwen3-30b-a3b-agentic/${MODE}}"

cd "$LUMENRL_DIR"

# ═══════════════════════════════════════════════════════════════════════
# Environment variables
# ═══════════════════════════════════════════════════════════════════════
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-/opt/rocm/lib}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens11np0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-0}"
export HSA_DISABLE_FRAGMENT_ALLOCATOR="${HSA_DISABLE_FRAGMENT_ALLOCATOR:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export LUMENRL_LOG_LEVEL=INFO MODEL_NAME="$MODEL_PATH"
export LUMEN_DISABLE_HF_ATTN_PATCH=1

# ATOM rollout env vars
export AITER_LOG_LEVEL=WARNING
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1
export ATOM_REQUIRES_GRAD=0
export TORCHDYNAMO_DISABLE=0
export ATOM_ISOLATE_TORCH_COMPILE_CACHE=1
export ATOM_TORCH_COMPILE_CACHE_ROOT="${ATOM_TORCH_COMPILE_CACHE_ROOT:-/tmp/atom_torch_compile_cache}"

# ATOM BF16 rollout
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_ROCM_USE_AITER_LINEAR=0
export USE_ROCM_AITER_ROPE_BACKEND=0

# vLLM v1
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN
export ATOM_DISABLE_VLLM_PLUGIN=1

# PYTHONPATH — include multi-tune-agent for GEAK gym
export PYTHONPATH="$LUMENRL_DIR:$LUMENRL_DIR/experiments/multi-tune-agent/src:$LUMENRL_DIR/experiments/multi-tune-agent:$AITER_DIR:$LUMEN_DIR:$ATOM_DIR:${PYTHONPATH:-}"

# W&B key
for _wandb_key in "$RL_ROOT/wandb.key" "$RL_ROOT/../wandb.key"; do
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$_wandb_key" ]; then
    export WANDB_API_KEY="$(cut -d= -f2- "$_wandb_key" | tr -d '[:space:]')"
  fi
done

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
CONFIG=examples/AgenticRL/Qwen3_30B_A3B/gspo_qwen3_30b_a3b_geak_smoke.yaml
CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"

mkdir -p "$(dirname "$LOG")"
echo "═══════════════════════════════════════════════════════════════"
echo "  Qwen3-30B-A3B Agentic RL — GSPO + GEAK + TRLOO"
echo "  MODE=$MODE  STEPS=$STEPS  CONFIG=$CONFIG"
echo "  MODEL=$MODEL_PATH"
echo "  GEAK_ROOT=$GEAK_ROOT"
echo "  CKPT=$CKPT_DIR  LOG=$LOG"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════
# Cleanup stale processes
# ═══════════════════════════════════════════════════════════════════════
if [ "${LUMENRL_KEEP_RAY_CLUSTER:-0}" = "1" ]; then
  echo "[agentic_rl] LUMENRL_KEEP_RAY_CLUSTER=1 -> preserve existing Ray cluster"
else
  ray stop --force >/dev/null 2>&1 || true
fi

# ═══════════════════════════════════════════════════════════════════════
# Launch training
# ═══════════════════════════════════════════════════════════════════════
python3 -u -m lumenrl.trainer.main --config "$CONFIG" \
  policy.model_name="$MODEL_PATH" \
  agentic_rl.geak_root="$GEAK_ROOT" \
  agentic_rl.cases_path="$CASES_PATH" \
  checkpointing.checkpoint_dir="$CKPT_DIR" \
  num_training_steps="$STEPS" \
  seed=10086 > "$LOG" 2>&1
EXIT_CODE=$?
echo "=== exit=$EXIT_CODE ==="
exit $EXIT_CODE
