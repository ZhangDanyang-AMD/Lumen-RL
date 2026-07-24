#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3-30B-A3B DAPO RL — Megatron TP4/EP8 + ATOM BF16 (MI308X)
#
# MILES-aligned: GRPO 32x8, 8k response, global batch 256, no KL, TP4/EP8
# ATOM BF16 rollout + FP8 KV cache (per_block_fp8 causes garbled output)
#
# Usage:
#   MODE=smoke   STEPS=3    bash examples/GRPO/run_grpo.sh   # smoke test
#   MODE=longrun STEPS=1000 bash examples/GRPO/run_grpo.sh   # full training
#   MODE=longrun STEPS=30   bash examples/GRPO/run_grpo.sh   # 30-step health check
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
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
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/data_cached/qwen3-30b-a3b-maxprompt1024/dapo-math-17k.filtered.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/data_cached/qwen3-30b-a3b-maxprompt1024/aime-2024.filtered.parquet}"

RUN_ID="${RUN_ID:-qwen3-30b-a3b-ep8-${MODE}-$(date +%Y%m%d-%H%M%S)}"
LOG="${LOG:-$DATA_ROOT/logs/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-$DATA_ROOT/ckpts/qwen3-30b-a3b-ep8/${MODE}}"

cd "$LUMENRL_DIR"

# ═══════════════════════════════════════════════════════════════════════
# Environment variables (BF16 training + ATOM BF16 rollout)
# ═══════════════════════════════════════════════════════════════════════
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 HSA_DISABLE_FRAGMENT_ALLOCATOR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# Global NOSET: required for ATOM rollout IPC (custom all-reduce needs all
# GPUs visible). Training workers isolate to 1 GPU via base_worker.py
# module-level code (ray.get_accelerator_ids → CUDA_VISIBLE_DEVICES).
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb"
export LUMENRL_LOG_LEVEL=INFO MODEL_NAME="$MODEL_PATH"
export LUMENRL_DEBUG="${LUMENRL_DEBUG:-0}"
export LUMEN_DISABLE_HF_ATTN_PATCH=1

# ATOM rollout env vars
export AITER_LOG_LEVEL=WARNING
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1
export ATOM_REQUIRES_GRAD=0

# ATOM no-eager + level=3: enable dynamo, isolate torch compile cache per replica
export TORCHDYNAMO_DISABLE=0
export ATOM_ISOLATE_TORCH_COMPILE_CACHE=1
export ATOM_TORCH_COMPILE_CACHE_ROOT="${ATOM_TORCH_COMPILE_CACHE_ROOT:-/tmp/atom_torch_compile_cache}"

# ATOM BF16 rollout (disable AITER online quantization to avoid per_block_fp8 garbled output)
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_ROCM_USE_AITER_LINEAR=0

# vLLM v1
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN
export ATOM_DISABLE_VLLM_PLUGIN=1

# PYTHONPATH
export PYTHONPATH="$LUMENRL_DIR:$AITER_DIR:$LUMEN_DIR:$ATOM_DIR:${PYTHONPATH:-}"

# W&B key (optional: $RL_ROOT/wandb.key format WANDB_API_KEY=xxxx)
for _wandb_key in "$RL_ROOT/wandb.key" "$RL_ROOT/../wandb.key"; do
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$_wandb_key" ]; then
    export WANDB_API_KEY="$(cut -d= -f2- "$_wandb_key" | tr -d '[:space:]')"
  fi
done

# ═══════════════════════════════════════════════════════════════════════
# Config selection
# ═══════════════════════════════════════════════════════════════════════
BACKEND="${BACKEND:-atom}"  # atom or vllm
if [ "$MODE" = "longrun" ]; then
  CONFIG=examples/GRPO/configs/grpo_qwen3_30b_a3b_${BACKEND}_ep8_longrun.yaml
else
  CONFIG=examples/GRPO/configs/grpo_qwen3_30b_a3b_${BACKEND}_ep8_smoke.yaml
fi
CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"

echo "$LOG" > /tmp/run_grpo_log.txt
echo "═══════════════════════════════════════════════════════════════"
echo "  Qwen3-30B-A3B DAPO RL — Megatron TP4/EP8 + ATOM BF16"
echo "  MODE=$MODE  STEPS=$STEPS  CONFIG=$CONFIG"
echo "  MODEL=$MODEL_PATH"
echo "  CKPT=$CKPT_DIR  LOG=$LOG"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════
# Cleanup stale processes (ATOM / vLLM Ray server + orphan EngineCore)
# ═══════════════════════════════════════════════════════════════════════
ray stop --force >/dev/null 2>&1 || true
python3 - <<'PY'
import os, signal, subprocess
patterns = (
    "lumenrl.trainer.main", "VLLMRayServer", "ATOMRayServer",
    "VLLM::EngineCore", "EngineCore", "spawn_main",
    "torch/_inductor/compile_worker", "multiprocessing.resource_tracker",
)
skip = {os.getpid(), os.getppid()}
out = subprocess.check_output(["ps", "-eo", "pid,ppid,stat,cmd"], text=True)
for line in out.splitlines()[1:]:
    parts = line.strip().split(None, 3)
    if len(parts) < 4:
        continue
    pid, stat, cmd = int(parts[0]), parts[2], parts[3]
    if pid in skip or "Z" in stat:
        continue
    if any(p in cmd for p in patterns):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY
sleep 8

# ═══════════════════════════════════════════════════════════════════════
# Launch training
# ═══════════════════════════════════════════════════════════════════════
python3 -u -m lumenrl.trainer.main --config "$CONFIG" \
  policy.model_name="$MODEL_PATH" \
  reward.dataset="$TRAIN_FILE" \
  val_dataset="$VAL_FILE" \
  checkpointing.checkpoint_dir="$CKPT_DIR" \
  num_training_steps="$STEPS" \
  seed=10086 > "$LOG" 2>&1
EXIT_CODE=$?
echo "=== exit=$EXIT_CODE ==="
exit $EXIT_CODE
