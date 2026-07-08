#!/usr/bin/env bash
# 统一 DAPO 启动：MODE=bf16|fp8, TRAIN_FP8=0|1, STEPS=N。路径取容器内 $RL_ROOT/$DATA_ROOT。
set -uo pipefail
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"
MODE="${MODE:-bf16}"; TRAIN_FP8="${TRAIN_FP8:-0}"; STEPS="${STEPS:-1000}"
MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/Qwen3-8B-Base}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet}"
RUN_ID="${RUN_ID:-${MODE}$([ "$TRAIN_FP8" = 1 ] && echo -e2e)-ray-vllm-8b-$(date +%Y%m%d-%H%M%S)}"
LOG="${LOG:-$DATA_ROOT/logs/${RUN_ID}.log}"
cd "$RL_ROOT/Lumen-RL"

# ---- 通用 env（BF16/FP8 共用）----
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1 HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 HSA_DISABLE_FRAGMENT_ALLOCATOR=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN ATOM_DISABLE_VLLM_PLUGIN=1
export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export LUMEN_DISABLE_HF_ATTN_PATCH=1 MODEL_NAME="$MODEL_PATH"
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb" LUMENRL_LOG_LEVEL=INFO
export PYTHONPATH="$RL_ROOT/Lumen-RL:$RL_ROOT/aiter:$RL_ROOT/Lumen:${PYTHONPATH:-}"
[ -f "$RL_ROOT/wandb.key" ] && export WANDB_API_KEY="$(cut -d= -f2- "$RL_ROOT/wandb.key" | tr -d '[:space:]')"

if [ "$MODE" = "fp8" ]; then
  CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml
  # rollout per_block_fp8 + AITER unified attention
  export LUMENRL_FP8_PER_BLOCK=1
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_AITER_LINEAR=0
  if [ "$TRAIN_FP8" = "1" ]; then    # FP8 E2E 训练（blockwise2d，param manager 必须关）
    export LUMEN_FP8=1 FP8_PARAM_MANAGER=0 LUMEN_NORM=1
    export LUMEN_FP8_SCALING=blockwise2d LUMEN_FP8_FORMAT=fp8_e4m3 LUMEN_FP8_BLOCK_SIZE=128 LUMEN_FP8_ATTN=none
  fi
else
  CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_longrun.yaml
  export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_LINEAR=0
fi

echo "$LOG" > /tmp/run_dapo_log.txt
echo "=== MODE=$MODE TRAIN_FP8=$TRAIN_FP8 STEPS=$STEPS  CONFIG=$CONFIG  LOG=$LOG ==="

# 清理旧进程
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f lumenrl.trainer.main 2>/dev/null || true; pkill -9 -f VLLMRayServer 2>/dev/null || true; sleep 3

python3 -u -m lumenrl.trainer.main --config "$CONFIG" \
  policy.model_name="$MODEL_PATH" reward.dataset="$TRAIN_FILE" val_dataset="$VAL_FILE" \
  num_training_steps="$STEPS" seed=10086 > "$LOG" 2>&1
echo "=== exit=$? ==="
