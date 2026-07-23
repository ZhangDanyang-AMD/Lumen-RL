#!/usr/bin/env bash
# 统一 DAPO 启动：MODE=bf16|fp8|atomfp8, TRAIN_FP8=0|1, STEPS=N。路径取容器内 $RL_ROOT/$DATA_ROOT。
set -uo pipefail
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"
MODE="${MODE:-bf16}"; TRAIN_FP8="${TRAIN_FP8:-0}"; STEPS="${STEPS:-1000}"
MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/Qwen3-8B-Base}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet}"
RUN_ID="${RUN_ID:-${MODE}$([ "$TRAIN_FP8" = 1 ] && echo -e2e)-ray-vllm-8b-$(date +%Y%m%d-%H%M%S)}"
LOG="${LOG:-$DATA_ROOT/logs/${RUN_ID}.log}"
USER_PYTHONPATH="${PYTHONPATH:-}"
LUMEN_DIR="${LUMEN_DIR:-$RL_ROOT/Lumen}"
if [ ! -f "$LUMEN_DIR/lumen/config.py" ]; then
  LUMEN_DIR="$RL_ROOT/Lumen-RL/third_party/Lumen"
fi
if [ ! -f "$LUMEN_DIR/lumen/config.py" ]; then
  LUMEN_DIR="$RL_ROOT/../rl_base/Lumen"
fi
AITER_DIR="${AITER_DIR:-$RL_ROOT/aiter}"
if [ ! -d "$AITER_DIR/aiter" ]; then
  AITER_DIR="$RL_ROOT/Lumen-RL/third_party/aiter"
fi
if [ ! -d "$AITER_DIR/aiter" ]; then
  AITER_DIR="$RL_ROOT/../rl_base/aiter"
fi
ATOM_DIR="${ATOM_DIR:-$RL_ROOT/ATOM}"
if [ ! -f "$ATOM_DIR/atom/rollout/async_engine.py" ]; then
  ATOM_DIR="$RL_ROOT/../rl_base/ATOM"
fi
if [ ! -f "$ATOM_DIR/atom/rollout/async_engine.py" ]; then
  ATOM_DIR="$RL_ROOT/Lumen-RL/third_party/ATOM"
fi
cd "$RL_ROOT/Lumen-RL"

# ---- 通用 env（BF16/FP8 共用）----
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1 HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 HSA_DISABLE_FRAGMENT_ALLOCATOR=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN ATOM_DISABLE_VLLM_PLUGIN=1
export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export LUMEN_DISABLE_HF_ATTN_PATCH=1 MODEL_NAME="$MODEL_PATH"
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb" LUMENRL_LOG_LEVEL=INFO
export PYTHONPATH="$RL_ROOT/Lumen-RL:$AITER_DIR:$LUMEN_DIR:$ATOM_DIR:${PYTHONPATH:-}"
for _wandb_key in "$RL_ROOT/wandb.key" "$RL_ROOT/../wandb.key"; do
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$_wandb_key" ]; then
    export WANDB_API_KEY="$(cut -d= -f2- "$_wandb_key" | tr -d '[:space:]')"
  fi
done

EXTRA_ARGS=()
if [ "$MODE" = "atomfp8" ] || [ "$MODE" = "atom_fp8" ]; then
  if [ "${ATOM_DEBUG:-0}" = "1" ]; then
    CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_debug.yaml
  else
    CONFIG=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_longrun.yaml
  fi
  ATOM_ONLINE_QUANT="${ATOM_ONLINE_QUANT:-per_block_fp8}"
  unset ATOM_DISABLE_VLLM_PLUGIN
  export LUMENRL_ATOM_AITER_SRC="${LUMENRL_ATOM_AITER_SRC:-$AITER_DIR}"
  export PYTHONPATH="$RL_ROOT/Lumen-RL/examples/DAPO/atom_aiter_shim:$RL_ROOT/Lumen-RL:$AITER_DIR:$LUMEN_DIR:$ATOM_DIR:$USER_PYTHONPATH"
  # ATOM FP8 正式方案：no-eager + compilation level=3，需开启 dynamo，且每个 colocated
  # replica 用独立 torch compile cache，避免 8 个 rank0 并发写同一路径触发 Inductor rename race。
  export TORCHDYNAMO_DISABLE=0 ATOM_ISOLATE_TORCH_COMPILE_CACHE=1
  export ATOM_TORCH_COMPILE_CACHE_ROOT="${ATOM_TORCH_COMPILE_CACHE_ROOT:-/tmp/atom_torch_compile_cache}"
  export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 VLLM_ROCM_USE_AITER_LINEAR=0
  # Match the vLLM fp8 training-side configuration exactly: standard Lumen FP8
  # blockwise2d linear + norm, no HF attention patch and no rollout-specific
  # early-return path. Rollout is ATOM; actor training stays on the vLLM fp8 path.
  unset LUMEN_ROLLOUT
  export LUMEN_DISABLE_HF_ATTN_PATCH=1 LUMEN_NORM=1
  if [ "$TRAIN_FP8" = "1" ]; then
    export LUMEN_FP8=1 FP8_PARAM_MANAGER=0
    export LUMEN_FP8_SCALING=blockwise2d LUMEN_FP8_FORMAT=fp8_e4m3 LUMEN_FP8_BLOCK_SIZE=128
    export LUMEN_FP8_ATTN=none LUMEN_FP8_QUANT_TYPE=blockwise LUMEN_ATTN_BACKEND=auto
    export LUMEN_FP8_WGRAD="${LUMEN_FP8_WGRAD:-0}"
  fi
  # no-eager level=3 正式方案：enforce_eager=false + compilation_config.level=3 + sleep2
  # （sleep_mode 训练前释放 rollout KV/weights/CUDA graph，避免 backward OOM）。
  EXTRA_ARGS+=(
    policy.generation.atom_cfg.online_quant_config.global_quant_config="$ATOM_ONLINE_QUANT"
    policy.generation.vllm_cfg.enforce_eager=false
    policy.generation.atom_cfg.engine_kwargs.enforce_eager=false
    policy.generation.atom_cfg.engine_kwargs.compilation_config.level=3
    policy.generation.vllm_cfg.enable_sleep_mode=true
    policy.generation.vllm_cfg.sleep_level=2
  )
elif [ "$MODE" = "fp8" ]; then
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
CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"

echo "$LOG" > /tmp/run_dapo_log.txt
echo "=== MODE=$MODE TRAIN_FP8=$TRAIN_FP8 STEPS=$STEPS  CONFIG=$CONFIG  LOG=$LOG ==="

# 清理旧进程
ray stop --force >/dev/null 2>&1 || true
python3 - <<'PY'
import os
import signal
import subprocess

patterns = (
    "lumenrl.trainer.main",
    "VLLMRayServer",
    "ATOMRayServer",
    "VLLM::EngineCore",
    "EngineCore",
    "spawn_main",
    "multiprocessing.resource_tracker",
)
skip = {os.getpid(), os.getppid()}
out = subprocess.check_output(["ps", "-eo", "pid,ppid,stat,cmd"], text=True)
for line in out.splitlines()[1:]:
    parts = line.strip().split(None, 3)
    if len(parts) < 4:
        continue
    pid = int(parts[0])
    stat = parts[2]
    cmd = parts[3]
    if pid in skip or "Z" in stat:
        continue
    if any(p in cmd for p in patterns):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY
sleep 8

python3 -u -m lumenrl.trainer.main --config "$CONFIG" \
  policy.model_name="$MODEL_PATH" reward.dataset="$TRAIN_FILE" val_dataset="$VAL_FILE" \
  num_training_steps="$STEPS" seed=10086 "${EXTRA_ARGS[@]}" ${EXTRA_OVERRIDE:-} > "$LOG" 2>&1
echo "=== exit=$? ==="
