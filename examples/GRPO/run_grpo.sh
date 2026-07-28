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
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-/opt/rocm/lib}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
# Cross-node weight broadcast uses the RoCE path validated in IB-RDMA-Test:
# node1/node2 ens11np0, mlx5_0, GID 3. RCCL still owns the transport; no MILES
# package is imported at runtime.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens11np0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export HIP_FORCE_DEV_KERNARG=1
# Long runs must allow ROCm to reclaim kernel scratch/events. Forcing
# HSA_NO_SCRATCH_RECLAIM=1 eventually exhausts HSA queue resources even when
# VRAM is available (observed after 15 full training steps on MI308X).
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-0}"
# This is a memory-fault debugging switch, not a fragmentation mitigation.
# Disabling ROCr's fragment cache increases allocation/event churn in long runs.
export HSA_DISABLE_FRAGMENT_ALLOCATOR="${HSA_DISABLE_FRAGMENT_ALLOCATOR:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# Global NOSET: required for ATOM rollout IPC (custom all-reduce needs all
# GPUs visible). Training workers isolate to 1 GPU via base_worker.py
# module-level code (ray.get_accelerator_ids → CUDA_VISIBLE_DEVICES).
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
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
# Apex selects AITER fused RoPE independently of vLLM and warns that it has
# lower precision. Keep Megatron's RoPE on the standard BF16 path so R3 only
# needs to reconcile expert routing, not a second attention-position mismatch.
export USE_ROCM_AITER_ROPE_BACKEND=0

# vLLM v1
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=WARN
export ATOM_DISABLE_VLLM_PLUGIN=1

# PYTHONPATH
export PYTHONPATH="$LUMENRL_DIR:$AITER_DIR:$LUMEN_DIR:$ATOM_DIR:${PYTHONPATH:-}"

# The cached ROCm flash-attn wheel contains the older 21-argument native
# varlen ABI while its Python wrapper includes the newer trailing num_splits.
# Apply the idempotent compatibility fix before Ray workers import it.
python3 - <<'PY'
from pathlib import Path

path = Path("/opt/venv/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py")
if path.exists():
    source = path.read_text()
    old = "        None,\n        num_splits,\n    )"
    if old in source:
        path.write_text(source.replace(old, "        None,\n    )", 1))
        print("[run_grpo] patched flash-attn ROCm varlen ABI")
PY

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
if [ "${LUMENRL_KEEP_RAY_CLUSTER:-0}" = "1" ]; then
  echo "[run_grpo] LUMENRL_KEEP_RAY_CLUSTER=1 -> preserve existing Ray cluster"
else
  ray stop --force >/dev/null 2>&1 || true
fi
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
EXTRA_OVERRIDES=()
if [ -n "${WANDB_RUN_NAME:-}" ]; then
  EXTRA_OVERRIDES+=("logger.wandb.name=$WANDB_RUN_NAME")
fi
if [ -n "${RESUME_OVERRIDE:-}" ]; then
  EXTRA_OVERRIDES+=("checkpointing.resume=$RESUME_OVERRIDE")
fi
if [ -n "${WEIGHT_SYNC_BACKEND:-}" ]; then
  EXTRA_OVERRIDES+=("weight_sync.backend=$WEIGHT_SYNC_BACKEND")
fi
python3 -u -m lumenrl.trainer.main --config "$CONFIG" \
  policy.model_name="$MODEL_PATH" \
  reward.dataset="$TRAIN_FILE" \
  val_dataset="$VAL_FILE" \
  checkpointing.checkpoint_dir="$CKPT_DIR" \
  num_training_steps="$STEPS" \
  seed=10086 \
  "${EXTRA_OVERRIDES[@]}" > "$LOG" 2>&1
EXIT_CODE=$?
echo "=== exit=$EXIT_CODE ==="
exit $EXIT_CODE
