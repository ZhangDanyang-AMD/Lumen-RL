#!/usr/bin/env bash
# Launch Qwen3-8B RL training on miles with LumenRL plugin hooks.
#
# miles is a fork of vime with SGLang rollout and async orchestration.
# LumenRL hooks are injected via miles's --custom-*-path CLI arguments,
# with the same signatures as vime plus miles-specific FP8 reset hooks.
#
# Prerequisites:
#   - miles and lumenrl both installed
#   - Megatron-LM (ROCm fork) installed or on PYTHONPATH
#   - SGLang installed
#   - Model and data downloaded to $DATA_ROOT
#
# Usage:
#   DATA_ROOT=/data bash examples/plugin/miles/run.sh           # 3-step smoke
#   DATA_ROOT=/data STEPS=200 bash examples/plugin/miles/run.sh # long run
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing models/ and data_cached/}"
: "${MILES_ROOT:?Set MILES_ROOT to the miles repo directory}"

STEPS="${STEPS:-3}"
MODEL="${DATA_ROOT}/models/Qwen3-8B-Base"
TRAIN_DATA="${DATA_ROOT}/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet"
LOG="${LOG:-${DATA_ROOT}/logs/miles-plugin-qwen3-8b-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"

# Enable FP8 training (set to 0 to disable)
TRAIN_FP8="${TRAIN_FP8:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=7200

# FP8 environment variables (only when TRAIN_FP8=1)
if [ "$TRAIN_FP8" = "1" ]; then
    export LUMEN_FP8=1 FP8_PARAM_MANAGER=0 LUMEN_NORM=1
    export LUMEN_FP8_SCALING=blockwise2d LUMEN_FP8_FORMAT=fp8_e4m3
    export LUMEN_FP8_BLOCK_SIZE=128 LUMEN_FP8_ATTN=none
fi

# Verify hooks are importable
python3 -c "
from lumenrl.plugin.miles.hooks import (
    custom_megatron_init,
    custom_model_provider,
    generate_rollout,
    custom_megatron_before_train_step_hook,
    custom_megatron_before_log_prob_hook,
    custom_megatron_post_save_hook,
)
print('LumenRL miles hooks: all 6 hooks importable')
"

echo "=== miles + LumenRL plugin: Qwen3-8B Megatron + SGLang ==="
echo "    STEPS=$STEPS  TRAIN_FP8=$TRAIN_FP8  LOG=$LOG"

# LumenRL plugin hooks
PLUGIN_ARGS=(
    --custom-megatron-init-path
        lumenrl.plugin.miles.hooks.custom_megatron_init
    --custom-model-provider-path
        lumenrl.plugin.miles.hooks.custom_model_provider
    --custom-megatron-before-train-step-hook-path
        lumenrl.plugin.miles.hooks.custom_megatron_before_train_step_hook
    --custom-megatron-before-log-prob-hook-path
        lumenrl.plugin.miles.hooks.custom_megatron_before_log_prob_hook
    --custom-megatron-post-save-hook-path
        lumenrl.plugin.miles.hooks.custom_megatron_post_save_hook
)

cd "$MILES_ROOT"
python3 -u train.py \
    --hf-checkpoint "$MODEL" \
    --prompt-data "$TRAIN_DATA" \
    --input-key prompt \
    --num-rollout "$STEPS" \
    --rollout-batch-size 128 \
    --train-micro-batch-size 1 \
    --num-generation 16 \
    --max-response-len 2048 \
    --lr 1.0e-6 \
    --lr-warmup-iters 10 \
    --loss-type grpo \
    --update-weight-transfer-mode broadcast \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    "${PLUGIN_ARGS[@]}" \
    2>&1 | tee "$LOG"
