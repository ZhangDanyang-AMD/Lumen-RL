#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test: 2-step FP8 E2E (blockwise) run to verify the full Lumen FP8
# stack works before launching full experiments.
#
# Usage:
#   bash examples/DAPO/smoke_test_fp8.sh
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="LUMENRL-SMOKE"
export EXP_NAME="smoke-qwen3-8b-fp8-blockwise"

export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"

# Minimal config
export TOTAL_STEPS="2"
export TEST_FREQ="1"
export SAVE_FREQ="999"
export TRAIN_BSZ="16"
export GEN_BSZ="16"
export MINI_BSZ="16"
export N_RESP="2"
export MAX_RESPONSE_LENGTH="512"
export GPU_MEM_UTIL="0.5"
export MAX_NUM_SEQS="64"
export FREE_CACHE_ENGINE="true"

# FP8 rollout + TIS
export ROLLOUT_QUANTIZATION="fp8"
export ROLLOUT_IS="token"
export ROLLOUT_IS_THRESHOLD="2.0"

# FP8 E2E training (blockwise)
export FP8_PARAM_MANAGER="1"
export LUMEN_FP8="1"
export LUMEN_NORM="1"
export LUMEN_FP8_SCALING="blockwise"
export LUMEN_FP8_FORMAT="fp8_e4m3"
export LUMEN_FP8_BLOCK_SIZE="128"

source "${SCRIPT_DIR}/common.sh"
launch_training
