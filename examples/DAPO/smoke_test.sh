#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test: 2-step BF16 run to verify the DAPO pipeline works end-to-end.
# Run this BEFORE launching full experiments.
#
# Usage:
#   bash examples/DAPO/smoke_test.sh
#   MODEL_PATH=/path/to/model bash examples/DAPO/smoke_test.sh
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="LUMENRL-SMOKE"
export EXP_NAME="smoke-qwen3-8b-bf16"

export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"

# Minimal config for fast iteration
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
export ROLLOUT_QUANTIZATION="null"
export ROLLOUT_IS="null"
export FREE_CACHE_ENGINE="true"

source "${SCRIPT_DIR}/common.sh"
launch_training
