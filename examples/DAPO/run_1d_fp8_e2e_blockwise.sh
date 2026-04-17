#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1D: Qwen3-8B-Base — FP8 E2E (blockwise 128) + FP8 rollout + TIS
#
# Reference: FP8_TRAINING_ALIGNMENT_PLAN.md §Experiment 1
# Full Lumen FP8 E2E training with blockwise scaling (128-element blocks),
# matching the FP8-RL paper's TE blockwise approach.
#
# This is the PRIORITY 1 FP8 E2E run — closest match to the FP8 reference.
# Expected: 1D ≈ 1A (FP8 training aligns with BF16).
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="LUMENRL-FP8-ALIGN"
export EXP_NAME="1D-qwen3-8b-fp8-e2e-blockwise"

export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"

# ─── Rollout: FP8 with TIS correction ────────────────────────────────────────
export ROLLOUT_QUANTIZATION="fp8"
export ROLLOUT_IS="token"
export ROLLOUT_IS_THRESHOLD="2.0"

# ─── Training: Lumen FP8 E2E (blockwise scaling) ────────────────────────────
export FP8_PARAM_MANAGER="1"
export LUMEN_FP8="1"
export LUMEN_NORM="1"
export LUMEN_FP8_SCALING="blockwise"
export LUMEN_FP8_FORMAT="fp8_e4m3"
export LUMEN_FP8_BLOCK_SIZE="128"

# ─── Batch sizes ─────────────────────────────────────────────────────────────
export TRAIN_BSZ="32"
export GEN_BSZ="32"
export MINI_BSZ="32"
export N_RESP="16"

# ─── ROCm adaptations ────────────────────────────────────────────────────────
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
export MAX_NUM_SEQS="64"
export FREE_CACHE_ENGINE="true"
export OFFLOAD="true"
export ROLLOUT_TP="1"
export SP_SIZE="1"
export PPO_MAX_TOKEN_LEN="21504"
export LOG_PROB_MAX_TOKEN_LEN="21504"
export TOTAL_STEPS="${TOTAL_STEPS:-275}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-20}"

export EXTRA_OVERRIDES="actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2560"

source "${SCRIPT_DIR}/common.sh"
launch_training
