#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1A: Qwen3-8B-Base — BF16 training + BF16 rollout (baseline)
#
# Reference: FP8_TRAINING_ALIGNMENT_PLAN.md §Experiment 1
# This is the pure BF16 baseline against which all FP8 variants are compared.
#
# Config aligned to VERL FP8 docs (Qwen3-8B-Base, 8×MI350X):
#   train_bsz=32, gen_bsz=32, mini_bsz=32, n=16, max_resp=20K
#   rollout_tp=1, sp_size=1
#
# ROCm adaptations:
#   gpu_memory_utilization=0.3 (vLLM sleep() is no-op on ROCm)
#   free_cache_engine=True (explicitly frees KV cache between phases)
#   max_num_seqs=64 (prevents KV cache OOM with long sequences)
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="LUMENRL-FP8-ALIGN"
export EXP_NAME="1A-qwen3-8b-bf16-baseline"

export MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"

# ─── Rollout: BF16, no quantization, no TIS ──────────────────────────────────
export ROLLOUT_QUANTIZATION="null"
export ROLLOUT_IS="null"

# ─── Training: BF16 (no Lumen FP8) ───────────────────────────────────────────
export FP8_PARAM_MANAGER="0"
export LUMEN_FP8="0"
export LUMEN_NORM="0"

# ─── Batch sizes (matching plan §Experiment 1 Config) ────────────────────────
export TRAIN_BSZ="32"
export GEN_BSZ="32"
export MINI_BSZ="32"
export N_RESP="16"

# ─── ROCm memory adaptations ─────────────────────────────────────────────────
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
export MAX_NUM_SEQS="64"
export FREE_CACHE_ENGINE="true"
export OFFLOAD="true"
export ROLLOUT_TP="1"
export SP_SIZE="1"

# ─── Token length limits ─────────────────────────────────────────────────────
export PPO_MAX_TOKEN_LEN="21504"
export LOG_PROB_MAX_TOKEN_LEN="21504"

# ─── Steps ────────────────────────────────────────────────────────────────────
export TOTAL_STEPS="${TOTAL_STEPS:-275}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-20}"

# Increase weight sync bucket for unsharded embed_tokens with TP=1
export EXTRA_OVERRIDES="actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2560"

source "${SCRIPT_DIR}/common.sh"
launch_training
