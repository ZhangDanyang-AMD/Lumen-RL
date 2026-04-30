#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K2.5 Eagle3 Two-Phase Training (lightseekorg recipe)
#
# Phase 1: Foundation — perfectblend subset (296K samples, 20K steps)
# Phase 2: Mixed Domain — VL, Chinese, tool-call, agent, writing (181K, 20K steps)
#
# GPU split (8x GPU):
#   GPUs 0-3: torchrun FSDP2 draft model training (BF16)
#   GPUs 4-7: SGLang + ATOM teacher inference (TP=4, MXFP4)
#
# Usage:
#   bash examples/Kimi_K25_SDDD/run_full_training.sh
#   bash examples/Kimi_K25_SDDD/run_full_training.sh --phase2-only
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

PHASE2_ONLY=false
for arg in "$@"; do
    case "${arg}" in
        --phase2-only) PHASE2_ONLY=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LUMENRL_LOG_LEVEL=INFO
export NCCL_TIMEOUT=7200
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1

ATOM_REPO="${ATOM_REPO:-/home/danyzhan/ATOM_repo}"
if [ -d "${ATOM_REPO}" ]; then
    export PYTHONPATH="${ATOM_REPO}:${PYTHONPATH:-}"
fi

# Step 0: Split dataset if not already done
if [ ! -f /datasets/kimi-mtp-dataset-phase1/train.jsonl ]; then
    echo ">>> Splitting kimi-mtp-dataset into Phase 1 and Phase 2..."
    python3 "${SCRIPT_DIR}/split_dataset.py"
fi

# Phase 1: Foundation
if [ "${PHASE2_ONLY}" = false ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Phase 1: Foundation Training (perfectblend, 20K steps)"
    echo "  Teacher: /datasets/Kimi-K2.5-BF16 (SGLang+ATOM, TP=4)"
    echo "  Draft:   Eagle3, 1-layer Transformer, from scratch"
    echo "═══════════════════════════════════════════════════════════════"

    PHASE1_LOG="/datasets/checkpoints/kimi_k25_eagle3_phase1/phase1.log"
    mkdir -p /datasets/checkpoints/kimi_k25_eagle3_phase1

    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
        torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" \
            -m lumenrl.trainer.main \
            --config "${SCRIPT_DIR}/configs/phase1_foundation.yaml" \
            2>&1 | tee "${PHASE1_LOG}"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo ">>> Phase 1 FAILED." >&2
        exit 1
    fi
    echo ">>> Phase 1 completed."
fi

# Phase 2: Mixed Domain
echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 2: Mixed Domain Training (VL+CN+tool+agent, 20K steps)"
echo "  Teacher: /datasets/Kimi-K2.5-BF16 (SGLang+ATOM, TP=4)"
echo "  Draft:   Eagle3, resume from Phase 1 checkpoint"
echo "═══════════════════════════════════════════════════════════════"

PHASE2_LOG="/datasets/checkpoints/kimi_k25_eagle3_phase2/phase2.log"
mkdir -p /datasets/checkpoints/kimi_k25_eagle3_phase2

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" \
        -m lumenrl.trainer.main \
        --config "${SCRIPT_DIR}/configs/phase2_mixed.yaml" \
        2>&1 | tee "${PHASE2_LOG}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ">>> Phase 2 FAILED." >&2
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Two-phase training completed!"
echo "  Phase 1 checkpoint: /datasets/checkpoints/kimi_k25_eagle3_phase1"
echo "  Phase 2 checkpoint: /datasets/checkpoints/kimi_k25_eagle3_phase2"
echo "═══════════════════════════════════════════════════════════════"
