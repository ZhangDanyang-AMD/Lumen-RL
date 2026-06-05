#!/usr/bin/env bash
# Experiment 1A — BF16 Baseline (Native LumenRL Trainer)
# No VERL dependency. Uses lumenrl.trainer.main entry point.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ─── Paths ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/dev/shm/model/qwen3-8b-base}"
CONFIG="${SCRIPT_DIR}/configs/1a_bf16_baseline.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/DAPO/1a-bf16-baseline}"

# ─── Prerequisite checks ──────────────────────────────────────────
errors=0
[ ! -d "${MODEL_PATH}" ] && echo "ERROR: Model not found: ${MODEL_PATH}" >&2 && errors=1
[ ! -f "${CONFIG}" ]     && echo "ERROR: Config not found: ${CONFIG}" >&2 && errors=1
python3 -c "import lumenrl" 2>/dev/null || { echo "ERROR: lumenrl not importable" >&2; errors=1; }
[ "${errors}" -ne 0 ] && exit 1

mkdir -p "${OUTPUT_DIR}"

# ─── Environment ──────────────────────────────────────────────────
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LUMENRL_LOG_LEVEL=INFO

# ─── Banner ───────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LumenRL Native — Exp 1A: BF16 Baseline                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:   ${MODEL_PATH}"
echo "║  Config:  ${CONFIG}"
echo "║  Output:  ${OUTPUT_DIR}"
echo "║  Entry:   lumenrl.trainer.main"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Launch ───────────────────────────────────────────────────────
python3 -m lumenrl.trainer.main \
    --config "${CONFIG}" \
    policy.model_name="${MODEL_PATH}" \
    checkpointing.checkpoint_dir="${OUTPUT_DIR}/ckpts" \
    ${EXTRA_OVERRIDES:-} \
    2>&1 | tee "${OUTPUT_DIR}/1a-bf16-baseline.log"
