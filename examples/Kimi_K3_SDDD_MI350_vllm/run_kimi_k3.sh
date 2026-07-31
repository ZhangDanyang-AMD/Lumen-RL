#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K3 DSpark Draft Model Distillation (vLLM, 8-GPU Sequential) — MI350
#
# Sequential mode: K3 requires TP=8 for inference (~1TB MoE model), so we
# cannot use the 4+4 GPU split. Instead:
#   Phase A: All 8 GPUs run vLLM inference (TP=8, generate + extract hidden states)
#   Phase B: All 8 GPUs run FSDP2 training (DSpark draft model)
#
# DSpark architecture (Inferact/Kimi-K3-DSpark):
#   - 5-layer parallel backbone with MLA attention
#   - Markov head (vanilla, rank=256) for sequential correction
#   - Confidence head for acceptance prediction
#   - Loss: CE(0.1) + TV(0.9) + BCE(1.0)
#
# Usage:
#   bash examples/Kimi_K3_SDDD_MI350_vllm/run_kimi_k3.sh
#   bash examples/Kimi_K3_SDDD_MI350_vllm/run_kimi_k3.sh --smoke-test
#   MODEL_PATH=/path/to/Kimi-K3 bash examples/Kimi_K3_SDDD_MI350_vllm/run_kimi_k3.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXP_NAME="kimi-k3-dspark-vllm-mi350"
OUTPUT_DIR="${REPO_ROOT}/output/Kimi_K3_SDDD/LumenRL"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"

NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-8}"

MODEL_PATH="${MODEL_PATH:-/dev/shm/Kimi-K3}"
if [ "${SMOKE_TEST}" = true ]; then
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/kimi_k3_dspark_smoke_test}"
    CONFIG="${SCRIPT_DIR}/configs/smoke_test.yaml"
    echo ">>> SMOKE TEST: 5-step DSpark validation (vLLM TP=8, Kimi-K3)"
else
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/kimi_k3_dspark_vllm}"
    CONFIG="${SCRIPT_DIR}/configs/train.yaml"
fi

# Environment
export PYTHONUNBUFFERED=1
# Deliberately NOT expandable_segments:True. On ROCm those segments are never
# unmapped by empty_cache(), so the draft-training process keeps ~90 GiB of
# address space after _offload_draft_to_cpu() and vLLM refuses to restart at
# round 1 ("Free memory on device cuda:N is less than desired GPU memory
# utilization"). Peak usage is ~103 GiB out of 288, so fragmentation is not a
# concern here.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
export LUMENRL_LOG_LEVEL=INFO
export NCCL_TIMEOUT=7200
export GLOG_minloglevel="${GLOG_minloglevel:-3}"
export GLOG_v="${GLOG_v:-0}"
export GLOG_logtostderr="${GLOG_logtostderr:-1}"
export MOONCAKE_LOG_LEVEL="${MOONCAKE_LOG_LEVEL:-FATAL}"

mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"

# Clear pyc caches to pick up volume-mounted code changes
find /root/lumenrl -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Step 0a: Clean stale lock files and compile caches to prevent AITER JIT deadlocks
echo ">>> Cleaning stale lock files and compile caches..."
find /tmp -name '*.lock' -path '*aiter*' -delete 2>/dev/null || true
find /tmp -name '*.lock' -path '*hiprtc*' -delete 2>/dev/null || true
rm -rf /root/.cache/atom/* 2>/dev/null || true
rm -rf /root/aiter/aiter/jit/build 2>/dev/null || true
rm -rf /root/Lumen/third_party/aiter/aiter/jit/build 2>/dev/null || true
echo ">>> Lock and build cache cleanup done."

# Step 0b: Split dataset if not already done
if [ ! -f /dev/shm/kimi-mtp-dataset-phase1/train.jsonl ]; then
    echo ">>> Splitting kimi-mtp-dataset into Phase 1 and Phase 2..."
    python3 "${SCRIPT_DIR}/split_dataset.py"
fi

# Apply vLLM patches (volume-mounted, idempotent).
# Only for the 0.19.1 tree these patches were written against — applying them to
# another vLLM build lands stray hunks and leaves .rej files behind.
PATCH_DIR="${SCRIPT_DIR}/docker/patches/vllm/v0.19.1"
VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
if [ -d "${PATCH_DIR}" ] && [ "${VLLM_VERSION%%+*}" = "0.19.1" ]; then
    VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))" 2>/dev/null || echo "")
    if [ -n "${VLLM_DIR}" ]; then
        for p in "${PATCH_DIR}"/*.patch; do
            [ -f "$p" ] && (cd "$VLLM_DIR" && patch -p1 --forward < "$p" 2>/dev/null || true)
        done
    fi
else
    echo ">>> Skipping v0.19.1 vLLM patches (vllm=${VLLM_VERSION:-unknown})"
fi

cleanup_orphans() {
    pkill -9 -f "VLLM::Worker" 2>/dev/null || true
    pkill -9 -f "EngineCore"  2>/dev/null || true
    pkill -9 -f "AsyncLLMEngine" 2>/dev/null || true
    pkill -9 -f "mooncake_master" 2>/dev/null || true
}
trap cleanup_orphans EXIT

echo "═══════════════════════════════════════════════════════════════"
echo "  Kimi K3 DSpark Draft Model Distillation (vLLM) — MI350"
echo "  Teacher:     ${MODEL_PATH} (vLLM, TP=8, MXFP4 MoE)"
echo "  Draft:       DSpark (5-layer, Markov+confidence, BF16)"
echo "  Mode:        8-GPU sequential (infer → train)"
echo "  Train GPUs:  ${NUM_TRAIN_GPUS} GPUs (FSDP2+aiter)"
echo "  Transfer:    Cached to /dev/shm"
echo "  Config:      ${CONFIG}"
echo "  Output:      ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

OVERRIDES=(
    "policy.model_name=${MODEL_PATH}"
    "algorithm.teacher.model_name=${MODEL_PATH}"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
)

# Space-separated omegaconf dotlist entries, e.g.
#   EXTRA_OVERRIDES="algorithm.spec_distill.cache_dir=/dev/shm/tc num_training_steps=20"
if [ -n "${EXTRA_OVERRIDES:-}" ]; then
    read -r -a _extra <<< "${EXTRA_OVERRIDES}"
    OVERRIDES+=("${_extra[@]}")
fi
echo ">>> Overrides: ${OVERRIDES[*]}"

MOONCAKE_NOISE_RE='scoped_vlog_timer|MasterClient::(Ping|FetchTasks)|transfer_task\.cpp|client_service\.cpp|BatchGet completed|Transfer (completed|engine operation)|Setting transfer result'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" \
        -m lumenrl.trainer.main \
        --config "${CONFIG}" \
        ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
        2>&1 \
    | grep --line-buffered -v -E "^[IWEF][0-9]{4} [0-9:.]+\s+[0-9]+ \S+\.(cpp|cc|h):" \
    | grep --line-buffered -vE "${MOONCAKE_NOISE_RE}" \
    | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -eq 0 ] && grep -qE \
    '(Traceback \(most recent call last\)|HfHubHTTPError|Training failed|FAILED|exitcode\s*:\s*-[0-9]+|MEMORY_APERTURE_VIOLATION|out of memory)' \
    "${LOG_FILE}" 2>/dev/null; then
    echo ">>> Crash detected in log despite torchrun exit code 0." >&2
    EXIT_CODE=1
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Kimi K3 DSpark distillation (vLLM, MI350) completed successfully."
else
    echo ">>> Kimi K3 DSpark distillation (vLLM, MI350) failed with exit code ${EXIT_CODE}." >&2
fi
echo ">>> Log: ${LOG_FILE}"
exit ${EXIT_CODE}
