#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K3 DSpark Draft Model Distillation (ATOM teacher, 8-GPU Sequential) — MI350
#
# K3 needs TP=8 for inference (~1.5 TB MoE), so the 4+4 GPU split is impossible:
#   Phase A1: all 8 GPUs decode on-policy responses (ATOM, capture parked)
#   Phase A2: same engine re-prefills the full sequences and ships hidden states
#   Phase B:  all 8 GPUs train the DSpark draft (FSDP2)
#
# Unlike the vLLM variant, A1 and A2 share one engine: ATOM captures through
# forward hooks, and whether a request is captured is decided by whether it
# carries an external id, so the two sweeps differ only in what they submit --
# nothing switches and nothing reloads 1.5 TB of weights. vLLM was abandoned
# here because its K3 decode path faults the GPU every 15-20 batches at B=64,
# well short of a 50-batch round.
#
# DSpark architecture (Inferact/Kimi-K3-DSpark):
#   - 5-layer parallel backbone with MLA attention
#   - Markov head (vanilla, rank=256) for sequential correction
#   - Confidence head for acceptance prediction
#   - Loss: CE(0.1) + TV(0.9) + BCE(1.0)
#
# Usage:
#   bash examples/Kimi_K3_SDDD_MI350_ATOM/run_kimi_k3.sh
#   bash examples/Kimi_K3_SDDD_MI350_ATOM/run_kimi_k3.sh --smoke-test
#   MODEL_PATH=/path/to/Kimi-K3 bash examples/Kimi_K3_SDDD_MI350_ATOM/run_kimi_k3.sh
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

EXP_NAME="kimi-k3-dspark-atom-mi350"
OUTPUT_DIR="${REPO_ROOT}/output/Kimi_K3_SDDD/LumenRL"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}.log"

NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-8}"

MODEL_PATH="${MODEL_PATH:-/dev/shm/Kimi-K3}"
if [ "${SMOKE_TEST}" = true ]; then
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/kimi_k3_dspark_atom_smoke}"
    CONFIG="${SCRIPT_DIR}/configs/smoke_test.yaml"
    echo ">>> SMOKE TEST: 5-step DSpark validation (ATOM TP=8, Kimi-K3)"
else
    CKPT_DIR="${CKPT_DIR:-/dev/shm/checkpoints/kimi_k3_dspark_atom}"
    CONFIG="${SCRIPT_DIR}/configs/train.yaml"
fi

# Environment
export PYTHONUNBUFFERED=1
# Deliberately NOT expandable_segments:True. On ROCm those segments are never
# unmapped by empty_cache(), so the draft-training process keeps ~90 GiB of
# address space after _offload_draft_to_cpu(), starving the teacher. Peak usage
# is ~103 GiB out of 288, so fragmentation is not a concern here.
#
# Unset rather than defaulted: "${VAR:-}" inherits, and inheriting it once cost
# a day of round transitions failing with the teacher unable to find 253 GiB.
unset PYTORCH_CUDA_ALLOC_CONF
export LUMENRL_LOG_LEVEL=INFO
export NCCL_TIMEOUT=7200
export GLOG_minloglevel="${GLOG_minloglevel:-3}"
export GLOG_v="${GLOG_v:-0}"
export GLOG_logtostderr="${GLOG_logtostderr:-1}"
export MOONCAKE_LOG_LEVEL="${MOONCAKE_LOG_LEVEL:-FATAL}"
# A decode sweep runs for minutes, far longer than a control message, so it has
# its own budget. Deliberately NOT defaulted here: this variable is an outright
# override of the engine's budget, and the engine derives that budget from the
# work the sweep was handed (prompts x max_tokens). A fixed 3600 was safe when
# each sweep held one 64-prompt batch, but a merged whole-round sweep runs ~36
# minutes, which 3600 clears by only 1.7x -- one slow round and the run is killed
# mid-sweep. Set it only to override deliberately.
if [ -n "${LUMENRL_TEACHER_GENERATE_TIMEOUT_SECONDS:-}" ]; then
    export LUMENRL_TEACHER_GENERATE_TIMEOUT_SECONDS
fi

mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"

# Clear pyc caches to pick up volume-mounted code changes
find /root/lumenrl -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Stale JIT locks deadlock aiter's first compile. Note aiter lives at
# /app/aiter-test in the ATOM base image, not /root/aiter as in the vLLM one.
echo ">>> Cleaning stale lock files and compile caches..."
find /tmp -name '*.lock' -path '*aiter*' -delete 2>/dev/null || true
find /tmp -name '*.lock' -path '*hiprtc*' -delete 2>/dev/null || true
rm -rf /root/.cache/atom/* 2>/dev/null || true
echo ">>> Lock and build cache cleanup done."

# Catch ATOM API drift and config typos in seconds, rather than 20 minutes into
# a weight load. Needs no GPU.
echo ">>> Running ATOM preflight checks..."
if ! python3 "${SCRIPT_DIR}/selfcheck/preflight.py"; then
    echo ">>> Preflight failed; not launching. See the FAIL lines above." >&2
    exit 1
fi

cleanup_orphans() {
    # ATOM spawns one model-runner process per TP rank via AsyncIOProcManager;
    # they outlive a killed parent and would hold the GPUs against the next run.
    pkill -9 -f "AsyncLLMEngine" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "atom_teacher" 2>/dev/null || true
    pkill -9 -f "mooncake_master" 2>/dev/null || true
}
trap cleanup_orphans EXIT

echo "═══════════════════════════════════════════════════════════════"
echo "  Kimi K3 DSpark Draft Model Distillation (ATOM) — MI350"
echo "  Teacher:     ${MODEL_PATH} (ATOM, TP=8)"
echo "  Draft:       DSpark (5-layer, Markov+confidence, BF16)"
echo "  Mode:        8-GPU sequential (decode → extract → train)"
echo "  Train GPUs:  ${NUM_TRAIN_GPUS} GPUs (FSDP2+aiter)"
echo "  Transfer:    Mooncake TCP → /dev/shm cache"
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

# Append rather than truncate: a long run gets restarted after every crash, and
# overwriting throws away the eval-AL trajectory that says whether the model is
# still converging. Crash detection below reads only the bytes this run wrote,
# so an error from an earlier attempt is not mistaken for a fresh one.
mkdir -p "${OUTPUT_DIR}"
LOG_OFFSET=$(wc -c < "${LOG_FILE}" 2>/dev/null || echo 0)
{
    echo "═══════════════════════════════════════════════════════════════"
    echo ">>> run started $(date '+%Y-%m-%d %H:%M:%S') — config=${CONFIG}"
    echo "═══════════════════════════════════════════════════════════════"
} >> "${LOG_FILE}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" \
        -m lumenrl.trainer.main \
        --config "${CONFIG}" \
        ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
        2>&1 \
    | grep --line-buffered -v -E "^[IWEF][0-9]{4} [0-9:.]+\s+[0-9]+ \S+\.(cpp|cc|h):" \
    | grep --line-buffered -vE "${MOONCAKE_NOISE_RE}" \
    | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -eq 0 ] && tail -c "+$((LOG_OFFSET + 1))" "${LOG_FILE}" 2>/dev/null | grep -qE \
    '(Traceback \(most recent call last\)|HfHubHTTPError|Training failed|FAILED|exitcode\s*:\s*-[0-9]+|MEMORY_APERTURE_VIOLATION|out of memory)'; then
    echo ">>> Crash detected in log despite torchrun exit code 0." >&2
    EXIT_CODE=1
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Kimi K3 DSpark distillation (ATOM, MI350) completed successfully."
else
    echo ">>> Kimi K3 DSpark distillation (ATOM, MI350) failed with exit code ${EXIT_CODE}." >&2
fi
echo ">>> Log: ${LOG_FILE}"
exit ${EXIT_CODE}
