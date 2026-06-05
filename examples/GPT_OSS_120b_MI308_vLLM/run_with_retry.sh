#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Auto-restart wrapper around run_docker.sh, with idle-log watchdog.
#
# Absorbs two known MI308 / ROCm failure modes:
#   - Bug A: aiter triton flash attention launch error (HSA fault). Bypass by
#     exporting LUMENRL_DRAFT_FLASH_BACKEND=matmul before invoking this script.
#   - Bug B: silent step-700-class hang (rank 0 stuck in .item() after backward;
#     other ranks stuck in metrics all_reduce). NCCL watchdog cannot detect it
#     because rank 0 is in a CUDA sync, not a collective. We poll the log
#     mtime instead; if no new line for HANG_IDLE_SEC (default 600s), kill the
#     container and let the wrapper loop restart.
#
# Each crash/hang loses at most save_steps worth of progress (resume:true picks
# up the latest ckpt). Loop exits 0 when the trainer prints the success marker
# "SpecDistillTrainer.train finished after N steps".
#
# Per-attempt log is rotated to *.attempt-NN.log so the next attempt's tee
# doesn't clobber forensic evidence.
#
# Usage:
#   LUMENRL_DRAFT_FLASH_BACKEND=matmul \
#       bash examples/GPT_OSS_120b_MI308_vLLM/run_with_retry.sh
#   MAX_ATTEMPTS=50 HANG_IDLE_SEC=900 HF_TOKEN=hf_xxx \
#       bash examples/.../run_with_retry.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"
HANG_IDLE_SEC="${HANG_IDLE_SEC:-600}"        # 10 min of log silence ⇒ hang
WATCHDOG_POLL_SEC="${WATCHDOG_POLL_SEC:-30}" # check log mtime every 30s
CONTAINER_NAME="${CONTAINER_NAME:-gpt_oss_120b_eagle3_vllm_mi308}"

LOG_DIR="${REPO_ROOT}/output/GPT_OSS_120b_SDDD/LumenRL"
LOG_FILE="${LOG_DIR}/gpt-oss-120b-eagle3-vllm-mi308.log"
SUCCESS_RE='SpecDistillTrainer\.train finished after [0-9]+ steps'

mkdir -p "${LOG_DIR}"

# Hang watchdog: if log mtime stops advancing for HANG_IDLE_SEC, kill container.
# Runs in background per-attempt and exits when container disappears.
start_watchdog() {
    # IMPORTANT: redirect stdout of the backgrounded subshell to /dev/null
    # so $(start_watchdog) command substitution can return as soon as the
    # function prints the PID. Otherwise the bg subshell holds the function's
    # stdout pipe open and the caller hangs forever.
    (
        # Wait for container to actually exist before watching.
        for _ in $(seq 1 60); do
            docker ps -q --filter "name=${CONTAINER_NAME}" 2>/dev/null | grep -q . && break
            sleep 2
        done
        while docker ps -q --filter "name=${CONTAINER_NAME}" 2>/dev/null | grep -q .; do
            sleep "${WATCHDOG_POLL_SEC}"
            if [ -f "${LOG_FILE}" ]; then
                age=$(( $(date +%s) - $(stat "${LOG_FILE}" --format='%Y' 2>/dev/null || echo 0) ))
                if [ "${age}" -gt "${HANG_IDLE_SEC}" ]; then
                    echo "[retry-wrapper/watchdog] log idle ${age}s > ${HANG_IDLE_SEC}s — killing container ${CONTAINER_NAME}" >&2
                    docker stop -t 10 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
                    break
                fi
            fi
        done
    ) >/dev/null 2>&1 &
    echo $!
}

attempt=0
while [ "${attempt}" -lt "${MAX_ATTEMPTS}" ]; do
    attempt=$((attempt + 1))
    printf '\n[retry-wrapper] ── attempt %d/%d at %s ─────────────────\n' \
        "${attempt}" "${MAX_ATTEMPTS}" "$(date '+%Y-%m-%d %H:%M:%S')"

    wd_pid=$(start_watchdog)

    bash "${SCRIPT_DIR}/run_docker.sh" "$@"
    rc=$?

    # Stop watchdog; container is already gone or just got killed by us.
    kill "${wd_pid}" 2>/dev/null || true
    wait "${wd_pid}" 2>/dev/null || true

    # Rotate the just-finished attempt log.
    if [ -f "${LOG_FILE}" ]; then
        rotated="${LOG_DIR}/gpt-oss-120b-eagle3-vllm-mi308.attempt-$(printf '%02d' "${attempt}").log"
        cp "${LOG_FILE}" "${rotated}" 2>/dev/null || true
    fi

    # Reap GPU coredumps (4-17 GB each per faulting rank). They're root-owned
    # because dropped by the in-container HSA runtime, so use a one-shot
    # container to delete them. Several crashes in a row otherwise fill the
    # host disk quickly.
    if ls "${REPO_ROOT}"/gpucore.*.gpu >/dev/null 2>&1; then
        docker run --rm -v "${REPO_ROOT}":/host lumenrl-vllm-mi308:latest \
            bash -c 'rm -f /host/gpucore.*.gpu' >/dev/null 2>&1 || true
        echo "[retry-wrapper] reaped GPU coredump files in ${REPO_ROOT}"
    fi

    if [ -f "${LOG_FILE}" ] && grep -qE "${SUCCESS_RE}" "${LOG_FILE}"; then
        echo "[retry-wrapper] training reached completion marker — done."
        echo "[retry-wrapper] attempts used: ${attempt}/${MAX_ATTEMPTS}"
        exit 0
    fi

    echo "[retry-wrapper] attempt ${attempt} ended (exit=${rc}) without completion marker."
    echo "[retry-wrapper] sleeping ${RETRY_SLEEP}s before next resume from latest ckpt..."
    sleep "${RETRY_SLEEP}"
done

echo "[retry-wrapper] exhausted ${MAX_ATTEMPTS} attempts without completion." >&2
exit 1
