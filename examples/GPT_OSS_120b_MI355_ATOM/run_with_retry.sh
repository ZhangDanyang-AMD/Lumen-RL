#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Auto-restart wrapper around run_docker.sh, with idle-log watchdog — MI350.
#
# Absorbs ROCm failure modes by polling log mtime; if no new line for
# HANG_IDLE_SEC (default 600s), kills the container and lets the wrapper loop
# restart from the latest checkpoint (resume:true in config).
#
# Each crash/hang loses at most save_steps worth of progress. Loop exits 0
# when the trainer prints "SpecDistillTrainer.train finished after N steps".
#
# Usage:
#   bash examples/GPT_OSS_120b_MI355_ATOM/run_with_retry.sh
#   MAX_ATTEMPTS=50 HANG_IDLE_SEC=900 HF_TOKEN=hf_xxx \
#       bash examples/GPT_OSS_120b_MI355_ATOM/run_with_retry.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"
HANG_IDLE_SEC="${HANG_IDLE_SEC:-600}"
WATCHDOG_POLL_SEC="${WATCHDOG_POLL_SEC:-30}"
CONTAINER_NAME="${CONTAINER_NAME:-gpt_oss_120b_eagle3_mi350}"

LOG_DIR="${REPO_ROOT}/output/GPT_OSS_120b_SDDD/LumenRL"
LOG_FILE="${LOG_DIR}/gpt-oss-120b-eagle3-mi350.log"
SUCCESS_RE='SpecDistillTrainer\.train finished after [0-9]+ steps'

mkdir -p "${LOG_DIR}"

start_watchdog() {
    (
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

    kill "${wd_pid}" 2>/dev/null || true
    wait "${wd_pid}" 2>/dev/null || true

    if [ -f "${LOG_FILE}" ]; then
        rotated="${LOG_DIR}/gpt-oss-120b-eagle3-mi350.attempt-$(printf '%02d' "${attempt}").log"
        cp "${LOG_FILE}" "${rotated}" 2>/dev/null || true
    fi

    # Reap GPU coredumps
    if ls "${REPO_ROOT}"/gpucore.*.gpu >/dev/null 2>&1; then
        docker run --rm -v "${REPO_ROOT}":/host "${DOCKER_IMAGE:-lumenrl-vllm-mi350:latest}" \
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
