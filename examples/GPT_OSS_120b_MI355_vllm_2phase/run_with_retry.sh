#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Auto-restart wrapper for two-phase Eagle3 v3 training — MI355.
#
# Runs Phase 1 to completion, then Phase 2 to completion, with auto-restart
# on crashes. Absorbs ROCm failure modes by polling log mtime; if no new line
# for HANG_IDLE_SEC (default 600s), kills the container and restarts from
# the latest checkpoint.
#
# Usage:
#   bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_with_retry.sh
#   MAX_ATTEMPTS=50 bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_with_retry.sh
#   PHASE=2 bash examples/GPT_OSS_120b_MI355_vllm_2phase/run_with_retry.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"
HANG_IDLE_SEC="${HANG_IDLE_SEC:-600}"
WATCHDOG_POLL_SEC="${WATCHDOG_POLL_SEC:-30}"
START_PHASE="${PHASE:-1}"

LOG_DIR="${REPO_ROOT}/output/GPT_OSS_120b_SDDD_v3/LumenRL"
SUCCESS_RE='SpecDistillTrainer\.train finished after [0-9]+ steps'

mkdir -p "${LOG_DIR}"

start_watchdog() {
    local container_name="$1"
    local log_file="$2"
    (
        for _ in $(seq 1 60); do
            docker ps -q --filter "name=${container_name}" 2>/dev/null | grep -q . && break
            sleep 2
        done
        while docker ps -q --filter "name=${container_name}" 2>/dev/null | grep -q .; do
            sleep "${WATCHDOG_POLL_SEC}"
            if [ -f "${log_file}" ]; then
                age=$(( $(date +%s) - $(stat "${log_file}" --format='%Y' 2>/dev/null || echo 0) ))
                if [ "${age}" -gt "${HANG_IDLE_SEC}" ]; then
                    echo "[retry-wrapper/watchdog] log idle ${age}s > ${HANG_IDLE_SEC}s — killing container ${container_name}" >&2
                    docker stop -t 10 "${container_name}" >/dev/null 2>&1 || true
                    break
                fi
            fi
        done
    ) >/dev/null 2>&1 &
    echo $!
}

run_phase() {
    local phase="$1"
    local container_name="gpt_oss_120b_eagle3_v3_phase${phase}_mi355"
    local log_file="${LOG_DIR}/gpt-oss-120b-eagle3-v3-phase${phase}-mi355.log"

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  Starting Phase ${phase} with auto-retry (max ${MAX_ATTEMPTS} attempts)        ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    local attempt=0
    while [ "${attempt}" -lt "${MAX_ATTEMPTS}" ]; do
        attempt=$((attempt + 1))
        printf '\n[retry-wrapper] ── Phase %d, attempt %d/%d at %s ─────────────────\n' \
            "${phase}" "${attempt}" "${MAX_ATTEMPTS}" "$(date '+%Y-%m-%d %H:%M:%S')"

        wd_pid=$(start_watchdog "${container_name}" "${log_file}")

        PHASE="${phase}" CONTAINER_NAME="${container_name}" \
            bash "${SCRIPT_DIR}/run_docker.sh" --phase="${phase}"
        rc=$?

        kill "${wd_pid}" 2>/dev/null || true
        wait "${wd_pid}" 2>/dev/null || true

        if [ -f "${log_file}" ]; then
            rotated="${LOG_DIR}/gpt-oss-120b-eagle3-v3-phase${phase}-mi355.attempt-$(printf '%02d' "${attempt}").log"
            cp "${log_file}" "${rotated}" 2>/dev/null || true
        fi

        # Reap GPU coredumps
        if ls "${REPO_ROOT}"/gpucore.*.gpu >/dev/null 2>&1; then
            docker run --rm -v "${REPO_ROOT}":/host "${DOCKER_IMAGE:-gpt_oss_eagle3_vllm_train:latest}" \
                bash -c 'rm -f /host/gpucore.*.gpu' >/dev/null 2>&1 || true
            echo "[retry-wrapper] reaped GPU coredump files"
        fi

        if [ -f "${log_file}" ] && grep -qE "${SUCCESS_RE}" "${log_file}"; then
            echo "[retry-wrapper] Phase ${phase} reached completion marker — done."
            echo "[retry-wrapper] attempts used: ${attempt}/${MAX_ATTEMPTS}"
            return 0
        fi

        echo "[retry-wrapper] Phase ${phase} attempt ${attempt} ended (exit=${rc}) without completion marker."
        echo "[retry-wrapper] sleeping ${RETRY_SLEEP}s before next resume from latest ckpt..."
        sleep "${RETRY_SLEEP}"
    done

    echo "[retry-wrapper] Phase ${phase}: exhausted ${MAX_ATTEMPTS} attempts without completion." >&2
    return 1
}

# Run phases sequentially
for phase in $(seq "${START_PHASE}" 2); do
    if ! run_phase "${phase}"; then
        echo "[retry-wrapper] Phase ${phase} failed. Aborting." >&2
        exit 1
    fi
done

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Two-phase Eagle3 v3 training completed successfully!       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
exit 0
