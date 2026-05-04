#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3-8B Eagle3 SDDD — Docker Launch (vLLM + Mooncake TCP, MI350)
#
# GPU split (8x MI350):
#   GPUs 0-3: FSDP2 draft model training (BF16, LumenRL + aiter)
#   GPUs 4-7: vLLM teacher inference (TP=4)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

DOCKER_IMAGE="${DOCKER_IMAGE:-lumenrl-vllm-mi350:latest}"
CONTAINER_NAME="qwen3_8b_eagle3_vllm_mi350"
LUMENRL_DIR="/home/danyzhan/Lumen-RL"

if [ "${SMOKE_TEST}" = true ]; then
    RUN_CMD="bash examples/Qwen3_8B_SDDD_MI350_vLLM/run_qwen3_8b.sh --smoke-test"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Qwen3-8B Eagle3 Smoke Test (Docker) — vLLM, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  GPUs:     8x MI350 (0-3 training, 4-7 inference)"
    echo "  Transfer: Mooncake TCP"
    echo "═══════════════════════════════════════════════════════════════"
else
    RUN_CMD="bash examples/Qwen3_8B_SDDD_MI350_vLLM/run_qwen3_8b.sh"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Qwen3-8B Eagle3 Training (Docker) — vLLM, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  GPUs:     8x MI350 (0-3 training, 4-7 inference)"
    echo "═══════════════════════════════════════════════════════════════"
fi

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    --shm-size 64G \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e PYTORCH_ROCM_ARCH=gfx950 \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e LUMENRL_LOG_LEVEL=INFO \
    -e NCCL_TIMEOUT=7200 \
    -e WANDB_MODE=disabled \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v /dev/shm:/dev/shm \
    -v "${LUMENRL_DIR}/output:/root/lumenrl/output" \
    -w /root/lumenrl \
    "${DOCKER_IMAGE}" \
    bash -c "${RUN_CMD}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Docker run (vLLM, MI350) completed successfully."
    echo ">>> Logs: ${LUMENRL_DIR}/output/Qwen3_8B_SDDD/LumenRL/"
else
    echo ">>> Docker run (vLLM, MI350) failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
