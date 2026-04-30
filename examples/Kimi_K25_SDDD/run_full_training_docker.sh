#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K2.5 Eagle3 Two-Phase Training — Docker Launch
#
# Uses lumenrl_sddd_snapshot:latest container with ROCm + PyTorch + SGLang
#
# GPU split (8x GPU):
#   GPUs 0-3: FSDP2 draft model training (BF16)
#   GPUs 4-7: SGLang + ATOM teacher inference (TP=4, MXFP4)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

PHASE2_ONLY=false
for arg in "$@"; do
    case "${arg}" in
        --phase2-only) PHASE2_ONLY=true ;;
    esac
done

DOCKER_IMAGE="${DOCKER_IMAGE:-lumenrl_sddd_snapshot:latest}"
CONTAINER_NAME="kimi_k25_eagle3_training"

# Paths to mount
LUMENRL_DIR="/home/danyzhan/Lumen-RL"
ATOM_DIR="/home/danyzhan/ATOM_repo"
TORCHSPEC_DIR="/home/danyzhan/TorchSpec"
DATASETS_DIR="/datasets"

# Select phase config
if [ "${PHASE2_ONLY}" = true ]; then
    PHASE_ARG="--phase2-only"
else
    PHASE_ARG=""
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Kimi K2.5 Eagle3 Training (Docker)"
echo "  Image:    ${DOCKER_IMAGE}"
echo "  GPUs:     8 (0-3 training, 4-7 inference)"
echo "  Phase:    ${PHASE2_ONLY:+Phase 2 only}${PHASE2_ONLY:-Phase 1 + 2}"
echo "═══════════════════════════════════════════════════════════════"

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    --shm-size 64G \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e LUMENRL_LOG_LEVEL=INFO \
    -e NCCL_TIMEOUT=7200 \
    -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    -e SGLANG_DISABLE_CUDNN_CHECK=1 \
    -e PYTHONPATH="/home/danyzhan/ATOM_repo:/home/danyzhan/TorchSpec:${PYTHONPATH:-}" \
    -v "${LUMENRL_DIR}:${LUMENRL_DIR}" \
    -v "${ATOM_DIR}:${ATOM_DIR}" \
    -v "${TORCHSPEC_DIR}:${TORCHSPEC_DIR}" \
    -v "${DATASETS_DIR}:${DATASETS_DIR}" \
    -v /docker:/docker \
    -w "${LUMENRL_DIR}" \
    "${DOCKER_IMAGE}" \
    bash -c "bash examples/Kimi_K25_SDDD/run_full_training.sh ${PHASE_ARG}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Docker training completed successfully."
else
    echo ">>> Docker training failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
