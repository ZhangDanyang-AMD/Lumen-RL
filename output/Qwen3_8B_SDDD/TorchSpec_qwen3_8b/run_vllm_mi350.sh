#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3-8B Eagle3 SDDD Smoke Test — TorchSpec + vLLM on MI350 (ROCm gfx950)
#
# GPU allocation (4x MI350):
#   GPUs 0-1: vLLM teacher inference (TP=2, extract_hidden_states)
#   GPUs 2-3: FSDP2 Eagle3 draft model training (BF16)
#
# Data flow:
#   Dataset -> vLLM Forward (GPUs 0-1) -> hidden states [B,T,D]
#     -> Mooncake TCP -> Eagle3 Draft Forward (GPUs 2-3, with grad)
#     -> Forward KL loss -> Backward -> Update
#
# Usage:
#   bash run_vllm_mi350.sh                        # 5-step smoke test
#   bash run_vllm_mi350.sh --steps 500            # full training
#   MODEL_PATH=/path/to/model bash run_vllm_mi350.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

NUM_STEPS="${NUM_STEPS:-5}"
for arg in "$@"; do
    case "${arg}" in
        --steps) shift; NUM_STEPS="${1}"; shift ;;
        --steps=*) NUM_STEPS="${arg#*=}" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/dev/shm/Qwen3-8B}"
DOCKER_IMAGE="${DOCKER_IMAGE:-torchspec-vllm-mi350:latest}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Qwen3-8B Eagle3 SDDD — TorchSpec + vLLM on MI350"
echo "  Model:       ${MODEL_PATH}"
echo "  GPUs:        ${GPUS} (2 inference TP=2, 2 training FSDP2)"
echo "  Steps:       ${NUM_STEPS}"
echo "  Docker:      ${DOCKER_IMAGE}"
echo "  Transfer:    Mooncake TCP"
echo "═══════════════════════════════════════════════════════════════"

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --shm-size=64g \
    --security-opt seccomp=unconfined \
    -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
    -e CUDA_VISIBLE_DEVICES="${GPUS}" \
    -e PYTORCH_ROCM_ARCH=gfx950 \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e TORCHSPEC_LOG_LEVEL=INFO \
    -e WANDB_MODE=disabled \
    "${DOCKER_IMAGE}" \
    python3 -m torchspec.train_entry \
        --config configs/vllm_qwen3_8b.yaml \
        model.target_model_path="${MODEL_PATH}" \
        training.num_train_steps="${NUM_STEPS}" \
        training.training_num_gpus_per_node=2 \
        inference.inference_num_gpus=2 \
        inference.inference_num_gpus_per_engine=2 \
        inference.inference_num_gpus_per_node=4 \
        inference.vllm.tp_size=2 \
        mooncake.protocol=tcp

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Qwen3-8B Eagle3 SDDD (vLLM, MI350) completed successfully."
else
    echo ">>> Qwen3-8B Eagle3 SDDD (vLLM, MI350) failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
