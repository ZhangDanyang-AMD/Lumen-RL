#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# GPT-OSS-120B Eagle3 SDDD — Docker Launch (vLLM 0.11 + Mooncake TCP, MI308)
#
# GPU split (8x MI308):
#   GPUs 0-3: FSDP2 draft model training (BF16, LumenRL + aiter)
#   GPUs 4-7: vLLM teacher inference (TP=4, native MXFP4 gpt-oss-120b)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

SMOKE_TEST=false
PHASE2=false
for arg in "$@"; do
    case "${arg}" in
        --smoke-test) SMOKE_TEST=true ;;
        --phase2)     PHASE2=true ;;
    esac
done
if [ "${SMOKE_TEST}" = true ] && [ "${PHASE2}" = true ]; then
    echo "ERROR: --smoke-test and --phase2 are mutually exclusive" >&2
    exit 2
fi

DOCKER_IMAGE="${DOCKER_IMAGE:-lumenrl-vllm-mi308:latest}"
# Distinct container name per phase so phase 2 can launch alongside an
# accidentally-leftover phase 1 container instead of name-colliding.
if [ "${PHASE2}" = true ]; then
    CONTAINER_NAME="gpt_oss_120b_eagle3_vllm_mi308_phase2"
else
    CONTAINER_NAME="gpt_oss_120b_eagle3_vllm_mi308"
fi
LUMENRL_DIR="/home/leiwu/Lumen-RL"

if [ "${SMOKE_TEST}" = true ]; then
    RUN_CMD="bash examples/GPT_OSS_120b_MI308_ATOM/run_gpt_oss_120b.sh --smoke-test"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  GPT-OSS-120B Eagle3 Smoke Test (Docker) — vLLM, MI308"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  GPUs:     8x MI308 (0-3 training, 4-7 inference)"
    echo "  Transfer: Mooncake TCP"
    echo "═══════════════════════════════════════════════════════════════"
elif [ "${PHASE2}" = true ]; then
    RUN_CMD="bash examples/GPT_OSS_120b_MI308_ATOM/run_gpt_oss_120b.sh --phase2"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  GPT-OSS-120B Eagle3 Training (Docker) — Phase 2, vLLM, MI308"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  Dataset:  Magpie-Llama-3.1-Pro-300K-Filtered (resumes phase 1 ckpt)"
    echo "  GPUs:     8x MI308 (0-3 training, 4-7 inference)"
    echo "═══════════════════════════════════════════════════════════════"
else
    RUN_CMD="bash examples/GPT_OSS_120b_MI308_ATOM/run_gpt_oss_120b.sh"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  GPT-OSS-120B Eagle3 Training (Docker) — Phase 1, vLLM, MI308"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  Dataset:  HuggingFaceH4/ultrachat_200k"
    echo "  GPUs:     8x MI308 (0-3 training, 4-7 inference)"
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
    -e PYTORCH_ROCM_ARCH=gfx942 \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e LUMENRL_LOG_LEVEL=INFO \
    -e NCCL_TIMEOUT=7200 \
    -e GLOG_minloglevel=3 \
    -e GLOG_v=0 \
    -e GLOG_logtostderr=1 \
    -e MOONCAKE_LOG_LEVEL=FATAL \
    -e WANDB_MODE=disabled \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    --log-opt max-size=500m \
    --log-opt max-file=2 \
    -e MODEL_PATH="${MODEL_PATH:-}" \
    -e CKPT_DIR="${CKPT_DIR:-}" \
    -e LUMENRL_DRAFT_FLASH_BACKEND="${LUMENRL_DRAFT_FLASH_BACKEND:-}" \
    -e HSA_COREDUMP_FILE=/dev/null \
    -e HSA_ENABLE_COREDUMP=0 \
    -v /dev/shm:/dev/shm \
    -v "${LUMENRL_DIR}/lumenrl:/root/lumenrl/lumenrl" \
    -v "${LUMENRL_DIR}/examples:/root/lumenrl/examples" \
    -v "${LUMENRL_DIR}/output:/root/lumenrl/output" \
    -v "${LUMENRL_DIR}/third_party/Lumen/lumen:/root/Lumen/lumen" \
    -w /root/lumenrl \
    "${DOCKER_IMAGE}" \
    bash -c "${RUN_CMD}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Docker run (vLLM, MI308) completed successfully."
    echo ">>> Logs: ${LUMENRL_DIR}/output/GPT_OSS_120b_SDDD/LumenRL/"
else
    echo ">>> Docker run (vLLM, MI308) failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
