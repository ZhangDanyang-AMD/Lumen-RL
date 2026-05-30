#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K2.5 Eagle3 Two-Phase Training — Docker Launch (vLLM+ATOM, MI350)
#
# Uses self-contained Docker image with:
#   - vLLM 0.19.1 + extract_hidden_states patches
#   - ATOM (vLLM plugin, MXFP4 online quantization) from third_party/ATOM
#   - AITER (GPU kernels, shared by ATOM and LumenRL) from third_party/aiter
#   - LumenRL (FSDP2 training, aiter-patched) from third_party/Lumen
#   - MORI (MoE all-to-all) from third_party/mori
#   - Mooncake Transfer Engine (source-built HIP, TCP)
#
# GPU split (8x MI350):
#   GPUs 0-3: FSDP2 draft model training (BF16, LumenRL + aiter)
#   GPUs 4-7: vLLM + ATOM teacher inference (TP=4, MXFP4, aiter)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

PHASE2_ONLY=false
SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --phase2-only) PHASE2_ONLY=true ;;
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

DOCKER_IMAGE="${DOCKER_IMAGE:-lumenrl-vllm-mi350:latest}"
CONTAINER_NAME="kimi_k25_eagle3_v2_phase1"

LUMENRL_DIR="/home/danyzhan/Lumen-RL"

# Select run command
if [ "${SMOKE_TEST}" = true ]; then
    RUN_CMD="bash examples/Kimi_K25_SDDD_MI350_vLLM/run_kimi_k25.sh --smoke-test"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Kimi K2.5 Eagle3 Smoke Test (Docker) — vLLM+ATOM, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  GPUs:     8x MI350 (0-3 training, 4-7 inference)"
    echo "  Teacher:  Qwen3-8B (vLLM+ATOM, MXFP4, aiter)"
    echo "  Transfer: Mooncake TCP"
    echo "═══════════════════════════════════════════════════════════════"
elif [ "${PHASE2_ONLY}" = true ]; then
    RUN_CMD="bash examples/Kimi_K25_SDDD_MI350_vLLM/run_full_training.sh --phase2-only"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Kimi K2.5 Eagle3 Training (Docker) — vLLM+ATOM, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  GPUs:     8x MI350 (0-3 training, 4-7 inference)"
    echo "  Phase:    Phase 2 only"
    echo "═══════════════════════════════════════════════════════════════"
else
    RUN_CMD="bash examples/Kimi_K25_SDDD_MI350_vLLM/run_full_training.sh"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Kimi K2.5 Eagle3 Training (Docker) — vLLM+ATOM, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  GPUs:     8x MI350 (0-3 training, 4-7 inference)"
    echo "  Phase:    Phase 1 + 2"
    echo "═══════════════════════════════════════════════════════════════"
fi

# Clean stale lock files and compile caches that can deadlock AITER JIT
find /tmp -name '*.lock' -path '*aiter*' -delete 2>/dev/null || true
find /tmp -name '*.lock' -path '*hiprtc*' -delete 2>/dev/null || true
rm -rf /root/.cache/atom/* 2>/dev/null || true

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
    -e LUMENRL_TEACHER_READY_TIMEOUT_SECONDS=1800 \
    -e ATOM_WARMUP_MAX_TOKENS=8192 \
    -e ATOM_USE_TRITON_MOE=1 \
    -e GLOG_minloglevel=3 \
    -e MOONCAKE_LOG_LEVEL=FATAL \
    -e MOONCAKE_VLOG_LEVEL=-1 \
    -e WANDB_MODE=disabled \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    --log-opt max-size=500m \
    --log-opt max-file=2 \
    -v /dev/shm:/dev/shm \
    -v "${LUMENRL_DIR}/lumenrl:/root/lumenrl/lumenrl" \
    -v "${LUMENRL_DIR}/examples:/root/lumenrl/examples" \
    -v "${LUMENRL_DIR}/output:/root/lumenrl/output" \
    -v "${LUMENRL_DIR}/third_party/Lumen/lumen:/root/Lumen/lumen" \
    -v "${LUMENRL_DIR}/third_party/ATOM/atom:/root/ATOM/atom" \
    -v "${LUMENRL_DIR}/third_party/triton_kernels:/root/triton_kernels" \
    -w /root/lumenrl \
    "${DOCKER_IMAGE}" \
    bash -c "pip install -e /root/triton_kernels 2>/dev/null; find /root/aiter -name 'amd_buffer_addressing_builtins.hpp' -exec sed -i 's/#if __clang_major__ >= 21 && __clang_major__ < 23/#if 0/' {} \; 2>/dev/null; rm -rf /root/aiter/aiter/jit/build/module_moe_ck2stages* 2>/dev/null; ${RUN_CMD}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Docker run (vLLM+ATOM, MI350) completed successfully."
else
    echo ">>> Docker run (vLLM+ATOM, MI350) failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
