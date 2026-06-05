#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K2.5 Eagle3 Phase 2 Training — ATOM MXFP4 Teacher (Docker)
#
# Uses phase2_atom.yaml config with:
#   - ATOM MXFP4 teacher (TP=4, GPUs 4-7) via Mooncake TCP
#   - FSDP2 draft model training (BF16, GPUs 0-3)
#   - Fixed RoPE (rope_theta=50000 + YaRN), loss mask (auto), VLM content parsing
#
# Resume from Phase 1 HF checkpoint.
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

DOCKER_IMAGE="${DOCKER_IMAGE:-lumenrl-vllm-mi350:latest}"
CONTAINER_NAME="kimi_k25_eagle3_v2_phase2_atom"
LUMENRL_DIR="/home/danyzhan/Lumen-RL"

echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 2: Mixed Domain Training (ATOM MXFP4 teacher)"
echo "  Image:     ${DOCKER_IMAGE}"
echo "  Container: ${CONTAINER_NAME}"
echo "  GPUs:      8x MI350 (0-3 training, 4-7 ATOM teacher)"
echo "  Teacher:   /dev/shm/Kimi-K2.5-MXFP4 (ATOM, TP=4)"
echo "  Config:    phase2_atom.yaml"
echo "  Fixes:     RoPE=50000+YaRN, loss_mask=auto, VLM parse"
echo "═══════════════════════════════════════════════════════════════"

# Clean stale lock files and compile caches
find /tmp -name '*.lock' -path '*aiter*' -delete 2>/dev/null || true
find /tmp -name '*.lock' -path '*hiprtc*' -delete 2>/dev/null || true

# Clear old tokenized cache (config changed: ltlo=auto, RoPE)
rm -rf /dev/shm/lumenrl_cache/tokenized_dataset/ 2>/dev/null || true
echo ">>> Cleared tokenized dataset cache"

# Create checkpoint dir
mkdir -p /dev/shm/checkpoints/kimi_k25_eagle3_v2_phase2_atom

docker run -d \
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
    -v /home/danyzhan/Kimi_K25_eagle3_v2_phase1_HF:/home/danyzhan/Kimi_K25_eagle3_v2_phase1_HF:ro \
    -w /root/lumenrl \
    "${DOCKER_IMAGE}" \
    bash -c "pip install -e /root/triton_kernels 2>/dev/null; find /root/aiter -name 'amd_buffer_addressing_builtins.hpp' -exec sed -i 's/#if __clang_major__ >= 21 && __clang_major__ < 23/#if 0/' {} \; 2>/dev/null; rm -rf /root/aiter/aiter/jit/build/module_moe_ck2stages* 2>/dev/null; rm -rf /root/aiter/aiter/jit/build 2>/dev/null; rm -rf /root/.cache/atom/* 2>/dev/null; mkdir -p /dev/shm/checkpoints/kimi_k25_eagle3_v2_phase2_atom; CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m lumenrl.trainer.main --config examples/Kimi_K25_SDDD_MI350_ATOM/configs/phase2_atom.yaml"

echo ""
echo ">>> Container '${CONTAINER_NAME}' started in detached mode."
echo ">>> Monitor with: docker logs -f ${CONTAINER_NAME}"
echo ">>> Dashboard:    dashboards/SDDD/Kimi_K25_SDDD_MI350/phase2.html"
