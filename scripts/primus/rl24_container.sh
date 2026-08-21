#!/usr/bin/env bash
# The LumenRL container on a glibc-2.39 base, so it can use the ionic RoCE fabric.
# Runs on the node (docker host side).
#
# Why this image rather than vllm/vime-rocm (dapo-lumenrl-native-vllm-fsdp-runbook §4):
# the ANP plugin that RDMA needs wants GLIBC_2.38 + GLIBCXX_3.4.32, and every
# 22.04 ROCm image tops out at 2.35 / 3.4.30 (32gpu-runbook §6.2.2). This one is
# Ubuntu 24.04 / glibc 2.39 / GLIBCXX 3.4.33 AND ships vLLM 0.23.1, which is the
# version the runbook's rollout path is written against -- rocm/primus is 24.04
# too but has no vLLM at all.
#
#   image     rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
#   ⚠️ cdna    the `rdna` tag of the same build is for consumer GPUs, not MI350
#
# The ANP bind-mounts are included so this one container can do both training and
# RDMA; they are inert until run_a2a_anp.sh-style env is set.
set -euo pipefail
# Per-allocation settings; see ray_start_primus.sh.
source "${LUMEN_CLUSTER_ENV:-/home/xysheng/4node/env.sh}"

IMAGE=${RL24_IMAGE:-rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0}
NAME=${RL24_CONTAINER:-rl-vllm-24}
PLUGIN=${ANP_PLUGIN:-/opt/rocm-7.0.1/lib/librccl-net.so}

MOUNTS=()
IONIC_LIB="$(ls /usr/lib/x86_64-linux-gnu/libionic.so.* 2>/dev/null | head -1 || true)"
if [ -n "$IONIC_LIB" ] && [ -f "$PLUGIN" ]; then
  MOUNTS+=(-v /opt/openmpi:/opt/openmpi:ro
           -v "$IONIC_LIB":/usr/lib/x86_64-linux-gnu/libionic.so.1:ro)
  for n in librccl-net.so librccl-net-anp.so libnccl-net.so libnccl-net-anp.so; do
    MOUNTS+=(-v "$PLUGIN:/opt/anp/$n:ro")
  done
else
  echo "WARNING: no ionic lib or ANP plugin on this host; RDMA will not be available" >&2
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --entrypoint /bin/bash \
  --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  --shm-size 64G \
  -v /home/xysheng:/home/xysheng \
  -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  "${MOUNTS[@]}" \
  -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" -e SCRATCH_ROOT="$SCRATCH_ROOT" \
  -e HF_HOME="$DATA_ROOT/hf_home" \
  -e LUMEN_DIR="$RL_ROOT/Lumen" -e AITER_DIR="$RL_ROOT/aiter" -e ATOM_DIR="$RL_ROOT/ATOM" \
  "$IMAGE" -lc "sleep infinity"

docker exec "$NAME" bash -lc '
. /etc/os-release; echo "  $PRETTY_NAME  glibc $(ldd --version | head -1 | grep -oE "[0-9]+\.[0-9]+$")"
python3 -c "import torch, vllm, ray; print(f\"  torch {torch.__version__}  vllm {vllm.__version__}  ray {ray.__version__}  gpus {torch.cuda.device_count()}\")"
echo "  ulimit -l: $(ulimit -l)"'
echo "container $NAME up on $(hostname)"
