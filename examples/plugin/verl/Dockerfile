# Build a verl + LumenRL plugin image for Qwen3-8B GRPO (FSDP2 + ATOM).
#
# Usage:
#   docker build -f examples/plugin/verl/Dockerfile -t lumenrl-verl-plugin .
#   docker run --gpus all -v /data:/data -e DATA_ROOT=/data lumenrl-verl-plugin
#
# Build context must be the Lumen-RL repo root.

ARG BASE_IMAGE=rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
FROM ${BASE_IMAGE}

WORKDIR /workspace

# ─── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake \
        libopenmpi-dev libibverbs-dev rdma-core && \
    rm -rf /var/lib/apt/lists/*

# ─── verl ────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir verl

# ─── LumenRL (includes plugin entry-point for verl.plugins) ─────────────────
COPY . /workspace/Lumen-RL
RUN cd /workspace/Lumen-RL && pip install -e ".[engine]"

# ─── AITER (shared CK kernels) ──────────────────────────────────────────────
RUN if [ -d /workspace/Lumen-RL/third_party/aiter ]; then \
        cd /workspace/Lumen-RL/third_party/aiter && \
        PREBUILD_KERNELS=1 pip install -e . ; \
    fi

# ─── Lumen (FP8 training engine) ────────────────────────────────────────────
RUN if [ -d /workspace/Lumen-RL/third_party/Lumen ]; then \
        cd /workspace/Lumen-RL/third_party/Lumen && pip install -e . ; \
    fi

# ─── ATOM (inference engine) ────────────────────────────────────────────────
RUN if [ -d /workspace/Lumen-RL/third_party/ATOM ]; then \
        cd /workspace/Lumen-RL/third_party/ATOM && pip install -e . ; \
    fi

# ─── Verify plugin registration ─────────────────────────────────────────────
RUN python3 -c "\
import verl; \
from verl.workers.rollout.base import _ROLLOUT_REGISTRY; \
assert ('atom', 'async') in _ROLLOUT_REGISTRY, 'ATOM rollout not registered'; \
print('Plugin verification: OK')"

ENV DATA_ROOT=/data
WORKDIR /workspace/Lumen-RL

CMD ["bash", "examples/plugin/verl/run.sh"]
