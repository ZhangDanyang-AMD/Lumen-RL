# Build a vime + LumenRL plugin image for Qwen3-8B Megatron training.
#
# Usage:
#   docker build \
#     -f examples/plugin/vime/Dockerfile \
#     --build-arg VIME_REPO=https://github.com/your-org/vime.git \
#     -t lumenrl-vime-plugin .
#
#   docker run --gpus all -v /data:/data \
#     -e DATA_ROOT=/data -e VIME_ROOT=/workspace/vime \
#     lumenrl-vime-plugin
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

# ─── Megatron-LM (ROCm fork) ────────────────────────────────────────────────
ARG MEGATRON_REPO=https://github.com/ROCm/Megatron-LM.git
ARG MEGATRON_BRANCH=rocm_dev
RUN git clone --depth 1 -b ${MEGATRON_BRANCH} ${MEGATRON_REPO} /workspace/Megatron-LM && \
    cd /workspace/Megatron-LM && pip install -e .

# ─── vime ────────────────────────────────────────────────────────────────────
ARG VIME_REPO=https://github.com/your-org/vime.git
ARG VIME_BRANCH=main
RUN git clone --depth 1 -b ${VIME_BRANCH} ${VIME_REPO} /workspace/vime && \
    cd /workspace/vime && pip install -e .

# ─── LumenRL (includes plugin hooks for vime) ───────────────────────────────
COPY . /workspace/Lumen-RL
RUN cd /workspace/Lumen-RL && pip install -e ".[engine,megatron]"

# ─── AITER ───────────────────────────────────────────────────────────────────
RUN if [ -d /workspace/Lumen-RL/third_party/aiter ]; then \
        cd /workspace/Lumen-RL/third_party/aiter && \
        PREBUILD_KERNELS=1 pip install -e . ; \
    fi

# ─── Lumen (FP8 training engine) ────────────────────────────────────────────
RUN if [ -d /workspace/Lumen-RL/third_party/Lumen ]; then \
        cd /workspace/Lumen-RL/third_party/Lumen && pip install -e . ; \
    fi

# ─── Verify hooks ───────────────────────────────────────────────────────────
RUN python3 -c "\
from lumenrl.plugin.vime.hooks import custom_megatron_init, custom_model_provider; \
print('vime hooks verification: OK')"

ENV DATA_ROOT=/data
ENV VIME_ROOT=/workspace/vime
WORKDIR /workspace/Lumen-RL

CMD ["bash", "examples/plugin/vime/run.sh"]
