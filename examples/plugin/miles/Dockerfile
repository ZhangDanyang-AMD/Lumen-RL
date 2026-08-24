# Build a miles + LumenRL plugin image for Qwen3-8B Megatron + SGLang training.
#
# Usage:
#   docker build \
#     -f examples/plugin/miles/Dockerfile \
#     --build-arg MILES_REPO=https://github.com/your-org/miles.git \
#     -t lumenrl-miles-plugin .
#
#   docker run --gpus all -v /data:/data \
#     -e DATA_ROOT=/data -e MILES_ROOT=/workspace/miles \
#     lumenrl-miles-plugin
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

# ─── SGLang ──────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir sglang

# ─── miles ───────────────────────────────────────────────────────────────────
ARG MILES_REPO=https://github.com/your-org/miles.git
ARG MILES_BRANCH=main
RUN git clone --depth 1 -b ${MILES_BRANCH} ${MILES_REPO} /workspace/miles && \
    cd /workspace/miles && pip install -e .

# ─── LumenRL (includes plugin hooks for miles) ──────────────────────────────
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
from lumenrl.plugin.miles.hooks import custom_megatron_init, custom_megatron_before_train_step_hook; \
print('miles hooks verification: OK')"

ENV DATA_ROOT=/data
ENV MILES_ROOT=/workspace/miles
WORKDIR /workspace/Lumen-RL

CMD ["bash", "examples/plugin/miles/run.sh"]
