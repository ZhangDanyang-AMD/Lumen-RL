# Default base image. Override with --build-arg BASE_IMAGE=... to match the
# FP8 alignment plan environment (rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260224).
ARG BASE_IMAGE=rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
FROM ${BASE_IMAGE}

WORKDIR /workspace

# ─── System build dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake \
        libopenmpi-dev libibverbs-dev rdma-core libpci-dev && \
    rm -rf /var/lib/apt/lists/*

# ─── Copy LumenRL source tree (.git included for submodule refs) ─────────────
# Prerequisites (run on host BEFORE docker build):
#   git submodule update --init --recursive
COPY . /workspace/Lumen-RL

# ─── AITER (single shared copy for ALL components) ──────────────────────────
# Install from LumenRL's top-level third_party/aiter.  Both Lumen and ATOM
# import aiter at runtime; by installing once and symlinking into Lumen's own
# third_party/ we guarantee they share the same AITER build.
# PREBUILD_KERNELS=1 pre-compiles CK attention .so at install time.
RUN cd /workspace/Lumen-RL/third_party/aiter && \
    PREBUILD_KERNELS=1 pip install -e .

# Symlink Lumen's own third_party/aiter → the shared copy.
RUN rm -rf /workspace/Lumen-RL/third_party/Lumen/third_party/aiter && \
    ln -s /workspace/Lumen-RL/third_party/aiter \
          /workspace/Lumen-RL/third_party/Lumen/third_party/aiter

# ─── MORI — SDMA communication library (sdma-new branch, editable) ──────────
# MORI's setup.py imports torch for cmake prefix and GPU arch detection.
# pip's build isolation hides the base image's torch, so we build the C++
# extension in-place and rely on PYTHONPATH for the Python package.
RUN pip install setuptools-scm && \
    cd /workspace/Lumen-RL/third_party/mori && \
    PYTORCH_ROCM_ARCH="gfx942" python setup.py build_ext --inplace

# Symlink Lumen's own third_party/mori → the shared copy.
RUN rm -rf /workspace/Lumen-RL/third_party/Lumen/third_party/mori && \
    ln -s /workspace/Lumen-RL/third_party/mori \
          /workspace/Lumen-RL/third_party/Lumen/third_party/mori

# ─── Lumen (training engine, dev/RL branch, editable) ───────────────────────
RUN cd /workspace/Lumen-RL/third_party/Lumen && pip install -e ".[dev]"
RUN pip install -r /workspace/Lumen-RL/third_party/Lumen/requirements.txt

# ─── ATOM (inference engine, Lumen/RL branch, editable) ────────────────────
# ATOM imports aiter at runtime — the shared pip-editable install above
# makes the aiter package available globally.
RUN cd /workspace/Lumen-RL/third_party/ATOM && pip install -e .

# ─── torchao (quantization reference) ───────────────────────────────────────
RUN pip install 'torchao>=0.8,<0.10'

# ─── RL training dependencies ──────────────────────────────────────────────
RUN pip install datasets "math_verify[antlr4_13_2]"

# ─── Megatron-LM-AMD (optional Megatron backend) ────────────────────────────
ARG MEGATRON_COMMIT=8dd45f5a51378ec1ee7937dee3c20d8626df4763
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git /workspace/megatron_lm && \
    cd /workspace/megatron_lm && git checkout ${MEGATRON_COMMIT} && \
    pip install -e . -U --force-reinstall --no-deps && \
    cd megatron/core/datasets && make

# ─── LumenRL (this package, editable) ───────────────────────────────────────
RUN cd /workspace/Lumen-RL && pip install -e ".[test]"

# ─── PYTHONPATH: submodules accessible without install ──────────────────────
ENV PYTHONPATH="/workspace/megatron_lm:/workspace/Lumen-RL/third_party/ATOM:/workspace/Lumen-RL/third_party/mori/python:/workspace/Lumen-RL/third_party/aiter:/workspace/Lumen-RL/third_party/Lumen:/workspace/Lumen-RL:${PYTHONPATH:-}"

# ─── Runtime env (override HIP_VISIBLE_DEVICES at container launch) ─────────
ENV MORI_ENABLE_SDMA=1
ENV VLLM_USE_V1=1
ENV TORCHDYNAMO_DISABLE=1

# ─── ATOM/vLLM attention backend ────────────────────────────────────────────
# Prefer AITER CK/ASM Flash Attention (ROCM_AITER_FA). If the build is broken
# (e.g. gfx950 fmha_fwd_v3 undefined on MI300X), fall back to
# ROCM_AITER_UNIFIED_ATTN which uses AITER's unified Triton kernel and does
# NOT depend on the fmha_v3 module.
# Scripts override this via VLLM_ROCM_ATTN_BACKEND env var at runtime.
ENV VLLM_ROCM_ATTN_BACKEND="ROCM_AITER_FA"
ENV VLLM_ROCM_ATTN_BACKEND_FALLBACK="ROCM_AITER_UNIFIED_ATTN"

RUN python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')" \
        > /etc/ld.so.conf.d/torch.conf && ldconfig

WORKDIR /workspace/Lumen-RL

CMD ["/bin/bash"]
