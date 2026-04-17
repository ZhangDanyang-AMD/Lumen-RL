# Default base image. Override with --build-arg BASE_IMAGE=... to match the
# FP8 alignment plan environment (rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260224).
ARG BASE_IMAGE=rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0
FROM ${BASE_IMAGE}

WORKDIR /workspace

# ─── System build dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake \
        libopenmpi-dev libibverbs-dev rdma-core libpci-dev && \
    rm -rf /var/lib/apt/lists/*

# ─── Copy LumenRL source (includes submodules) ──────────────────────────────
COPY . /workspace/Lumen-RL

# ─── AITER (lumen/triton_kernels branch, editable) ──────────────────────────
# PREBUILD_KERNELS=1 pre-compiles CK attention .so at install time.
# Without it the first CK call triggers JIT compilation which can SIGSEGV
# inside mp.spawn subprocesses (fork + HIP driver issue).
RUN cd /workspace/Lumen-RL/third_party/aiter && \
    PREBUILD_KERNELS=1 pip install -e .

# ─── MORI — SDMA communication library (sdma-new branch, editable) ──────────
RUN cd /workspace/Lumen-RL/third_party/mori && \
    git submodule update --init --recursive && \
    pip install setuptools-scm && \
    pip install -e .

# ─── Lumen (training engine, dev/RL branch, editable) ───────────────────────
RUN cd /workspace/Lumen-RL/third_party/Lumen && pip install -e ".[dev]"
RUN pip install -r /workspace/Lumen-RL/third_party/Lumen/requirements.txt

# ─── torchao (quantization reference) ───────────────────────────────────────
RUN pip install torchao>=0.8

# ─── Megatron-LM-AMD (optional Megatron backend) ────────────────────────────
ARG MEGATRON_COMMIT=8dd45f5a51378ec1ee7937dee3c20d8626df4763
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git /workspace/megatron_lm && \
    cd /workspace/megatron_lm && git checkout ${MEGATRON_COMMIT} && \
    pip install -e . -U --force-reinstall --no-deps && \
    cd megatron/core/datasets && make

# ─── LumenRL (this package, editable) ───────────────────────────────────────
RUN cd /workspace/Lumen-RL && pip install -e ".[test]"

# ─── PYTHONPATH: submodules accessible without install ──────────────────────
ENV PYTHONPATH="/workspace/megatron_lm:/workspace/Lumen-RL/third_party/mori:/workspace/Lumen-RL/third_party/aiter:/workspace/Lumen-RL/third_party/Lumen:/workspace/Lumen-RL:${PYTHONPATH:-}"

# ─── Runtime env (override HIP_VISIBLE_DEVICES at container launch) ─────────
ENV MORI_ENABLE_SDMA=1
ENV VLLM_USE_V1=1
ENV TORCHDYNAMO_DISABLE=1

RUN python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')" \
        > /etc/ld.so.conf.d/torch.conf && ldconfig

WORKDIR /workspace/Lumen-RL

CMD ["/bin/bash"]
