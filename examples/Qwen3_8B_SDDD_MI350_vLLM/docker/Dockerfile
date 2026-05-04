# ═══════════════════════════════════════════════════════════════════════════════
# LumenRL + vLLM + AITER — MI350 (ROCm gfx950)
#
# Self-contained image for Qwen3-8B Eagle3 SDDD training.
# All key libraries built from third_party/ source:
#   - aiter:  GPU compute kernels (shared by ATOM inference + LumenRL training)
#   - ATOM:   vLLM plugin for MXFP4 online quantization (uses aiter)
#   - Lumen:  Quantized training engine with aiter-patched FSDP2 (uses aiter)
#
# Also includes:
#   - vLLM 0.19.1 + TorchSpec patches (extract_hidden_states + MooncakeHiddenStatesConnector)
#   - Mooncake Transfer Engine (source-built HIP, TCP)
#   - TorchSpec (mooncake connectors for vLLM)
#   - LumenRL (training framework)
#
# Build (docker buildx with additional context for TorchSpec):
#   docker buildx build \
#       -f examples/Qwen3_8B_SDDD_MI350_vLLM/docker/Dockerfile \
#       --build-context torchspec=/home/danyzhan/TorchSpec \
#       -t lumenrl-vllm-mi350:latest .
#
# Or use the build script:
#   bash examples/Qwen3_8B_SDDD_MI350_vLLM/docker/build.sh
#
# (primary build context = Lumen-RL repo root)
# ═══════════════════════════════════════════════════════════════════════════════
FROM vllm/vllm-openai-rocm:v0.19.1

WORKDIR /root/

# ---- System dependencies ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git rsync dnsutils libibverbs-dev libnuma-dev \
        cmake build-essential libjsoncpp-dev libgoogle-glog-dev \
        wget golang-go pybind11-dev && \
    rm -rf /var/lib/apt/lists/*

# ---- Mooncake: pip scaffolding first, then source-build HIP backend ----
RUN pip install --no-cache-dir "mooncake-transfer-engine==0.3.10.post1" 2>/dev/null || \
    pip install --no-cache-dir "mooncake-transfer-engine-non-cuda>=0.3.10.post1"

ARG MOONCAKE_COMMIT=f2853a80a494b999d2115449dfd153dd25f5b42d
RUN git clone https://github.com/kvcache-ai/Mooncake.git /tmp/mooncake && \
    cd /tmp/mooncake && \
    git checkout ${MOONCAKE_COMMIT} && \
    git submodule update --init --recursive && \
    bash dependencies.sh -y && \
    pip install pybind11 && \
    export PATH="/usr/local/go/bin:$PATH" && \
    mkdir build && cd build && \
    cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON && \
    make -j$(nproc) && \
    make install && \
    SITE=$(python3 -c "import mooncake; import os; print(os.path.dirname(mooncake.__file__))") && \
    echo "Replacing pip mooncake .so files in $SITE with source-built versions" && \
    SRC_STORE=$(find /tmp/mooncake/build -name 'store*.so' -path '*/mooncake-integration/*' | head -1) && \
    SRC_ENGINE=$(find /tmp/mooncake/build -name 'engine*.so' -path '*/mooncake-integration/*' | head -1) && \
    if [ -n "$SRC_STORE" ]; then \
        rm -f "$SITE"/store*.so && \
        cp -v "$SRC_STORE" "$SITE/store.so" && \
        echo "Replaced all store .so with source-built HIP version"; \
    fi && \
    if [ -n "$SRC_ENGINE" ]; then \
        rm -f "$SITE"/engine*.so && \
        cp -v "$SRC_ENGINE" "$SITE/engine.so" && \
        echo "Replaced all engine .so with source-built HIP version"; \
    fi && \
    SRC_MASTER=$(find /tmp/mooncake/build -name 'mooncake_master' -type f -executable | head -1) && \
    if [ -n "$SRC_MASTER" ]; then cp -v "$SRC_MASTER" "$SITE/mooncake_master"; fi && \
    rm -rf /tmp/mooncake

# ---- Apply TorchSpec patches to vLLM (extract_hidden_states + MooncakeHiddenStatesConnector) ----
COPY --from=torchspec patches/vllm/v0.19.1/*.patch /tmp/patches/
RUN VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && \
    cd "$VLLM_DIR" && \
    for p in /tmp/patches/*.patch; do patch -p1 < "$p"; done && \
    rm -rf /tmp/patches

# ---- AITER: GPU compute kernels (shared by ATOM + LumenRL) ----
COPY third_party/aiter /root/aiter
RUN cd /root/aiter && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 PYTORCH_ROCM_ARCH=gfx950 pip install -e . && \
    echo "AITER installed: $(python3 -c 'import aiter; print(aiter.__version__)')"

# ---- ATOM: vLLM plugin for MXFP4 online quantization (uses aiter) ----
COPY third_party/ATOM /root/ATOM
RUN cd /root/ATOM && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install --no-deps -e . && \
    echo "ATOM installed: $(python3 -c 'import atom; print(atom.__version__)')"

# ---- Lumen: Quantized training engine with aiter-patched FSDP2 (uses aiter) ----
COPY third_party/Lumen /root/Lumen
RUN cd /root/Lumen && pip install --no-deps -e . && \
    echo "Lumen installed: $(python3 -c 'import lumen; print(lumen.__version__)')"

# ---- TorchSpec: mooncake connectors used by vLLM's MooncakeHiddenStatesConnector ----
COPY --from=torchspec . /root/torchspec
RUN cd /root/torchspec && pip install --no-deps -e . && \
    pip install --no-cache-dir omegaconf ray accelerate wandb pyzmq qwen-vl-utils \
    datasets transformers huggingface_hub safetensors pydantic openai numba

# ---- LumenRL: training framework ----
COPY . /root/lumenrl
RUN cd /root/lumenrl && pip install --no-deps -e .

# ---- Uninstall SGLang if present (not needed for vLLM path) ----
RUN pip uninstall -y sglang sglkernel 2>/dev/null || true

# ---- Environment ----
ENV PYTORCH_ROCM_ARCH=gfx950
ENV HIP_FORCE_DEV_KERNARG=1

ENTRYPOINT []
WORKDIR /root/lumenrl
