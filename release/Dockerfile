# LumenRL release image — everything the DAPO colocated examples need, so that a
# fresh container only has to pull code (or bind-mount it) and run.
#
# What is baked in:
#   - the four repos at the SHAs in release/versions.env
#   - flydsl 0.3.2 (the base image ships 0.1.4.2; aiter lumen/moe needs >= 0.2.4)
#   - megatron-core / Apex / TransformerEngine, for example 7
#   - the vLLM AITER RMSNorm patch, which otherwise has to be re-applied by hand
#     every time the container is recreated
#
# What is NOT baked in:
#   - model weights and datasets (bind-mount $DATA_ROOT)
#   - aiter's JIT kernels. Those need a GPU to compile, which `docker build` does
#     not have; release/precompile_kernels.sh does it in a running container and
#     commits the result. Skipping that step costs ~20 min on the first FP8 run.
#
# Build:  bash release/build_image.sh
# Run:    see release/README.md

ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.23.0
FROM ${BASE_IMAGE}

ARG PYTORCH_ROCM_ARCH=gfx950
ARG LUMENRL_REPO
ARG LUMENRL_BRANCH
ARG LUMENRL_SHA
ARG LUMEN_REPO
ARG LUMEN_BRANCH
ARG LUMEN_SHA
ARG AITER_REPO
ARG AITER_BRANCH
ARG AITER_SHA
ARG ATOM_REPO
ARG ATOM_BRANCH
ARG ATOM_SHA
ARG FLYDSL_VERSION=0.3.2
ARG MEGATRON_CORE_VERSION=0.18.2
ARG WITH_MEGATRON=1

ENV RL_ROOT=/opt/lumenrl \
    LUMEN_DIR=/opt/lumenrl/Lumen \
    AITER_DIR=/opt/lumenrl/aiter \
    ATOM_DIR=/opt/lumenrl/ATOM \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    DEBIAN_FRONTEND=noninteractive \
    GPU_ARCHS=${PYTORCH_ROCM_ARCH}

SHELL ["/bin/bash", "-lc"]

# ---------------------------------------------------------------- sources ----
# Shallow per-SHA fetches: the full histories are several GB and nothing in the
# runtime path reads them.
RUN mkdir -p "$RL_ROOT" && git config --global --add safe.directory '*'

RUN git init "$RL_ROOT/aiter" \
 && cd "$RL_ROOT/aiter" \
 && git remote add origin "${AITER_REPO}" \
 && git fetch --depth 1 origin "${AITER_SHA}" \
 && git checkout FETCH_HEAD \
 && git submodule update --init --depth 1 3rdparty/composable_kernel

RUN git init "$RL_ROOT/Lumen" \
 && cd "$RL_ROOT/Lumen" \
 && git remote add origin "${LUMEN_REPO}" \
 && git fetch --depth 1 origin "${LUMEN_SHA}" \
 && git checkout FETCH_HEAD

RUN git init "$RL_ROOT/ATOM" \
 && cd "$RL_ROOT/ATOM" \
 && git remote add origin "${ATOM_REPO}" \
 && git fetch --depth 1 origin "${ATOM_SHA}" \
 && git checkout FETCH_HEAD

RUN git init "$RL_ROOT/Lumen-RL" \
 && cd "$RL_ROOT/Lumen-RL" \
 && git remote add origin "${LUMENRL_REPO}" \
 && git fetch --depth 1 origin "${LUMENRL_SHA}" \
 && git checkout FETCH_HEAD

# ----------------------------------------------------------- python deps ----
# flydsl first and on its own line: the base image's amd-aiter wheel pins
# flydsl<0.1.5 while the aiter source tree requires >=0.2.4. They are mutually
# exclusive by design; PYTHONPATH decides which aiter wins at runtime, and the
# resulting pip "dependency conflict" warning about amd-aiter is expected.
RUN pip install --no-cache-dir "flydsl==${FLYDSL_VERSION}"

RUN pip install --no-cache-dir \
      "ray[default]>=2.9" "accelerate>=0.28" datasets \
      "math_verify[antlr4_13_2]" "omegaconf>=2.3,<2.4" safetensors wandb \
      psutil ninja pybind11 einops pandas

# Editable installs so a bind-mounted checkout over any of these paths keeps
# working without reinstalling.
RUN cd "$AITER_DIR" && AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop \
 && pip install --no-cache-dir -e "$LUMEN_DIR" --no-deps \
 && pip install --no-cache-dir -e "$RL_ROOT/Lumen-RL" --no-deps

# ------------------------------------------------- megatron / apex / TE ------
# Only needed by example 7. Set --build-arg WITH_MEGATRON=0 to skip (saves
# ~40 min of build time and ~20 GB).
RUN if [ "${WITH_MEGATRON}" = "1" ]; then \
      pip install --no-cache-dir --no-deps "megatron-core==${MEGATRON_CORE_VERSION}"; \
    fi

COPY build_apex_te.sh /tmp/build_apex_te.sh
RUN if [ "${WITH_MEGATRON}" = "1" ]; then bash /tmp/build_apex_te.sh; fi

# ------------------------------------------------- vLLM RMSNorm patch --------
# Examples 2/3/4 run the rollout with VLLM_ROCM_USE_AITER=1, and the plain AITER
# RMSNorm path has to pass use_model_sensitive_rmsnorm=1 to match Lumen's
# training-side norm. Patching the wheel here means it survives `docker rm`.
COPY patch_vllm_aiter_rmsnorm.py /opt/lumenrl/patch_vllm_aiter_rmsnorm.py
RUN python3 /opt/lumenrl/patch_vllm_aiter_rmsnorm.py

COPY entrypoint.sh /opt/lumenrl/entrypoint.sh
RUN chmod +x /opt/lumenrl/entrypoint.sh

# Also set these as image ENV, not just in the entrypoint: `docker exec` does not
# run the entrypoint, so without this an interactive `docker exec ... python3 -c
# "import aiter"` resolves to the base image's amd-aiter wheel and dies with
# "cannot import name 'fly_values' from flydsl.compiler.protocol" — the wheel
# pins flydsl<0.1.5 while we ship 0.3.2. The examples' own launcher sets its own
# PYTHONPATH, so only ad-hoc commands are affected, which makes it easy to miss.
ENV PYTHONPATH=/opt/lumenrl/Lumen-RL:/opt/lumenrl/aiter:/opt/lumenrl/Lumen \
    AITER_JIT_DIR=/opt/lumenrl/aiter-jit

WORKDIR /opt/lumenrl/Lumen-RL
ENTRYPOINT ["/opt/lumenrl/entrypoint.sh"]
CMD ["sleep", "infinity"]
