# syntax=docker/dockerfile:1.7
#
# SimWAM FlowGRPO environment for AMD Instinct MI350X.
#
# Torch and Triton stay on the base image (ROCm). This layer only adds the
# Python packages the trainer imports. Lumen and aiter are not baked in:
# launch_container.sh mounts the host checkouts and sets LUMEN_DIR / AITER_SRC.
#
# Build (no repo context is copied):
#   docker build \
#     -f examples/Embodied_SimWAM_NAVSIM/Dockerfile \
#     -t simwam-rl:mi350x \
#     examples/Embodied_SimWAM_NAVSIM
#
# Start, then install the mounted SimWAM tree:
#   IMAGE=simwam-rl:mi350x bash examples/Embodied_SimWAM_NAVSIM/launch_container.sh
#   docker exec -it simwam-rl bash examples/Embodied_SimWAM_NAVSIM/setup_env.sh
#
# Patches are selected at run time, not at build time:
#   LUMEN_ENABLE=hf_attn_patch,lumen_linear,aiter_conv

FROM zhangdanyangamd/lumen-rl:embodied-MI350X

ENV DEBIAN_FRONTEND=noninteractive \
    LUMEN_ENABLE=hf_attn_patch,lumen_linear,aiter_conv \
    LUMEN_DIR=/home/danyzhan/Lumen \
    AITER_SRC=/home/danyzhan/aiter \
    HF_HOME=/home/danyzhan/.cache/huggingface \
    MIOPEN_USER_DB_PATH=/home/danyzhan/.cache/miopen \
    MIOPEN_CUSTOM_CACHE_DIRECTORY=/home/danyzhan/.cache/miopen \
    PYTHONUNBUFFERED=1

# The base image symlinks /root/.cache/miopen to /work/cache/miopen, which is
# not present here. torch Conv3d fails if that link is left dangling.
RUN rm -f /root/.cache/miopen \
    && mkdir -p /root/.cache/miopen /home/danyzhan/.cache/miopen

# Do not install torch, triton, or aiter here. A newer wheel replaces the
# ROCm stack in the base image.
RUN python3 -m pip install --no-cache-dir \
      "hydra-core>=1.3" \
      "omegaconf>=2.3" \
      einops \
      accelerate \
      deepspeed \
      safetensors \
      imageio \
      imageio-ffmpeg \
      "huggingface-hub>=0.29" \
      modelscope \
      "transformers>=4.46" \
      scipy \
      matplotlib \
      pandas \
      pyquaternion \
      tqdm \
      pillow \
      av \
      tensorboard

WORKDIR /home/danyzhan/Lumen-RL
CMD ["sleep", "infinity"]
