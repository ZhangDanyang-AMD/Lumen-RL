#!/usr/bin/env bash
# Install SimWAM on top of the image's ROCm torch. Do not reinstall torch or triton.
# Lumen is the dev/world-model-rl checkout on PYTHONPATH ($LUMEN_DIR), not a wheel.
set -euo pipefail

ROOT="${ROOT:-/home/danyzhan}"
REPO="${REPO:-$ROOT/Lumen-RL}"
SIMWAM="${SIMWAM_ROOT:-$REPO/examples/SimWAM}"
export LUMEN_DIR="${LUMEN_DIR:-$ROOT/Lumen}"
export PYTHONPATH="${LUMEN_DIR}:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
PY

python3 -m pip install -e "$SIMWAM" --no-deps
if [[ -d "$SIMWAM/navsim" ]]; then
    python3 -m pip install -e "$SIMWAM/navsim" --no-deps || true
fi

# Packages the image may not have. Versions are not pinned to the CUDA requirements.txt.
python3 -m pip install \
    "hydra-core>=1.3" "omegaconf>=2.3" einops accelerate deepspeed \
    safetensors imageio imageio-ffmpeg "huggingface-hub>=0.29" modelscope \
    "transformers>=4.46" scipy matplotlib pandas pyquaternion tqdm pillow av \
    tensorboard

python3 - <<PY
import os, sys
sys.path.insert(0, os.environ["LUMEN_DIR"])
sys.path.insert(0, "$REPO")
import lumen
from lumen.config import LumenConfig
print("lumen", lumen.__file__)
try:
    from aiter.ops.triton.conv.conv2d import conv2d
    print("aiter conv2d", conv2d)
except Exception as exc:
    print("aiter conv2d unavailable, torch conv will be used:", exc)
import simwam
print("simwam", simwam.__file__)
print("SETUP OK")
PY
