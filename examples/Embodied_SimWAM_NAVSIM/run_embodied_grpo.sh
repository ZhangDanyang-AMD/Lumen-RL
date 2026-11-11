#!/usr/bin/env bash
# FlowGRPO for SimWAM on NAVSIM, same launch shape as the embodied GRPO scripts.
#
# Usage (inside the container, from any cwd):
#   bash examples/Embodied_SimWAM_NAVSIM/run_embodied_grpo.sh
#   NPROC=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash examples/Embodied_SimWAM_NAVSIM/run_embodied_grpo.sh
# Extra Hydra overrides are appended after the config file.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SIMWAM="${SIMWAM_ROOT:-$REPO/examples/SimWAM}"
NPROC="${NPROC:-4}"
CONFIG="${CONFIG:-$HERE/configs/embodied_simwam_navsim_grpo.yaml}"
MASTER_PORT="${MASTER_PORT:-29531}"

# 4-7 were free when this was written. Override if those cards are taken.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
export LUMEN_DIR="${LUMEN_DIR:-/home/danyzhan/Lumen}"
export AITER_SRC="${AITER_SRC:-/home/danyzhan/aiter}"
export LUMEN_ENABLE="${LUMEN_ENABLE:-hf_attn_patch,lumen_linear,aiter_conv}"
export PYTHONPATH="${LUMEN_DIR}:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$SIMWAM/checkpoints}"
export HF_HOME="${HF_HOME:-/home/danyzhan/.cache/huggingface}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-/home/danyzhan/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIRECTORY="${MIOPEN_CUSTOM_CACHE_DIRECTORY:-$MIOPEN_USER_DB_PATH}"
mkdir -p "$MIOPEN_USER_DB_PATH"

mapfile -t CFG_OVERRIDES < <(python3 - "$CONFIG" <<'PY'
import sys
path = sys.argv[1]
for raw in open(path, encoding="utf-8"):
    line = raw.split("#", 1)[0].strip()
    if not line or ":" not in line:
        continue
    key, val = line.split(":", 1)
    key, val = key.strip(), val.strip().strip("'").strip('"')
    if not key or not val or key.startswith("-"):
        continue
    print(f"{key}={val}")
PY
)

cd "$REPO"
exec env NPROC_PER_NODE="$NPROC" MASTER_PORT="$MASTER_PORT" \
    bash "$SIMWAM/scripts/train_navsim_grpo_zero1_torchrun.sh" \
    "${CFG_OVERRIDES[@]}" \
    "$@"
