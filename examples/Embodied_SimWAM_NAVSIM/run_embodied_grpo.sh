#!/usr/bin/env bash
# FlowGRPO for SimWAM on NAVSIM, same launch shape as the embodied GRPO scripts.
#
# Lumen is applied from this directory (plugin.py + train_grpo.py). The SimWAM
# checkout under examples/SimWAM stays unmodified.
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
# an aiter checkout with aiter/ops/triton/conv/conv3d.py (the image's aiter has none); unset, aiter_conv logs a
# warning and Conv3d (the DiT patch embedding) stays on torch
export AITER_SRC="${AITER_SRC:-}"
# aiter's tuned bf16 GEMM table for this example's shapes (SIMWAM_GRPO_AITER_GEMM), from the same checkout
_GEMM_TABLE="$AITER_SRC/aiter/configs/model_configs/simwam_grpo_bf16_tuned_gemm.csv"
if [ -z "${AITER_CONFIG_GEMM_BF16:-}" ] && [ -n "$AITER_SRC" ] && [ -f "$_GEMM_TABLE" ]; then
  export AITER_CONFIG_GEMM_BF16="$_GEMM_TABLE"
fi
# lumen_linear is left out: the tuned GEMM has no entry for these shapes and falls back to the torch
# GEMM, so results are unchanged and only the dispatch cost goes. hf_attn_patch stays so results are
# bit-identical to the original; aiter_conv alone (torch SDPA) is faster but changes log-probs (see README).
# flydsl_conv moves the VAE convolutions to aiter's FlyDSL kernel when it is available (see README);
# otherwise it logs a warning and the VAE keeps the aiter_conv / torch path. With both, the VAE runs on
# FlyDSL and the DiT patch embedding on aiter_conv's Triton conv3d (from $AITER_SRC).
export LUMEN_ENABLE="${LUMEN_ENABLE:-hf_attn_patch,aiter_conv,flydsl_conv}"
export SIMWAM_ROOT="$SIMWAM"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$SIMWAM/checkpoints}"
export HF_HOME="${HF_HOME:-/home/danyzhan/.cache/huggingface}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-/home/danyzhan/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIRECTORY="${MIOPEN_CUSTOM_CACHE_DIRECTORY:-$MIOPEN_USER_DB_PATH}"
mkdir -p "$MIOPEN_USER_DB_PATH"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${SIMWAM}/data/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${SIMWAM}/runs}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${SIMWAM}/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${SIMWAM}/data/navsim}"
export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/trainval}"
export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval}"
export NAVSIM_METRIC_CACHE_PATH="${NAVSIM_METRIC_CACHE_PATH:-${SIMWAM}/data/metric_cache_navtrain}"
export PYTHONPATH="${LUMEN_DIR}:${REPO}:${SIMWAM}/src:${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${SIMWAM}/scripts/ds_configs/ds_zero1_config.json"

for required_dir in \
  "${NUPLAN_MAPS_ROOT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim" \
  "${NAVSIM_LOG_PATH}" \
  "${NAVSIM_SENSOR_BLOBS_PATH}" \
  "${NAVSIM_METRIC_CACHE_PATH}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Error: required directory does not exist: ${required_dir}" >&2
    exit 1
  fi
done

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

TASK_BASENAME="train_grpo"
for arg in "${CFG_OVERRIDES[@]}" "$@"; do
  case "${arg}" in
    task=*)
      cfg="${arg#task=}"
      TASK_BASENAME="${cfg%.yaml}"
      ;;
  esac
done
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

cd "$SIMWAM"
exec torchrun \
  --nnodes 1 \
  --nproc_per_node "$NPROC" \
  --master_port "$MASTER_PORT" \
  "$HERE/train_grpo.py" \
  "output_dir=./runs/grpo/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${CFG_OVERRIDES[@]}" \
  "$@"
