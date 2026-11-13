#!/usr/bin/env bash
# NAVSIM navtest inference for a merged FlowGRPO checkpoint, then the caller
# scores the saved trajectories with the official PDM weights (5, 5, 2).
#
# Run inside the container, from any cwd:
#   CKPT=/path/to/step_020000.pt \
#     bash examples/Embodied_SimWAM_NAVSIM/run_navtest_eval.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SIMWAM="${SIMWAM_ROOT:-$REPO/examples/SimWAM}"
CKPT="${CKPT:?Set CKPT to a LoRA-merged checkpoint, e.g. step_020000.pt}"
OUT="${OUT:-$SIMWAM/evaluate_results/navtest}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-27891}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
export LUMEN_DIR="${LUMEN_DIR:-/home/danyzhan/Lumen}"
export AITER_SRC="${AITER_SRC:-/home/danyzhan/aiter}"
export LUMEN_ENABLE="${LUMEN_ENABLE:-hf_attn_patch,lumen_linear,aiter_conv}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$SIMWAM/checkpoints}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$SIMWAM/data/navsim/maps}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$SIMWAM/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$SIMWAM/data/navsim}"
export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-$OPENSCENE_DATA_ROOT/navsim_logs/test}"
export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-$OPENSCENE_DATA_ROOT/sensor_blobs/test}"
export NAVSIM_SPLIT=navtest
export PYTHONPATH="${LUMEN_DIR}:${REPO}:${SIMWAM}/src:${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
mkdir -p /work/tmp "$OUT"

EVAL_PY="$SIMWAM/experiments/navsim/eval_navsim.py"
if ! grep -q "Lumen patches enabled" "$EVAL_PY"; then
  git -C "$SIMWAM" apply "$HERE/patches/eval_navsim_lumen.patch"
fi

cd "$SIMWAM"
torchrun \
  --nproc_per_node "$NPROC" \
  --nnodes 1 \
  --master_port "$MASTER_PORT" \
  experiments/navsim/eval_navsim.py \
  task=navsim_grpo_action_pdm_384x672_flowgrpo_lora \
  "ckpt=${CKPT}" \
  experiment_name=navtest \
  model.redirect_common_files=false \
  '~model.checkpoint_path' \
  data.train.video_frame_mode=current_only \
  data.train.num_frames=1 \
  EVALUATION.save_videos=false \
  EVALUATION.save_gt_video=false \
  EVALUATION.num_inference_steps=10 \
  "EVALUATION.output_dir=${OUT}" \
  "$@"
