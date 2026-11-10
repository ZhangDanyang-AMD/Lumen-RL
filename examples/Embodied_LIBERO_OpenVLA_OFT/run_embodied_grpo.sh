#!/usr/bin/env bash
# GRPO for OpenVLA-OFT inside LIBERO (robosuite / MuJoCo). Simulation renders on CPU.
#
# Usage (inside the training container):
#   bash examples/Embodied_LIBERO_OpenVLA_OFT/run_embodied_grpo.sh
#   NPROC=4 bash examples/Embodied_LIBERO_OpenVLA_OFT/run_embodied_grpo.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NPROC="${NPROC:-8}"
CONFIG="${CONFIG:-$HERE/configs/embodied_libero_spatial_grpo.yaml}"
MASTER_PORT="${MASTER_PORT:-29521}"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
# llvmpipe threads per sim process; NPROC x num_workers processes render at once.
export LP_NUM_THREADS="${LP_NUM_THREADS:-2}"
export PYTHONPATH="/work/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m lumenrl.trainer.main --config "$CONFIG" "$@"
