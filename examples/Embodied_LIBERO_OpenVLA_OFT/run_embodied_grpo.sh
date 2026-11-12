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
export PYTHONPATH="/src/Lumen:/work/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
# 512 sim workers exceed the default soft nofile of 1024.
ulimit -n 65536
# OpenVLA BF16 GEMM table. Shapes that are absent fall back to torch.mm.
TUNED_GEMM="$HERE/openvla_bf16_tuned_gemm.csv"
if [[ -f "$TUNED_GEMM" ]]; then
  export AITER_CONFIG_GEMM_BF16="$TUNED_GEMM"
fi

cd "$REPO"
exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m lumenrl.trainer.main --config "$CONFIG" "$@"
