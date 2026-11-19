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
LUMEN_DIR="${LUMEN_DIR:-/src/Lumen}"
OPENVLA_DIR="$LUMEN_DIR/examples/openvla_oft"
export PYTHONPATH="$LUMEN_DIR:$OPENVLA_DIR:/work/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
# 512 sim workers exceed the default soft nofile of 1024.
ulimit -n 65536
# OpenVLA BF16 GEMM table. Shapes that are absent fall back to torch.mm.
TUNED_GEMM="$OPENVLA_DIR/openvla_bf16_tuned_gemm.csv"
if [[ -f "$TUNED_GEMM" ]]; then
  export AITER_CONFIG_GEMM_BF16="$TUNED_GEMM"
fi
# The tuned hipBLASLt kernels can return HIPBLAS_STATUS_INTERNAL_ERROR and
# leave the output unwritten. Opus covers the vision and language shapes.
export LUMEN_BF16_GEMM="${LUMEN_BF16_GEMM:-opus}"
# embodied.env.backend=hipphysics_libero: HP_BUILD is a libero-flavour hipPhysics build
# (cmake -DHP_FLAVOR=libero) and HP_REPO the tree it was built from.
if [[ -n "${HP_BUILD:-}" ]]; then
  export HP_REPO="${HP_REPO:?HP_BUILD needs HP_REPO}"
  export PYTHONPATH="$HP_BUILD/python:$PYTHONPATH"
  export LD_LIBRARY_PATH="$HP_BUILD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$REPO"
exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m lumenrl.trainer.main --config "$CONFIG" "$@"
