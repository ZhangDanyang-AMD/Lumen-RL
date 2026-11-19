#!/usr/bin/env bash
# Isaac Lab 2.3 Cartpole-Direct throughput benchmark (4096 envs, one GPU).
#
# Run inside the wm-rl container, which has the ROCm torch and the hipPhysics build:
#   docker exec wm-rl bash -lc 'bash /src/Lumen-RL/examples/Embodied_IsaacLab_Benchmark/run_bench_cartpole.sh'
# RGB camera: CONFIG=$PWD/configs/cartpole_rgb.yaml, renderer override vec_env.env_kwargs.renderer=hipphysics|hiprt.
# Full training instead of the benchmark: run_bench_cartpole.sh train
#
# HP_REPO is a hipPhysics tree whose python/hipphysics_lab symlink resolves (the container
# archive at /work/hipphysics/src). HP_BUILD is the directory that contains libhipgl.so and
# python/hipphysics*.so. Cartpole has no contacts, so the libero flavour build is enough.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HP_REPO="${HP_REPO:-/work/hipphysics/src}"
HP_BUILD="${HP_BUILD:-/work/hipphysics/build-libero}"

if [[ ! -e "$HP_REPO/python/hipphysics_lab" ]]; then
  echo "HP_REPO=$HP_REPO has no python/hipphysics_lab" >&2
  exit 1
fi
if [[ ! -d /opt/rocm/core-7.14/lib ]]; then
  echo "ROCm 7.14 lib dir is missing; run this inside the wm-rl container" >&2
  exit 1
fi

export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:$HP_BUILD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# hipRT_render (CONFIG=configs/cartpole_rgb.yaml vec_env.env_kwargs.renderer=hiprt): HIPRT dlopens the unversioned libamdhip64.so / libhiprtc.so
# first. torch's bundled ROCm (_rocm_sdk_core) only ships the .so.7 names, so that dlopen would load a second HIP
# runtime from /opt/rocm, in which HIPRT's kernels are invisible to torch's ("named symbol not found"). Unversioned
# links to torch's copies, searched first, keep one runtime.
TORCH_ROCM_LIB="$(python -c 'import os, _rocm_sdk_core; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib"))' 2>/dev/null || true)"
if [[ -n "$TORCH_ROCM_LIB" && -e "$TORCH_ROCM_LIB/libamdhip64.so.7" ]]; then
  LINKS="${TMPDIR:-/tmp}/torch_rocm_unversioned"
  mkdir -p "$LINKS"
  ln -sf "$TORCH_ROCM_LIB/libamdhip64.so.7" "$LINKS/libamdhip64.so"
  ln -sf "$TORCH_ROCM_LIB/libhiprtc.so.7" "$LINKS/libhiprtc.so"
  export LD_LIBRARY_PATH="$LINKS:$LD_LIBRARY_PATH"
fi

export PYTHONPATH="$HERE:$HP_REPO/python:$HP_BUILD/python${PYTHONPATH:+:$PYTHONPATH}"

# "train" runs the full training through Lumen-RL's entry point; anything else is bench.py (three FPS columns).
CONFIG="${CONFIG:-$HERE/configs/cartpole.yaml}"
if [[ "${1:-}" == "train" ]]; then
  shift
  exec python -m lumenrl.trainer.main --config "$CONFIG" "$@"
fi
exec python "$HERE/bench.py" --config "$CONFIG" "$@"
