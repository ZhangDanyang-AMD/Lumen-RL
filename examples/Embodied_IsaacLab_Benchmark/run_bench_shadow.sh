#!/usr/bin/env bash
# Isaac Lab 2.3 Shadow Hand in-hand repose throughput benchmark (8192 envs, one GPU).
#
# Run inside the wm-rl container, which has the ROCm torch and the hipPhysics build:
#   docker exec wm-rl bash -lc 'bash /src/Lumen-RL/examples/Embodied_IsaacLab_Benchmark/run_bench_shadow.sh'
#
# HP_REPO is a hipPhysics tree whose python/hipphysics_lab symlink resolves. HP_BUILD is the directory that contains
# libhipgl.so and python/hipphysics*.so. The Shadow hand has cylinder and capsule colliders, which the lab build
# (MJCCD_MAX_ITERATIONS 50) is for; set HP_BUILD to the lab build when the libero build refuses the model.
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

export PYTHONPATH="$HERE:$HP_REPO/python:$HP_BUILD/python${PYTHONPATH:+:$PYTHONPATH}"

# "train" runs the full training through Lumen-RL's entry point; anything else is bench.py (three FPS columns).
CONFIG="${CONFIG:-$HERE/configs/shadow.yaml}"
if [[ "${1:-}" == "train" ]]; then
  shift
  exec python -m lumenrl.trainer.main --config "$CONFIG" "$@"
fi
exec python "$HERE/bench.py" --config "$CONFIG" "$@"
