#!/usr/bin/env bash
# Isaac Lab 2.3 G1 rough-terrain velocity benchmark (4096 envs, one GPU).
#
# Run inside the wm-rl container, which has the ROCm torch and the hipPhysics build:
#   docker exec wm-rl bash -lc 'bash /src/Lumen-RL/examples/Embodied_IsaacLab_Benchmark/run_bench_g1.sh'
#
# The rough terrain is one heightfield: HP_HFIELD_COLLISION=1 turns on the hipPhysics heightfield collider (P6).
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
export HP_HFIELD_COLLISION=1

export PYTHONPATH="$HERE:$HP_REPO/python:$HP_BUILD/python${PYTHONPATH:+:$PYTHONPATH}"

# "train" runs the full training through Lumen-RL's entry point; anything else is bench.py (three FPS columns).
CONFIG="${CONFIG:-$HERE/configs/g1_rough.yaml}"
if [[ "${1:-}" == "train" ]]; then
  shift
  exec python -m lumenrl.trainer.main --config "$CONFIG" "$@"
fi
exec python "$HERE/bench.py" --config "$CONFIG" "$@"
