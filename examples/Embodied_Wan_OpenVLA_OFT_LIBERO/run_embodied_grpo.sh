#!/usr/bin/env bash
# GRPO for OpenVLA-OFT inside the Wan action-conditioned world model (LIBERO-Spatial).
#
# Usage:
#   bash examples/Embodied_Wan_OpenVLA_OFT_LIBERO/run_embodied_grpo.sh [extra key=value overrides...]
#   NPROC=8 CONFIG=path/to.yaml bash examples/Embodied_Wan_OpenVLA_OFT_LIBERO/run_embodied_grpo.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NPROC="${NPROC:-8}"
CONFIG="${CONFIG:-$HERE/configs/embodied_wan_openvla_oft_grpo.yaml}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "$REPO"
exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m lumenrl.trainer.main --config "$CONFIG" "$@"
