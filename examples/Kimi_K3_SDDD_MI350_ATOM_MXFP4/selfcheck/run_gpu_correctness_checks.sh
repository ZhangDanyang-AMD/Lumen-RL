#!/usr/bin/env bash
# Reuse the validated BF16 DSpark GPU correctness checks on an MXFP4 training image.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="${REPO_ROOT}/examples/Kimi_K3_SDDD_MI350_ATOM/selfcheck"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python3 "${BASE}/verify_rope_alignment.py" --device cuda
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python3 "${BASE}/verify_flex_attention.py"

echo "RoPE/YaRN and Flex Attention GPU correctness checks passed."
