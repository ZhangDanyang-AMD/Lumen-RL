#!/usr/bin/env bash
# Build the LumenRL + vLLM + ATOM Docker image for MI350 (ROCm gfx950)
#
# Usage:
#   bash examples/Kimi_K25_SDDD_MI350_ATOM/docker/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMENRL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-lumenrl-vllm-mi350:latest}"

echo "Building ${IMAGE_NAME}..."
echo "  LumenRL:   ${LUMENRL_DIR}"
echo "  third_party/aiter, ATOM, Lumen, mori: from ${LUMENRL_DIR}/third_party/"

docker buildx build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_NAME}" \
    "${LUMENRL_DIR}"

echo "Done: ${IMAGE_NAME}"
