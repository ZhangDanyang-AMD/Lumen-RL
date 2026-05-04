#!/usr/bin/env bash
# Build the LumenRL + vLLM + ATOM Docker image for MI350 (ROCm gfx950)
#
# Uses docker buildx with additional build context for TorchSpec.
#
# Usage:
#   bash examples/Kimi_K25_SDDD_MI350_vLLM/docker/build.sh
#   TORCHSPEC_DIR=/path/to/TorchSpec bash examples/Kimi_K25_SDDD_MI350_vLLM/docker/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMENRL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TORCHSPEC_DIR="${TORCHSPEC_DIR:-/home/danyzhan/TorchSpec}"
IMAGE_NAME="${IMAGE_NAME:-lumenrl-vllm-mi350:latest}"

if [ ! -d "${TORCHSPEC_DIR}" ]; then
    echo "ERROR: TorchSpec directory not found: ${TORCHSPEC_DIR}" >&2
    echo "Set TORCHSPEC_DIR to the correct path." >&2
    exit 1
fi

echo "Building ${IMAGE_NAME}..."
echo "  LumenRL:   ${LUMENRL_DIR}"
echo "  TorchSpec: ${TORCHSPEC_DIR}"
echo "  third_party/aiter, ATOM, Lumen, mori: from ${LUMENRL_DIR}/third_party/"

docker buildx build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --build-context torchspec="${TORCHSPEC_DIR}" \
    -t "${IMAGE_NAME}" \
    "${LUMENRL_DIR}"

echo "Done: ${IMAGE_NAME}"
