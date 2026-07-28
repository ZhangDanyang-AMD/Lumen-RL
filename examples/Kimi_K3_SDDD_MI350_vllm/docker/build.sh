#!/usr/bin/env bash
# Build the LumenRL + vLLM Docker image for Kimi K3 DSpark SDDD (MI350, ROCm gfx950)
#
# Usage:
#   bash examples/Kimi_K3_SDDD_MI350_vllm/docker/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMENRL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-kimi_k3_dspark_vllm_train:latest}"

echo "Building ${IMAGE_NAME}..."
echo "  LumenRL:   ${LUMENRL_DIR}"
echo "  third_party/aiter, ATOM, Lumen: from ${LUMENRL_DIR}/third_party/"

docker buildx build \
    -f "${SCRIPT_DIR}/../Dockerfile.train" \
    -t "${IMAGE_NAME}" \
    "${LUMENRL_DIR}"

echo "Done: ${IMAGE_NAME}"
