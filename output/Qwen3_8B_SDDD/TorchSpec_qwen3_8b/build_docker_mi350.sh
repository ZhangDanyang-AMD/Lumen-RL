#!/usr/bin/env bash
# Build the TorchSpec + vLLM Docker image for MI350 (ROCm gfx950)
#
# Prerequisites:
#   - Docker with ROCm support
#   - TorchSpec repo at $TORCHSPEC_DIR
#
# Usage:
#   bash build_docker_mi350.sh
#   TORCHSPEC_DIR=/path/to/TorchSpec bash build_docker_mi350.sh
set -euo pipefail

TORCHSPEC_DIR="${TORCHSPEC_DIR:-/home/danyzhan/TorchSpec}"
IMAGE_NAME="${IMAGE_NAME:-torchspec-vllm-mi350:latest}"

echo "Building ${IMAGE_NAME} from ${TORCHSPEC_DIR}..."

docker build \
    -f "${TORCHSPEC_DIR}/docker/vllm/rocm_mi350/Dockerfile" \
    -t "${IMAGE_NAME}" \
    "${TORCHSPEC_DIR}"

echo "Done: ${IMAGE_NAME}"
