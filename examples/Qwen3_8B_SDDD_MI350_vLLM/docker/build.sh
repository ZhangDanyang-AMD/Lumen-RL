#!/usr/bin/env bash
# Build lumenrl-vllm-mi350 Docker image for Qwen3-8B Eagle3 SDDD.
# TorchSpec is included via docker buildx additional build context.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TORCHSPEC_DIR="${TORCHSPEC_DIR:-/home/danyzhan/TorchSpec}"

echo "Building lumenrl-vllm-mi350:latest ..."
echo "  Repo root:  ${REPO_ROOT}"
echo "  TorchSpec:  ${TORCHSPEC_DIR}"

docker buildx build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --build-context torchspec="${TORCHSPEC_DIR}" \
    -t lumenrl-vllm-mi350:latest \
    "${REPO_ROOT}"
