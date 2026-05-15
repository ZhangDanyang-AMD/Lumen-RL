#!/usr/bin/env bash
# Build lumenrl-vllm-mi350 Docker image for Qwen3-8B Eagle3 SDDD.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "Building lumenrl-vllm-mi350:latest ..."
echo "  Repo root:  ${REPO_ROOT}"

docker buildx build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t lumenrl-vllm-mi350:latest \
    "${REPO_ROOT}"
