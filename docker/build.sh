#!/usr/bin/env bash
# Build LumenRL Docker images.
#
# Usage:
#   bash docker/build.sh              # production image
#   bash docker/build.sh --dev        # development image (with test/lint tools)
#   bash docker/build.sh --base IMG   # override base ROCm image
#
# Prerequisites:
#   git submodule update --init --recursive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEV_MODE=false
BASE_IMAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev) DEV_MODE=true; shift ;;
        --base) BASE_IMAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "${REPO_ROOT}"

if [ ! -d "third_party/Lumen/lumen" ]; then
    echo "ERROR: Lumen submodule not initialized. Run:"
    echo "  git submodule update --init --recursive"
    exit 1
fi
if [ ! -d "third_party/verl/verl" ]; then
    echo "ERROR: VERL submodule not initialized. Run:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

if [ "${DEV_MODE}" = true ]; then
    DOCKERFILE="docker/Dockerfile.dev"
    TAG="lumenrl:dev"
else
    DOCKERFILE="docker/Dockerfile"
    TAG="lumenrl:latest"
fi

BUILD_ARGS=""
if [ -n "${BASE_IMAGE}" ]; then
    BUILD_ARGS="--build-arg BASE_IMAGE=${BASE_IMAGE}"
fi

echo "Building ${TAG} from ${DOCKERFILE}..."
docker build \
    -f "${DOCKERFILE}" \
    ${BUILD_ARGS} \
    -t "${TAG}" \
    .

echo "Done. Run with:"
echo "  docker run --rm -it --device /dev/kfd --device /dev/dri -v /dev/shm:/dev/shm --shm-size=256g ${TAG}"
