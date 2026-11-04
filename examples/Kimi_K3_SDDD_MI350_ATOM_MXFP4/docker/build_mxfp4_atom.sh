#!/usr/bin/env bash
# Build the Kimi-K3 DSpark MXFP4 training image in three layers:
# AITER (reused when unchanged) -> Lumen -> Lumen-RL. Lumen is expected next to
# this Lumen-RL checkout; set LUMEN_ROOT to use another checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_RL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LUMEN_ROOT="${LUMEN_ROOT:-$(dirname "${LUMEN_RL_ROOT}")/Lumen}"

PINNED_ATOM_IMAGE="rocm/atom-dev@sha256:2f8bd4206ad15d014ae48115eae1ee9f1db83781848a8542de7177cfbd4ac914"
PINNED_LUMEN_REVISION="81927587179b3b1ec45c0311a7e2221b9585e71c"
PINNED_AITER_REVISION="70f7b492b4dc7e9c81c6556d307532b9a30cb3a2"
ATOM_IMAGE="${ATOM_IMAGE:-${PINNED_ATOM_IMAGE}}"
GPU_ARCH="${GPU_ARCH:-gfx950}"
MAX_JOBS="${MAX_JOBS:-64}"

AITER_IMAGE="${AITER_IMAGE:-kimi_k3_dspark_mxfp4_aiter:${GPU_ARCH}}"
LUMEN_IMAGE="${LUMEN_IMAGE:-kimi_k3_dspark_mxfp4_lumen:latest}"
TRAIN_IMAGE="${TRAIN_IMAGE:-kimi_k3_dspark_mxfp4_atom:latest}"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.mxfp4-atom"
AITER_STAGE_END='FROM ${AITER_BASE_IMAGE} AS runtime'
grep -qxF "${AITER_STAGE_END}" "${DOCKERFILE}"
# AITER reuse depends only on the AITER stages; revision, sources, arch and the
# ATOM base are compared through their own labels.
AITER_RECIPE_FINGERPRINT="$(
    awk -v end="${AITER_STAGE_END}" '$0 == end { exit } { print }' "${DOCKERFILE}" \
        | sha256sum | awk '{print $1}'
)"
BUILD_RECIPE_FINGERPRINT="$(
    cat "${DOCKERFILE}" "${BASH_SOURCE[0]}" | sha256sum | awk '{print $1}'
)"

command -v docker >/dev/null
command -v git >/dev/null
command -v python3 >/dev/null
command -v sha256sum >/dev/null
if [[ "${ATOM_IMAGE}" != "${PINNED_ATOM_IMAGE}" ]]; then
    echo "ATOM_IMAGE must remain pinned to ${PINNED_ATOM_IMAGE}" >&2
    exit 2
fi
test -f "${LUMEN_ROOT}/third_party/aiter/setup.py" || {
    echo "Lumen AITER submodule is missing. Run:" >&2
    echo "  git -C ${LUMEN_ROOT} submodule update --init --recursive" >&2
    exit 2
}
test -f "${LUMEN_RL_ROOT}/pyproject.toml"

LUMEN_REVISION="$(git -C "${LUMEN_ROOT}" rev-parse HEAD)"
AITER_REVISION="$(git -C "${LUMEN_ROOT}/third_party/aiter" rev-parse HEAD)"
LUMEN_RL_REVISION="$(git -C "${LUMEN_RL_ROOT}" rev-parse HEAD)"
if [[ "${LUMEN_REVISION}" != "${PINNED_LUMEN_REVISION}" ]]; then
    echo "Lumen must be at ${PINNED_LUMEN_REVISION}; got ${LUMEN_REVISION}. Run:" >&2
    echo "  git -C ${LUMEN_ROOT} checkout ${PINNED_LUMEN_REVISION}" >&2
    echo "  git -C ${LUMEN_ROOT} submodule update --init --recursive" >&2
    exit 2
fi
if [[ "${AITER_REVISION}" != "${PINNED_AITER_REVISION}" ]]; then
    echo "AITER must remain pinned to ${PINNED_AITER_REVISION}; got ${AITER_REVISION}" >&2
    exit 2
fi
AITER_SOURCE_FINGERPRINT="$(
    python3 - "${LUMEN_ROOT}/third_party/aiter" \
        "${LUMEN_ROOT}/third_party/aiter_vendor" <<'PY'
import hashlib
import sys
from pathlib import Path

h = hashlib.sha256()
ignored_parts = {".git", "__pycache__", "build", "dist", ".pytest_cache"}
for root_arg in sys.argv[1:]:
    root = Path(root_arg)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        if any(part in ignored_parts for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".so"}:
            continue
        h.update(root.name.encode())
        h.update(str(relative).encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
print(h.hexdigest())
PY
)"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ -n "$(git -C "${LUMEN_ROOT}" status --porcelain)" ]]; then
    LUMEN_SOURCE_STATE="dirty"
else
    LUMEN_SOURCE_STATE="clean"
fi
if [[ -n "$(git -C "${LUMEN_RL_ROOT}" status --porcelain)" ]]; then
    LUMEN_RL_SOURCE_STATE="dirty"
else
    LUMEN_RL_SOURCE_STATE="clean"
fi
if [[ "${ALLOW_DIRTY:-0}" != "1" ]] && {
    [[ "${LUMEN_SOURCE_STATE}" != "clean" ]] \
        || [[ "${LUMEN_RL_SOURCE_STATE}" != "clean" ]]
}; then
    echo "Refusing to build from dirty Lumen/Lumen-RL sources." >&2
    echo "Commit the changes, or set ALLOW_DIRTY=1 for a non-candidate image." >&2
    exit 2
fi

assert_frozen_sources() {
    local current
    current="$(git -C "${LUMEN_ROOT}" rev-parse HEAD)"
    [[ "${current}" == "${LUMEN_REVISION}" ]] || {
        echo "Lumen HEAD changed during build: ${LUMEN_REVISION} -> ${current}" >&2
        exit 2
    }
    current="$(git -C "${LUMEN_ROOT}/third_party/aiter" rev-parse HEAD)"
    [[ "${current}" == "${AITER_REVISION}" ]] || {
        echo "AITER HEAD changed during build: ${AITER_REVISION} -> ${current}" >&2
        exit 2
    }
    current="$(git -C "${LUMEN_RL_ROOT}" rev-parse HEAD)"
    [[ "${current}" == "${LUMEN_RL_REVISION}" ]] || {
        echo "Lumen-RL HEAD changed during build: ${LUMEN_RL_REVISION} -> ${current}" >&2
        exit 2
    }
    if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
        test -z "$(git -C "${LUMEN_ROOT}" status --porcelain)" || {
            echo "Lumen became dirty during build." >&2
            exit 2
        }
        test -z "$(git -C "${LUMEN_RL_ROOT}" status --porcelain)" || {
            echo "Lumen-RL became dirty during build." >&2
            exit 2
        }
    fi
}

REBUILD_AITER="${FORCE_AITER_REBUILD:-0}"
if ! docker image inspect "${AITER_IMAGE}" >/dev/null 2>&1; then
    REBUILD_AITER=1
elif [[ "${REBUILD_AITER}" != "1" ]]; then
    declare -A EXPECTED_AITER_LABELS=(
        [com.amd.lumen.aiter.build.complete]="true"
        [com.amd.lumen.aiter.revision]="${AITER_REVISION}"
        [com.amd.lumen.aiter.source.fingerprint]="${AITER_SOURCE_FINGERPRINT}"
        [com.amd.lumen.aiter.gpu.arch]="${GPU_ARCH}"
        [com.amd.lumen.aiter.atom.base]="${ATOM_IMAGE}"
        [com.amd.lumen.aiter.build.recipe.fingerprint]="${AITER_RECIPE_FINGERPRINT}"
    )
    for label in "${!EXPECTED_AITER_LABELS[@]}"; do
        actual="$(
            docker image inspect \
                --format "{{ index .Config.Labels \"${label}\" }}" \
                "${AITER_IMAGE}"
        )"
        if [[ "${actual}" != "${EXPECTED_AITER_LABELS[${label}]}" ]]; then
            echo ">>> AITER image label changed: ${label}" \
                "${actual} -> ${EXPECTED_AITER_LABELS[${label}]}"
            REBUILD_AITER=1
            break
        fi
    done
fi

assert_frozen_sources
if [[ "${REBUILD_AITER}" == "1" ]]; then
    echo ">>> Building reusable AITER image: ${AITER_IMAGE}"
    docker build \
        --file "${DOCKERFILE}" \
        --target aiter-build \
        --build-arg "ATOM_IMAGE=${ATOM_IMAGE}" \
        --build-arg "GPU_ARCH=${GPU_ARCH}" \
        --build-arg "MAX_JOBS=${MAX_JOBS}" \
        --build-arg "AITER_REVISION=${AITER_REVISION}" \
        --build-arg "AITER_SOURCE_FINGERPRINT=${AITER_SOURCE_FINGERPRINT}" \
        --build-arg "AITER_BUILD_RECIPE_FINGERPRINT=${AITER_RECIPE_FINGERPRINT}" \
        --tag "${AITER_IMAGE}" \
        "${LUMEN_ROOT}"
else
    echo ">>> Reusing AITER image: ${AITER_IMAGE}"
fi
assert_frozen_sources
AITER_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${AITER_IMAGE}")"

echo ">>> Building Lumen MXFP4 image: ${LUMEN_IMAGE}"
docker build \
    --file "${DOCKERFILE}" \
    --build-arg "ATOM_IMAGE=${ATOM_IMAGE}" \
    --build-arg "AITER_BASE_IMAGE=${AITER_IMAGE}" \
    --build-arg "GPU_ARCH=${GPU_ARCH}" \
    --build-arg "MAX_JOBS=${MAX_JOBS}" \
    --build-arg "LUMEN_REVISION=${LUMEN_REVISION}" \
    --build-arg "AITER_REVISION=${AITER_REVISION}" \
    --build-arg "LUMEN_SOURCE_STATE=${LUMEN_SOURCE_STATE}" \
    --build-arg "BUILD_DATE=${BUILD_DATE}" \
    --build-arg "AITER_BASE_IMAGE_ID=${AITER_IMAGE_ID}" \
    --build-arg "BUILD_RECIPE_FINGERPRINT=${BUILD_RECIPE_FINGERPRINT}" \
    --tag "${LUMEN_IMAGE}" \
    "${LUMEN_ROOT}"
assert_frozen_sources
LUMEN_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${LUMEN_IMAGE}")"

echo ">>> Building Kimi-K3 DSpark MXFP4 training image: ${TRAIN_IMAGE}"
docker build \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --build-arg "MXFP4_LUMEN_IMAGE=${LUMEN_IMAGE}" \
    --build-arg "MXFP4_LUMEN_IMAGE_ID=${LUMEN_IMAGE_ID}" \
    --build-arg "LUMEN_RL_REVISION=${LUMEN_RL_REVISION}" \
    --build-arg "LUMEN_RL_SOURCE_STATE=${LUMEN_RL_SOURCE_STATE}" \
    --build-arg "BUILD_DATE=${BUILD_DATE}" \
    --tag "${TRAIN_IMAGE}" \
    "${LUMEN_RL_ROOT}"
assert_frozen_sources

docker image inspect "${AITER_IMAGE}" "${LUMEN_IMAGE}" "${TRAIN_IMAGE}" \
    >/dev/null
echo ">>> Build complete"
echo "    AITER:    ${AITER_IMAGE}"
echo "    Lumen:    ${LUMEN_IMAGE}"
echo "    Training: ${TRAIN_IMAGE}"
