#!/usr/bin/env bash
# Build the LumenRL release image from release/versions.env.
#
#   bash release/build_image.sh                    # full image, incl. example 7
#   WITH_MEGATRON=0 bash release/build_image.sh    # skip Megatron/Apex/TE
#
# Run from the Lumen-RL repo root. Expect 60-90 min for the full build; the long
# pole is TransformerEngine's AOTriton step.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$HERE/.." && pwd)
# shellcheck disable=SC1091
source "$HERE/versions.env"

TAG=${TAG:-lumenrl:release-$(date +%Y%m%d)}
ARCH=${PYTORCH_ROCM_ARCH:-gfx950}
WITH_MEGATRON=${WITH_MEGATRON:-1}

# Resolve the branch to the SHA it points at right now, and hand the Dockerfile
# that instead of the branch name. Not a change of policy -- the image still
# tracks the tip -- but the layer that clones Lumen-RL is cached on the build
# args, so passing a branch name means a second build on the same machine
# silently re-uses the clone from the first one and ships whatever the tip was
# back then. GitHub serves a --depth 1 fetch of a reachable SHA, which is how
# the three upstreams below are already taken.
LUMENRL_REF=$(git ls-remote "$LUMENRL_REPO" "refs/heads/$LUMENRL_BRANCH" | cut -f1)
[ -n "$LUMENRL_REF" ] || { echo "cannot resolve $LUMENRL_BRANCH on $LUMENRL_REPO" >&2; exit 1; }

echo "==> building $TAG"
echo "    base        $BASE_IMAGE"
echo "    arch        $ARCH"
echo "    aiter       $AITER_BRANCH @ $AITER_SHA"
echo "    Lumen       $LUMEN_BRANCH @ $LUMEN_SHA"
echo "    ATOM        $ATOM_BRANCH @ $ATOM_SHA"
echo "    Lumen-RL    $LUMENRL_BRANCH @ $LUMENRL_REF (tip as of now, not pinned)"
echo "    megatron    $WITH_MEGATRON"

# Context is release/ itself, not the repo root: the image clones its own copies
# of all four repos at pinned SHAs, so nothing else needs to be sent.
cd "$HERE"
docker build -f Dockerfile -t "$TAG" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg PYTORCH_ROCM_ARCH="$ARCH" \
  --build-arg LUMENRL_REPO="$LUMENRL_REPO" \
  --build-arg LUMENRL_BRANCH="$LUMENRL_REF" \
  --build-arg LUMEN_REPO="$LUMEN_REPO" \
  --build-arg LUMEN_BRANCH="$LUMEN_BRANCH" \
  --build-arg LUMEN_SHA="$LUMEN_SHA" \
  --build-arg AITER_REPO="$AITER_REPO" \
  --build-arg AITER_BRANCH="$AITER_BRANCH" \
  --build-arg AITER_SHA="$AITER_SHA" \
  --build-arg ATOM_REPO="$ATOM_REPO" \
  --build-arg ATOM_BRANCH="$ATOM_BRANCH" \
  --build-arg ATOM_SHA="$ATOM_SHA" \
  --build-arg FLYDSL_VERSION="$FLYDSL_VERSION" \
  --build-arg MEGATRON_CORE_VERSION="$MEGATRON_CORE_VERSION" \
  --build-arg WITH_MEGATRON="$WITH_MEGATRON" \
  .

echo
echo "==> built $TAG"
echo "    Next, bake the aiter JIT kernels (needs GPUs, ~25 min once):"
echo "      TAG=$TAG bash release/precompile_kernels.sh"
