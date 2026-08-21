#!/usr/bin/env bash
# Build vLLM from source against primus's own torch, inside a throwaway container.
#
# Why source and not `docker cp` a donor tree (2026-08-10):
#   RDMA needs RCCL 2.28.9 (24.04 + 2.27.7 was tested clean and still SIGSEGVs),
#   and 2.28.9 only exists in rocm/primus:v26.4, whose torch is 2.12.0. Every
#   prebuilt *stable* vLLM 0.26.0 on this box is torch 2.11 / 22.04 / RCCL 2.27.7,
#   so copying one would drag 2.27.7 in and break RDMA. Building here links the
#   extensions against the exact torch + ROCm that will run them.
#
# Runs on the docker host. The build container is separate from anp-primus so the
# RDMA-verified container is never touched.
#
#   bash ~/4node/build_vllm_primus.sh            # full build
#   STAGE=deps bash ~/4node/build_vllm_primus.sh # just (re)install build deps
set -uo pipefail

IMAGE=${IMAGE:-rocm/primus:v26.4}
NAME=${NAME:-primus-build}
SRC=${SRC:-/mnt/m2m_nobackup/xysheng/vllm-build/vllm}
OUT=${OUT:-/home/xysheng/vllm_primus}          # NFS, so both nodes see the wheel
ARCH=${ARCH:-gfx950}                            # MI350X only; gfx942 would double the build
JOBS=${JOBS:-96}
STAGE=${STAGE:-all}

mkdir -p "$OUT/wheels"

if ! docker inspect "$NAME" >/dev/null 2>&1; then
  echo "=== starting build container $NAME from $IMAGE"
  docker run -d --name "$NAME" --entrypoint /bin/bash \
    --network=host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add=video \
    --shm-size 16G \
    -v /home/xysheng:/home/xysheng \
    -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
    "$IMAGE" -lc "sleep infinity" >/dev/null
fi

if [ "$STAGE" = deps ] || [ "$STAGE" = all ]; then
  # setuptools-scm shells out to git for the version, and the checkout is owned by
  # the host user while the container runs as root -> "detected dubious ownership"
  # aborts metadata generation before a single file compiles.
  docker exec "$NAME" bash -lc "git config --global --add safe.directory $SRC"

  echo "=== stripping torch pins (use_existing_torch.py) so pip keeps primus's 2.12"
  docker exec "$NAME" bash -lc "cd $SRC && git checkout -- requirements pyproject.toml 2>/dev/null; python3 use_existing_torch.py --prefix"

  echo "=== build deps (torch/triton/vision/audio deliberately excluded)"
  docker exec "$NAME" bash -lc "
    pip install --no-cache-dir -q \
      'cmake>=3.26.1,<4' ninja 'packaging>=24.2' \
      'setuptools>=77.0.3,<80.0.0' 'setuptools-scm>=8' 'setuptools-rust>=1.9.0' \
      wheel 'jinja2>=3.1.6' regex 2>&1 | tail -5
    python3 -c 'import setuptools,setuptools_scm,setuptools_rust,jinja2;print(\"build deps OK, setuptools\",setuptools.__version__)'
  " || exit 1
fi

if [ "$STAGE" = all ]; then
  echo "=== building wheel  ARCH=$ARCH  JOBS=$JOBS   (this is the long pole)"
  date -Is
  docker exec "$NAME" bash -lc "
    cd $SRC
    export VLLM_TARGET_DEVICE=rocm
    export PYTORCH_ROCM_ARCH=$ARCH
    export MAX_JOBS=$JOBS
    # --depth 1 leaves a grafted history, so setuptools-scm cannot see that HEAD IS
    # the tag and reports the next dev version (0.26.1.dev0). The source is exactly
    # v0.26.0; pin the string so nobody later reads this as a dev build.
    export SETUPTOOLS_SCM_PRETEND_VERSION=${PRETEND_VERSION:-0.26.0}
    export CMAKE_BUILD_TYPE=Release
    export VLLM_FA_CMAKE_GPU_ARCHES=$ARCH
    python3 -m pip wheel --no-build-isolation --no-deps -w $OUT/wheels . 2>&1
  "
  rc=$?
  date -Is
  echo "=== build exit=$rc"
  ls -la "$OUT/wheels/" 2>/dev/null
  exit $rc
fi
