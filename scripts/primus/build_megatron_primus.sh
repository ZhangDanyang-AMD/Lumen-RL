#!/usr/bin/env bash
# Build ROCm Apex + ROCm TransformerEngine against primus's torch 2.12, then put
# them in the same NFS tree as vLLM so both nodes see one copy.
#
# Same shape as build_vllm_primus.sh, and for the same reason: the pinned
# revisions (§5 of primus-24.04-rdma-dsv4-handoff.md) were verified on torch
# 2.9/2.10, so the extensions have to be compiled against the torch that will
# actually run them rather than copied from a donor image.
#
# Building happens in a throwaway `primus-build` container so the RDMA-verified
# anp-primus is never touched.
#
#   bash ~/4node/build_megatron_primus.sh              # apex + TE
#   STAGE=apex bash ~/4node/build_megatron_primus.sh   # one at a time
#   STAGE=te   bash ~/4node/build_megatron_primus.sh
set -uo pipefail

IMAGE=${IMAGE:-rocm/primus:v26.4}
NAME=${NAME:-primus-build}
SRCROOT=${SRCROOT:-/mnt/m2m_nobackup/xysheng/megatron_build}   # node-local: 2.8 G of submodules
OUT=${OUT:-/home/xysheng/vllm_primus}                          # NFS, shared with the vLLM tree
SITE=$OUT/site
ARCH=${ARCH:-gfx950}
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

docker exec "$NAME" bash -lc "git config --global --add safe.directory '*'"
docker exec "$NAME" bash -lc "pip install --no-cache-dir -q wheel 'setuptools>=77' ninja 2>&1 | tail -2"

if [ "$STAGE" = apex ] || [ "$STAGE" = all ]; then
  echo "=== apex wheel   ARCH=$ARCH  JOBS=$JOBS"
  date -Is
  docker exec "$NAME" bash -lc "
    cd $SRCROOT/apex_src
    rm -rf build dist apex.egg-info
    export PYTORCH_ROCM_ARCH=$ARCH MAX_JOBS=$JOBS TMPDIR=$SRCROOT/tmp
    mkdir -p \$TMPDIR
    # This revision IGNORES --cpp_ext/--cuda_ext -- setup.py strips them from
    # sys.argv and reads APEX_BUILD_CPP_OPS / APEX_BUILD_CUDA_OPS instead, both
    # defaulting to 0. Passing only the flags (as older notes say) produces a
    # wheel with zero .so files and installs the compatibility/ JIT stubs
    # (amp_C.py, fused_layer_norm_cuda.py, ...) which rebuild each op inside
    # every worker on first import.
    export APEX_BUILD_CPP_OPS=1 APEX_BUILD_CUDA_OPS=1
    python3 setup.py bdist_wheel --cpp_ext --cuda_ext 2>&1 | tail -30
    cp dist/apex-*.whl $OUT/wheels/
  "
  echo "=== apex exit=$?"
  date -Is
fi

if [ "$STAGE" = te ] || [ "$STAGE" = all ]; then
  echo "=== transformer-engine wheel   ARCH=$ARCH  JOBS=$JOBS"
  date -Is
  docker exec "$NAME" bash -lc "
    cd $SRCROOT/te_src
    rm -rf build dist transformer_engine.egg-info
    export TMPDIR=$SRCROOT/tmp
    mkdir -p \$TMPDIR
    export NVTE_FRAMEWORK=pytorch NVTE_USE_ROCM=1
    export NVTE_ROCM_ARCH=$ARCH PYTORCH_ROCM_ARCH=$ARCH
    export NVTE_FUSED_ATTN=1 NVTE_FUSED_ATTN_CK=1 NVTE_FUSED_ATTN_AOTRITON=1
    export MAX_JOBS=$JOBS
    # ROCm's \`hipcc -v\` exits 1 with no input file and CK-JIT's compiler-ABI
    # probe reads that as an unusable compiler.
    export TORCH_DONT_CHECK_COMPILER_ABI=1
    python3 -m pip wheel --no-build-isolation --no-deps -w $OUT/wheels . 2>&1 | tail -40
  "
  echo "=== te exit=$?"
  date -Is
fi

echo "=== wheels in $OUT/wheels"
ls -la "$OUT/wheels/"
echo
echo "next: bash ~/4node/install_megatron_primus.sh"
