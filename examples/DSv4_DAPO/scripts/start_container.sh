#!/usr/bin/env bash
# (Re)create the training container on THIS host and check that it sees the GPUs.
#
# Only a container that this example created (it carries the label below) is ever
# removed; any other container with the same name is an error.
#
# Differences from the container the recipe was validated in, none of which touch
# the numerics: not --privileged (explicit devices instead), no host RDMA libraries
# mounted (this recipe runs NCCL over TCP, NCCL_IB_DISABLE=1), and core dumps off (a
# GPU fault otherwise writes tens of GB into the working directory). The flags are
# TRAIN_DOCKER_FLAGS in lib.sh; EXTRA_DOCKER_ARGS is appended verbatim.
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
image_present "$IMAGE" || die "IMAGE not pulled; run scripts/pull_images.sh"

if dk inspect "$CONTAINER" >/dev/null 2>&1; then
  if [ "$(dk inspect -f '{{index .Config.Labels "org.lumenrl.example"}}' "$CONTAINER")" != dsv4-dapo ]; then
    die "a container named '$CONTAINER' exists and was not created by this example; set CONTAINER to another name"
  fi
  log "removing the previous $CONTAINER"
  dk rm -f "$CONTAINER" >/dev/null
fi

mapfile -t mounts < <(container_mounts)
read -r -a extra <<<"${EXTRA_DOCKER_ARGS:-}"
dk run -d --name "$CONTAINER" --label "$LABEL" --entrypoint /bin/bash \
  "${TRAIN_DOCKER_FLAGS[@]}" \
  "${mounts[@]}" \
  -e RL_ROOT="$RL_ROOT" -e DATA_ROOT="$DATA_ROOT" -e SCRATCH_ROOT="$WORK_DIR" \
  -e HF_HOME="$DATA_ROOT/hf_home" \
  -e LUMEN_DIR="$RL_ROOT/Lumen" -e AITER_DIR="$RL_ROOT/aiter" -e ATOM_DIR="$RL_ROOT/ATOM" \
  "${extra[@]}" \
  "$IMAGE" -lc "sleep infinity" >/dev/null

n=$(dk exec "$CONTAINER" python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1)
log "container $CONTAINER up on $(uname -n): torch sees ${n:-0} GPU(s)"
probe_gpus
[ "${n:-0}" = "$GPU_COUNT" ] || die "torch in the container sees ${n:-0} GPUs, rocm-smi sees $GPU_COUNT"
