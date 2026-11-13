#!/usr/bin/env bash
# Pull the two pinned images on this host. Both references carry a digest, so what
# lands is exactly what was validated.
#
#   bash scripts/pull_images.sh            # IMAGE and MILES_IMAGE
#   bash scripts/pull_images.sh --runtime  # IMAGE only (a node that only trains)
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env

images=("$IMAGE")
[ "${1:-}" = --runtime ] || images+=("$MILES_IMAGE")

for img in "${images[@]}"; do
  if image_present "$img"; then log "present: $img"; continue; fi
  bytes=$IMAGE_DISK_BYTES
  [ "$img" = "$MILES_IMAGE" ] && bytes=$MILES_IMAGE_DISK_BYTES
  free=$(docker_free_bytes)
  if [ -n "$free" ] && [ "$free" -lt "$bytes" ]; then
    die "Docker Root Dir has $(human_gb "$free") free, $img needs about $(human_gb "$bytes"). Move Docker's data-root or set DOCKER_HOST."
  fi
  log "pulling $img (about $(human_gb "$bytes") on disk)"
  t0=$(date +%s)
  dk pull "$img"
  log "pulled in $(($(date +%s) - t0))s"
done
for img in "${images[@]}"; do
  dk image inspect -f '{{.Id}}  {{range .RepoDigests}}{{.}} {{end}}' "$img"
done

# The training container's flags, now that IMAGE is here (check_env.sh could not
# try them before the pull).
probe_gpus
if n=$(check_train_flags); then
  [ "$n" = "$GPU_COUNT" ] || die "torch inside a training container sees '$n' GPUs, rocm-smi sees $GPU_COUNT"
  log "training-container flags accepted; torch sees $n GPU(s)"
else
  die "docker refused the training container's flags: $n (adjust TRAIN_DOCKER_FLAGS in scripts/lib.sh or ask the admin)"
fi
