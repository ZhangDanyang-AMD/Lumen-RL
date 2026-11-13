#!/usr/bin/env bash
# Check that this host (and, on the head, every node in NODES) can run the example.
#
#   bash scripts/check_env.sh            # this host, then the other NODES
#   bash scripts/check_env.sh --node     # this host only
#
# Fails on anything that would fail later; warns (and fails unless FORCE=1) on
# hardware that differs from the validated 8 x gfx950.
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
MODE=${1:-all}

fail=0
bad() { printf '[dsv4 FAIL] %s\n' "$*" >&2; fail=1; }

# ---- host tools ------------------------------------------------------------------
if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" = 4 ] && [ "${BASH_VERSINFO[1]}" -lt 3 ]; }; then
  bad "bash >= 4.3 required, found $BASH_VERSION"
fi
for t in git curl sha256sum sha1sum df realpath timeout tr tail sort; do
  command -v "$t" >/dev/null 2>&1 || bad "missing host command: $t"
done
command -v "${DOCKER_CMD[0]}" >/dev/null 2>&1 || bad "docker command not found: ${DOCKER_CMD[0]}"
dk info >/dev/null 2>&1 || die "cannot reach the docker daemon (DOCKER='$DOCKER' DOCKER_HOST='${DOCKER_HOST:-}')"
log "host $(uname -n): bash $BASH_VERSION, docker $(dk version -f '{{.Server.Version}}' 2>/dev/null || echo '?')"

# ---- images and Docker disk ----------------------------------------------------
need_docker=0
for spec in "IMAGE:$IMAGE_DISK_BYTES" "MILES_IMAGE:$MILES_IMAGE_DISK_BYTES"; do
  var=${spec%%:*} bytes=${spec#*:}
  if image_present "${!var}"; then log "$var present: ${!var}"
  else log "$var not pulled yet: ${!var}"; need_docker=$((need_docker + bytes)); fi
done
docker_root=$(dk info -f '{{.DockerRootDir}}' 2>/dev/null || echo '?')
docker_free=$(docker_free_bytes)

# ---- WORK_DIR disk -------------------------------------------------------------
# Measured per stage for 4 layers; the full model from its own conversions.
if [ "$DSV4_LAYERS" -lt "$MODEL_LAYERS_FULL" ]; then
  dl=$((1600000000 + DSV4_LAYERS * 6850000000))   # selected FP8 shards
  slice=$dl                                       # FP8 slice
  conv=$((3 * (2200000000 + DSV4_LAYERS * 13250000000)))  # bf16, torch_dist, bf16-native
else
  dl=295000000000 slice=0 conv=$((3 * 569000000000))
fi
[ -n "${MODEL_SRC_DIR:-}" ] && dl=0
data=5000000000   # parquets, raw datasets, HF / datasets caches
need=$((dl + slice + conv + data))
for d in "$FP8_DIR" "$BF16_DIR" "$DIST_DIR" "$NATIVE_DIR"; do
  [ -e "$d/.dsv4-done" ] && log "already built: $d"
done
free=$(path_free_bytes "$WORK_DIR")
same_fs=0
if [ -n "$docker_free" ] && [ "$(fs_of "$docker_root")" = "$(fs_of "$WORK_DIR")" ]; then same_fs=1; fi
if [ "$same_fs" = 1 ]; then
  # One disk holds both: the images and the artifacts have to fit together.
  total=$((need + need_docker))
  if [ "$free" -lt "$total" ]; then
    bad "Docker Root Dir ($docker_root) and WORK_DIR share a disk with $(human_gb "$free") free; images need $(human_gb "$need_docker") and DSV4_LAYERS=$DSV4_LAYERS about $(human_gb "$need"), $(human_gb "$total") together. Move Docker's data-root or WORK_DIR to a bigger disk."
  else
    log "Docker Root Dir and WORK_DIR share a disk: $(human_gb "$free") free, images need $(human_gb "$need_docker") + DSV4_LAYERS=$DSV4_LAYERS about $(human_gb "$need")"
  fi
else
  if [ "$need_docker" -gt 0 ]; then
    if [ -z "$docker_free" ]; then
      warn "cannot measure free space under Docker Root Dir ($docker_root); make sure it has $(human_gb "$need_docker")"
    elif [ "$docker_free" -lt "$need_docker" ]; then
      bad "Docker Root Dir $docker_root has $(human_gb "$docker_free") free, the images need $(human_gb "$need_docker"). Move Docker's data-root to a bigger disk or point DOCKER_HOST at a daemon that has one."
    else
      log "Docker Root Dir $docker_root: $(human_gb "$docker_free") free, images need $(human_gb "$need_docker")"
    fi
  fi
  if [ "$free" -lt "$need" ]; then
    bad "WORK_DIR $WORK_DIR has $(human_gb "$free") free; DSV4_LAYERS=$DSV4_LAYERS needs about $(human_gb "$need") (less whatever is already built)"
  else
    log "WORK_DIR $WORK_DIR: $(human_gb "$free") free, DSV4_LAYERS=$DSV4_LAYERS needs about $(human_gb "$need")"
  fi
fi

# ---- GPUs ----------------------------------------------------------------------
check_gpus
# The training container's flags, tried once: sites' docker authorisation policies
# reject some flags, and it is better to learn that here than at launch.
if image_present "$IMAGE"; then
  if n=$(check_train_flags); then
    if [ "$n" = "$GPU_COUNT" ]; then
      log "training-container flags accepted; torch sees $n GPU(s)"
    else
      bad "torch inside a training container sees '$n' GPUs, rocm-smi sees $GPU_COUNT"
    fi
  else
    bad "docker refused the training container's flags: $n"
  fi
else
  log "training-container flags not tried yet (IMAGE not pulled); pull_images.sh tries them after pulling"
fi

# ---- network, ports, foreign Ray -----------------------------------------------
detect_net_if
my_ip=$(local_ipv4 "$NET_IF") || bad "no IPv4 on $NET_IF"
log "NET_IF=$NET_IF ip=${my_ip:-?}${NET_IF_FROM_USER:+ (set in env.sh)}"
[ -d /proc/self ] || bad "/proc is not mounted here, so other Ray processes cannot be detected; run from a shell that has /proc"
pids=$(foreign_ray_pids | tr '\n' ' ')
if [ -n "${pids// /}" ]; then
  bad "Ray is already running on this host outside container '$CONTAINER' (pids: $pids). This example will not stop someone else's Ray; free the machine first."
fi
is_head=1
if [ "$MODE" = --node ] && [ -n "${HEAD_IP:-}" ] && [ "$HEAD_IP" != "${my_ip:-}" ]; then is_head=0; fi
if [ "$is_head" = 1 ] && [ -z "$(our_container_id)" ]; then
  HEAD_IP=${HEAD_IP:-$my_ip}
  for p in "$RAY_PORT" "$RAY_DASHBOARD_PORT"; do
    if port_in_use "$p"; then bad "port $p is already in use on this host; set RAY_PORT / RAY_DASHBOARD_PORT"; fi
  done
fi

# ---- the other nodes -----------------------------------------------------------
if [ "$MODE" != --node ] && is_multi_node; then
  HEAD_IP=${HEAD_IP:-$my_ip}
  [ "${NODE_ARR[0]}" = "$(uname -n)" ] || [ "${NODE_ARR[0]}" = localhost ] \
    || warn "the first NODES entry (${NODE_ARR[0]}) should be this host ($(uname -n)); scripts must run on the head"
  while read -r host; do
    log "---- checking $host via REMOTE_EXEC ----"
    remote_script "$host" check_env.sh --node || bad "check failed on $host"
  done < <(remote_nodes)
fi

[ "$fail" = 0 ] || die "environment check failed (see FAIL lines above)"
log "environment OK"
