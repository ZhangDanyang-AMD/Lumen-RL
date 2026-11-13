#!/usr/bin/env bash
# Shared helpers for examples/DSv4_DAPO/scripts. Sourced, never executed.
#
# Host requirements: bash >= 4.3, docker, git, curl, coreutils. Everything that needs
# Python runs inside one of the two pinned images.
# shellcheck disable=SC2034  # many variables here are consumed by the sourcing script

set -euo pipefail

DSV4_EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DSV4_REPO_DIR="$(cd "$DSV4_EXAMPLE_DIR/../.." && pwd)"

log()  { printf '[dsv4 %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
warn() { printf '[dsv4 WARN] %s\n' "$*" >&2; }
die()  { printf '[dsv4 ERROR] %s\n' "$*" >&2; exit 1; }

# ---- pinned versions (README: 固定版本) ------------------------------------------
PIN_AITER_URL=https://github.com/ZhangDanyang-AMD/aiter.git
PIN_AITER_REF=ff1006d03b53a693424c30e192c6e700e632bef8
PIN_CK_PATH=3rdparty/composable_kernel
PIN_CK_REF=83566edb0fded5e1c618c2c19110adbb74532762
PIN_LUMEN_URL=https://github.com/ZhangDanyang-AMD/Lumen.git
PIN_LUMEN_REF=e6379cbd9057b03c18213fbf65a4d891160545ca
PIN_ATOM_URL=https://github.com/xysheng-AMD/ATOM.git
PIN_ATOM_REF=7173f5b8f75849fec98f3943077deb44c00925d2
PIN_IMAGE=vllm/vime-rocm@sha256:3de38e0566effb9d97e14cd2412fa97467db96401776afb71fd424b09d9dfda9
PIN_MILES_IMAGE=rocm/sgl-dev:miles-rocm720-mi35x-20260803@sha256:fd662fc0ef3dbe99fb49f53fa6b393a921f50248f68d370655ac6562dc7f1301
PIN_HF_REPO=deepseek-ai/DeepSeek-V4-Flash-Base
PIN_HF_REV=8855555deef230a27a21a8d6f294b7b7497759b6
# sha256 over `sha256sum` lines of every file in the rebuilt megatron/ tree
# (LC_ALL=C path order, __pycache__ excluded). See build_megatron.sh.
MEGATRON_TREE_SHA256=60cd2a97717fb695485ff570486a61c61b45ed842fc852144c6cd0359c95fb45
MODEL_LAYERS_FULL=43

# Validated hardware: 8 GPUs of gfx950 per node, 288 GiB each.
EXPECT_GPUS=8
EXPECT_GFX=gfx950
EXPECT_VRAM_MIN=$((280 * 1024 * 1024 * 1024))

# Disk the two images take in Docker's data root. With the containerd image store
# the compressed layers stay next to the unpacked ones; the classic overlay2 store
# needs only the unpacked part (about 59.6 GB and 83.5 GB).
IMAGE_DISK_BYTES=81838650662
MILES_IMAGE_DISK_BYTES=115048795595

LABEL=org.lumenrl.example=dsv4-dapo

# ---- environment ---------------------------------------------------------------
load_env() {
  if [ "${DSV4_ENV_INLINE:-0}" != 1 ]; then
    local f="${DSV4_ENV_FILE:-$DSV4_EXAMPLE_DIR/env.sh}"
    [ -f "$f" ] || die "missing $f -- copy env.example.sh to env.sh and fill in WORK_DIR"
    # shellcheck disable=SC1090
    source "$f"
  fi
  : "${WORK_DIR:?WORK_DIR: absolute path of the working directory (models, logs, code, data go under it)}"
  case "$WORK_DIR" in /*) ;; *) die "WORK_DIR must be an absolute path: $WORK_DIR" ;; esac
  case "$WORK_DIR" in *'<'*|*'>'*) die "WORK_DIR still holds the placeholder from env.example.sh" ;; esac

  DSV4_LAYERS=${DSV4_LAYERS:-4}
  [[ $DSV4_LAYERS =~ ^[0-9]+$ ]] && [ "$DSV4_LAYERS" -ge 1 ] && [ "$DSV4_LAYERS" -le "$MODEL_LAYERS_FULL" ] \
    || die "DSV4_LAYERS must be 1..$MODEL_LAYERS_FULL, got '$DSV4_LAYERS'"
  RL_ROOT=${RL_ROOT:-$WORK_DIR/src}
  DATA_ROOT=${DATA_ROOT:-$WORK_DIR}
  CONTAINER=${CONTAINER:-dsv4-dapo}
  IMAGE=${IMAGE:-$PIN_IMAGE}
  MILES_IMAGE=${MILES_IMAGE:-$PIN_MILES_IMAGE}
  DOCKER=${DOCKER:-docker}
  RAY_PORT=${RAY_PORT:-6379}
  RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
  REMOTE_EXEC=${REMOTE_EXEC:-'ssh {host}'}
  HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
  STEPS=${STEPS:-3}
  FORCE=${FORCE:-0}

  MODELS_DIR=$WORK_DIR/models
  LOG_DIR=$WORK_DIR/logs
  MEGATRON_DSV4_DIR=$WORK_DIR/megatron_dsv4
  DSV4_SRC_MODEL_DIR=${MODEL_SRC_DIR:-$MODELS_DIR/DeepSeek-V4-Flash-Base}
  TRAIN_FILE=$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet
  # run_dapo.sh falls back to this when VAL_FILE is empty, and the trainer loads it.
  VAL_FILE_PATH=$DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet

  local n=$DSV4_LAYERS
  if [ "$n" -lt "$MODEL_LAYERS_FULL" ]; then FP8_DIR=$MODELS_DIR/dsv4-L$n-fp8; else FP8_DIR=$DSV4_SRC_MODEL_DIR; fi
  BF16_DIR=$MODELS_DIR/dsv4-L$n-bf16
  DIST_DIR=$MODELS_DIR/dsv4-L${n}_torch_dist
  NATIVE_DIR=$MODELS_DIR/dsv4-L$n-bf16-native

  read -r -a DOCKER_CMD <<<"$DOCKER"
  if [ -n "${DOCKER_HOST:-}" ]; then export DOCKER_HOST; fi
  if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; fi
  NET_IF_FROM_USER=${NET_IF:+1}
  read -r -a NODE_ARR <<<"${NODES:-}"
  NUM_NODES=${#NODE_ARR[@]}
  [ "$NUM_NODES" -ge 1 ] || NUM_NODES=1
}

dk() { "${DOCKER_CMD[@]}" "$@"; }

# ---- detection -----------------------------------------------------------------
detect_net_if() {
  [ -n "${NET_IF:-}" ] && return 0
  [ -r /proc/net/route ] || die "cannot read /proc/net/route; set NET_IF in env.sh"
  local iface dest _rest
  while read -r iface dest _rest; do
    if [ "$dest" = 00000000 ]; then NET_IF=$iface; return 0; fi
  done < <(tail -n +2 /proc/net/route)
  die "no default route found; set NET_IF in env.sh"
}

# IPv4 of an interface on THIS host. Uses ip(8) when present, else the image.
local_ipv4() {
  local ifname=$1 line
  if command -v ip >/dev/null 2>&1; then
    while read -r line; do
      if [[ $line =~ inet\ ([0-9.]+)/ ]]; then echo "${BASH_REMATCH[1]}"; return 0; fi
    done < <(ip -o -4 addr show dev "$ifname" 2>/dev/null)
    return 1
  fi
  dk run --rm --network host -v "$DSV4_EXAMPLE_DIR:$DSV4_EXAMPLE_DIR:ro" \
    --entrypoint python3 "$IMAGE" "$DSV4_EXAMPLE_DIR/tools/netinfo.py" "$ifname"
}

detect_head_ip() {
  detect_net_if
  if [ -z "${HEAD_IP:-}" ]; then
    HEAD_IP=$(local_ipv4 "$NET_IF") || die "no IPv4 address on $NET_IF; set NET_IF or HEAD_IP in env.sh"
  fi
}

# rocm-smi on the host if it has one, otherwise the one inside the image.
_smi_csv() {
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname --showmeminfo vram --csv 2>/dev/null
  else
    dk run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
      --entrypoint rocm-smi "$IMAGE" --showproductname --showmeminfo vram --csv 2>/dev/null
  fi
}

# Sets GPU_COUNT, GPU_GFX (comma list, unique), GPU_VRAM_MIN (bytes).
probe_gpus() {
  GPU_COUNT=0 GPU_GFX="" GPU_VRAM_MIN=0
  local -a hdr row
  local i ivram=-1 igfx=-1 first=1 line gfxs=""
  while IFS= read -r line; do
    [[ $line == *,* ]] || continue
    IFS=, read -r -a row <<<"$line"
    if [ "$first" = 1 ]; then
      hdr=("${row[@]}"); first=0
      for i in "${!hdr[@]}"; do
        case "${hdr[$i]}" in
          "VRAM Total Memory (B)") ivram=$i ;;
          "GFX Version") igfx=$i ;;
        esac
      done
      continue
    fi
    [[ ${row[0]} == card* ]] || continue
    GPU_COUNT=$((GPU_COUNT + 1))
    if [ "$ivram" -ge 0 ]; then
      local v=${row[$ivram]}
      if [ "$GPU_VRAM_MIN" = 0 ] || [ "$v" -lt "$GPU_VRAM_MIN" ]; then GPU_VRAM_MIN=$v; fi
    fi
    if [ "$igfx" -ge 0 ] && [[ ",$gfxs," != *",${row[$igfx]},"* ]]; then
      gfxs=${gfxs:+$gfxs,}${row[$igfx]}
    fi
  done < <(_smi_csv)
  GPU_GFX=$gfxs
}

check_gpus() {
  probe_gpus
  local gib=$((GPU_VRAM_MIN / 1024 / 1024 / 1024))
  log "GPUs: count=$GPU_COUNT arch=${GPU_GFX:-?} vram/gpu=${gib}GiB (validated: $EXPECT_GPUS x $EXPECT_GFX, 288GiB)"
  [ "$GPU_COUNT" -ge 1 ] || die "no AMD GPU visible (rocm-smi found none)"
  local bad=0
  [ "$GPU_COUNT" = "$EXPECT_GPUS" ] || { warn "expected $EXPECT_GPUS GPUs per node, found $GPU_COUNT"; bad=1; }
  [ "$GPU_GFX" = "$EXPECT_GFX" ] || { warn "expected $EXPECT_GFX, found '${GPU_GFX}' -- unvalidated architecture"; bad=1; }
  [ "$GPU_VRAM_MIN" -ge "$EXPECT_VRAM_MIN" ] || { warn "expected >= 280GiB per GPU (peak use was ~115GiB trainer + 0.25 x VRAM rollout), found ${gib}GiB"; bad=1; }
  if [ "$bad" = 1 ]; then
    [ "$FORCE" = 1 ] || die "hardware differs from the validated setup; set FORCE=1 to continue anyway"
    warn "FORCE=1: continuing on unvalidated hardware"
  fi
}

# Free bytes at a path, creating it first.
path_free_bytes() {
  mkdir -p "$1"
  local _fs _blocks _used avail _rest
  { read -r _; read -r _fs _blocks _used avail _rest; } < <(df -P -B1 "$1")
  echo "$avail"
}

# Free bytes under Docker's data root, or empty if it cannot be measured from here.
docker_free_bytes() {
  case "${DOCKER_HOST:-}" in tcp://*|ssh://*) return 0 ;; esac
  local root
  root=$(dk info -f '{{.DockerRootDir}}' 2>/dev/null) || return 0
  [ -n "$root" ] || return 0
  local _fs _blocks _used avail _rest
  { read -r _; read -r _fs _blocks _used avail _rest; } < <(df -P -B1 "$root" 2>/dev/null) || return 0
  echo "$avail"
}

image_present() { dk image inspect "$1" >/dev/null 2>&1; }

human_gb() { echo "$(($1 / 1000000000)) GB"; }

port_in_use() {
  local port=$1 h
  for h in 127.0.0.1 ${HEAD_IP:+"$HEAD_IP"}; do
    if timeout 2 bash -c "exec 3<>/dev/tcp/$h/$port" 2>/dev/null; then return 0; fi
  done
  return 1
}

# Container id of OUR container, if it exists.
our_container_id() { dk inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null || true; }

# PIDs of Ray processes on this host that do not belong to our container.
foreign_ray_pids() {
  local ours d cmd cg
  ours=$(our_container_id)
  for d in /proc/[0-9]*; do
    cmd=$(tr '\0' ' ' <"$d/cmdline" 2>/dev/null) || continue
    [[ $cmd =~ (^|/)(raylet|gcs_server)( |$) || $cmd == *"ray/core/src/ray/"* ]] || continue
    cg=$(<"$d/cgroup") 2>/dev/null || cg=""
    if [ -n "$ours" ] && [[ $cg == *"$ours"* ]]; then continue; fi
    echo "${d#/proc/}"
  done
}

# ---- docker helpers ------------------------------------------------------------
# Flags of the long-lived training container. No --privileged: GPU access is the two
# device nodes plus the video group, and nothing on this path needs more (RDMA is
# off). Many sites' docker authorisation rejects --privileged outright.
TRAIN_DOCKER_FLAGS=(
  --network=host --ipc=host
  --ulimit nofile=1048576:1048576 --ulimit memlock=-1 --ulimit core=0
  --device=/dev/kfd --device=/dev/dri --group-add=video
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G
)

# -v p:p for every directory the containers need, each mounted at the same path.
container_mounts() {
  local -a dirs=("$WORK_DIR" "$RL_ROOT" "$DATA_ROOT" "$DSV4_REPO_DIR")
  [ -n "${MODEL_SRC_DIR:-}" ] && dirs+=("$MODEL_SRC_DIR")
  [ -L "$RL_ROOT/Lumen-RL" ] && dirs+=("$(realpath "$RL_ROOT/Lumen-RL")")
  local d seen=" "
  for d in "${dirs[@]}"; do
    [ -e "$d" ] || mkdir -p "$d"
    d=$(realpath "$d")
    [[ $seen == *" $d "* ]] && continue
    seen+="$d "
    printf -- '-v\n%s:%s\n' "$d" "$d"
  done
}

# Start a throwaway container with the training container's flags and count GPUs in
# it. Sites' docker authorisation policies reject some flags; this is where that
# shows. Prints the GPU count; returns 1 (with docker's first error line) on refusal.
check_train_flags() {
  local -a extra
  local out
  read -r -a extra <<<"${EXTRA_DOCKER_ARGS:-}"
  if ! out=$(dk run --rm --entrypoint /bin/bash "${TRAIN_DOCKER_FLAGS[@]}" "${extra[@]}" "$IMAGE" \
               -c 'python3 -c "import torch; print(torch.cuda.device_count())"' 2>&1); then
    printf '%s\n' "$out" | head -1
    return 1
  fi
  echo "${out##*$'\n'}"
}

# Filesystem (device) holding a path, as df reports it.
fs_of() {
  local fs _rest
  { read -r _; read -r fs _rest; } < <(df -P "$1" 2>/dev/null) || return 0
  echo "$fs"
}

proxy_env_args() {
  local v
  for v in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy HF_ENDPOINT HF_TOKEN; do
    if [ -n "${!v:-}" ]; then printf -- '-e\n%s\n' "$v"; fi
  done
}

# chown -R back to the invoking user, as a shell snippet for the end of a root container.
own_cmd() {
  [ "$(id -u)" = 0 ] && return 0
  printf 'chown -R %s:%s' "$(id -u)" "$(id -g)"
  printf ' %q' "$@"
  printf ' 2>/dev/null || echo "[dsv4 WARN] could not chown %s back to the host user" >&2\n' "$*"
}

# Run a one-off root container from IMAGE or MILES_IMAGE.
#   dk_util [--gpu [DEVICES]] [--net] <image> <bash -lc script>
dk_util() {
  local -a extra=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --gpu) extra+=(--device=/dev/kfd --device=/dev/dri --group-add video
                     --security-opt seccomp=unconfined --shm-size 32G
                     -e "HIP_VISIBLE_DEVICES=$2"); shift 2 ;;
      --net) extra+=(--network host); shift ;;
      *) break ;;
    esac
  done
  local image=$1 script=$2
  local -a mounts env
  mapfile -t mounts < <(container_mounts)
  mapfile -t env < <(proxy_env_args)
  dk run --rm "${extra[@]}" "${mounts[@]}" "${env[@]}" -e HOME=/tmp \
    -e "HF_HOME=$DATA_ROOT/hf_home" -e HF_HUB_DISABLE_XET=1 \
    --entrypoint /bin/bash "$image" -lc "$script"
}

# ---- multi-node ----------------------------------------------------------------
is_multi_node() { [ "${#NODE_ARR[@]}" -gt 1 ]; }
remote_nodes() { local i; for ((i = 1; i < ${#NODE_ARR[@]}; i++)); do echo "${NODE_ARR[$i]}"; done; }

inline_env() {
  local v
  for v in WORK_DIR RL_ROOT DATA_ROOT DSV4_LAYERS CONTAINER IMAGE MILES_IMAGE DOCKER DOCKER_HOST \
           RAY_PORT RAY_DASHBOARD_PORT HEAD_IP NODES FORCE STEPS HF_ENDPOINT HF_TOKEN MODEL_SRC_DIR \
           HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    if [ -n "${!v:-}" ]; then printf 'export %s=%q\n' "$v" "${!v}"; fi
  done
  if [ "${NET_IF_FROM_USER:-}" = 1 ]; then printf 'export NET_IF=%q\n' "$NET_IF"; fi
  echo 'export DSV4_ENV_INLINE=1'
}

# Run scripts/<name> with args on a remote node through REMOTE_EXEC.
# The command is fed to `bash -s` on stdin, so REMOTE_EXEC only has to forward
# stdin (ssh and `kubectl exec -i` do, as do most schedulers' run-on-node commands).
# The repository must exist at the same absolute path on every node.
remote_script() {
  local host=$1 name=$2; shift 2
  local tmpl=${REMOTE_EXEC//\{host\}/$host}
  local -a cmd
  read -r -a cmd <<<"$tmpl"
  {
    inline_env
    # $(uname -n) is meant to expand on the remote node.
    # shellcheck disable=SC2016
    printf '[ -d %q ] || { echo "[dsv4 ERROR] %s not found on $(uname -n)" >&2; exit 1; }\n' \
      "$DSV4_EXAMPLE_DIR" "$DSV4_EXAMPLE_DIR"
    printf 'exec bash %q' "$DSV4_EXAMPLE_DIR/scripts/$name"
    printf ' %q' "$@"
    echo
  } | "${cmd[@]}" bash -s
}
