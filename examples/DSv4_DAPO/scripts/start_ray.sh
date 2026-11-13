#!/usr/bin/env bash
# Start this host's raylet inside the training container.
#
#   bash scripts/start_ray.sh head
#   bash scripts/start_ray.sh worker      # HEAD_IP must be set (run.sh passes it)
#
# The raylet is started with runtime/ray_env.sh sourced, because every actor
# inherits its raylet's environment and not the driver's. --num-gpus is stated
# rather than detected. Only Ray inside OUR container is ever stopped.
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
ROLE=${1:?usage: start_ray.sh head|worker}
detect_net_if
my_ip=$(local_ipv4 "$NET_IF") || die "no IPv4 on $NET_IF"
dk inspect "$CONTAINER" >/dev/null 2>&1 || die "container $CONTAINER is not running; run scripts/start_container.sh"
probe_gpus

pids=$(foreign_ray_pids | tr '\n' ' ')
[ -z "${pids// /}" ] || die "Ray is already running on $(uname -n) outside $CONTAINER (pids: $pids); not touching it"

case "$ROLE" in
  head)
    HEAD_IP=${HEAD_IP:-$my_ip}
    start="ray start --head --port=$RAY_PORT --node-ip-address=$my_ip --num-gpus=$GPU_COUNT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --disable-usage-stats" ;;
  worker)
    : "${HEAD_IP:?HEAD_IP: address of the Ray head}"
    start="ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$my_ip --num-gpus=$GPU_COUNT --disable-usage-stats" ;;
  *) die "usage: start_ray.sh head|worker" ;;
esac

dk exec -e NET_IF="$NET_IF" -e DSV4_EXAMPLE_DIR="$DSV4_EXAMPLE_DIR" -e MEGATRON_DSV4_DIR="$MEGATRON_DSV4_DIR" \
  -e DSV4_SRC_MODEL_DIR="$DSV4_SRC_MODEL_DIR" "$CONTAINER" bash -lc "
set -e
source \"\$DSV4_EXAMPLE_DIR/runtime/ray_env.sh\"
ray stop --force >/dev/null 2>&1 || true
# vLLM's EngineCore and its workers are plain children and outlive 'ray stop'; a
# leftover one holds GPU memory and the weight-sync endpoint, and the next run's
# sender then waits forever for its ack. Matched on the process NAME: this shell's
# own command line contains the pattern, so -f would kill the killer.
pkill -9 '^VLLM::' 2>/dev/null || true
sleep 3
$start
"
log "ray $ROLE started on $(uname -n) ($my_ip), $GPU_COUNT GPU(s)"
