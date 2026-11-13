#!/usr/bin/env bash
# Stop Ray (and leftover vLLM processes) inside the training container on every node.
#
#   bash scripts/stop.sh          # stop Ray, keep the containers
#   bash scripts/stop.sh --rm     # also remove the containers this example created
#
# Only our own container is touched.
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
RM=0; [ "${1:-}" = --rm ] && RM=1

stop_here() {
  if ! dk inspect "$CONTAINER" >/dev/null 2>&1; then log "$(uname -n): no container $CONTAINER"; return 0; fi
  [ "$(dk inspect -f '{{index .Config.Labels "org.lumenrl.example"}}' "$CONTAINER")" = dsv4-dapo ] \
    || die "$CONTAINER was not created by this example; not touching it"
  dk exec "$CONTAINER" bash -lc "ray stop --force >/dev/null 2>&1; pkill -9 '^VLLM::' 2>/dev/null; true"
  log "$(uname -n): ray stopped in $CONTAINER"
  if [ "$RM" = 1 ]; then dk rm -f "$CONTAINER" >/dev/null; log "$(uname -n): removed $CONTAINER"; fi
}

if [ "${DSV4_ENV_INLINE:-0}" = 1 ]; then stop_here; exit 0; fi
stop_here
if is_multi_node; then
  while read -r host; do remote_script "$host" stop.sh "$@"; done < <(remote_nodes)
fi
