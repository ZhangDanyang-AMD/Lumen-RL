#!/usr/bin/env bash
# Run one of the per-node setup scripts on this host and then on every other node in
# NODES through REMOTE_EXEC (the repository must be at the same path on every node).
#
#   bash scripts/on_all_nodes.sh pull_images.sh
#   bash scripts/on_all_nodes.sh prepare_code.sh
#   bash scripts/on_all_nodes.sh build_megatron.sh
#   bash scripts/on_all_nodes.sh rebuild_artifacts.sh
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
name=${1:?usage: on_all_nodes.sh <script> [args]}; shift
[ -f "$DSV4_EXAMPLE_DIR/scripts/$name" ] || die "no such script: scripts/$name"
log "==== $(uname -n) (local): $name $* ===="
bash "$DSV4_EXAMPLE_DIR/scripts/$name" "$@"
if is_multi_node; then
  detect_head_ip
  while read -r host; do
    log "==== $host: $name $* ===="
    remote_script "$host" "$name" "$@"
  done < <(remote_nodes)
fi
