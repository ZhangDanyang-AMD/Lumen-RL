#!/usr/bin/env bash
# Delete files that the root-run training container left behind. They belong to root
# on the host, so they are removed from inside a container.
#
#   bash scripts/clean.sh --pycache                         # __pycache__ the container wrote into
#                                                           # this clone, RL_ROOT and megatron_dsv4
#   bash scripts/clean.sh --aiter-jit                       # compiled aiter modules under $RL_ROOT/aiter
#   bash scripts/clean.sh "$WORK_DIR/models/dsv4-L4-bf16"   # any path strictly inside WORK_DIR,
#                                                           # RL_ROOT or DATA_ROOT
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
image_present "$IMAGE" || die "IMAGE not pulled"
[ $# -gt 0 ] || die "usage: clean.sh --pycache | --aiter-jit | <path>..."

targets=() pycache=0
for a in "$@"; do
  case "$a" in
    --pycache) pycache=1; continue ;;
    --aiter-jit)
      targets+=("$RL_ROOT/aiter/aiter/jit/build")
      for so in "$RL_ROOT"/aiter/aiter/jit/*.so; do [ -e "$so" ] && targets+=("$so"); done
      continue ;;
  esac
  p=$(realpath -m "$a")
  ok=0
  for root in "$WORK_DIR" "$RL_ROOT" "$DATA_ROOT"; do
    r=$(realpath -m "$root")
    [ "$p" != "$r" ] && [[ $p == "$r"/* ]] && ok=1
  done
  [ "$ok" = 1 ] || die "refusing to delete $p: not strictly inside WORK_DIR, RL_ROOT or DATA_ROOT"
  targets+=("$p")
done

script=""
if [ ${#targets[@]} -gt 0 ]; then
  log "removing: ${targets[*]}"
  script="rm -rf"
  for t in "${targets[@]}"; do script+=" $(printf %q "$t")"; done
  script+=";"
fi
if [ "$pycache" = 1 ]; then
  log "removing __pycache__ under $DSV4_REPO_DIR, $RL_ROOT, $MEGATRON_DSV4_DIR"
  for d in "$DSV4_REPO_DIR" "$RL_ROOT" "$MEGATRON_DSV4_DIR"; do
    [ -d "$d" ] && script+=" find $(printf %q "$d") -name __pycache__ -type d -prune -exec rm -rf {} +;"
  done
fi
dk_util "$IMAGE" "$script"
