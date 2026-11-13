#!/usr/bin/env bash
# Put the pinned source trees under $RL_ROOT:
#
#   $RL_ROOT/Lumen-RL  this clone (a symlink to it, unless it already lives there)
#   $RL_ROOT/aiter     ZhangDanyang-AMD/aiter     @ PIN_AITER_REF, + composable_kernel
#   $RL_ROOT/Lumen     ZhangDanyang-AMD/Lumen     @ PIN_LUMEN_REF
#   $RL_ROOT/ATOM      xysheng-AMD/ATOM           @ PIN_ATOM_REF
#
# A directory that already exists is only verified, never changed: a wrong commit
# is an error, not something this script overwrites.
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
mkdir -p "$RL_ROOT"

# ---- Lumen-RL: this clone ------------------------------------------------------
dst="$RL_ROOT/Lumen-RL"
if [ -e "$dst" ]; then
  [ "$(realpath "$dst")" = "$(realpath "$DSV4_REPO_DIR")" ] \
    || [ -d "$dst/examples/DSv4_DAPO" ] \
    || die "$dst exists but is neither this clone nor a Lumen-RL with examples/DSv4_DAPO"
  log "Lumen-RL: $dst -> $(realpath "$dst")"
else
  ln -s "$DSV4_REPO_DIR" "$dst"
  log "Lumen-RL: linked $dst -> $DSV4_REPO_DIR"
fi
log "Lumen-RL HEAD $(git -C "$DSV4_REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"

head_of() { git -C "$1" rev-parse HEAD 2>/dev/null || true; }

# clone_pinned <dir> <url> <commit>
clone_pinned() {
  local dir=$1 url=$2 ref=$3 have
  if [ -e "$dir" ]; then
    have=$(head_of "$dir")
    [ "$have" = "$ref" ] || die "$dir is at '${have:-not a git checkout}', expected $ref. Move it away or set RL_ROOT elsewhere; this script does not overwrite."
    log "$(basename "$dir"): $ref (already there)"
    return 0
  fi
  log "$(basename "$dir"): cloning $url @ $ref"
  git clone --filter=blob:none --no-checkout "$url" "$dir"
  git -C "$dir" -c advice.detachedHead=false checkout "$ref"
  [ "$(head_of "$dir")" = "$ref" ] || die "$dir: checkout of $ref failed"
}

clone_pinned "$RL_ROOT/aiter" "$PIN_AITER_URL" "$PIN_AITER_REF"
ck="$RL_ROOT/aiter/$PIN_CK_PATH"
ck_have=$(git -C "$RL_ROOT/aiter" ls-tree HEAD "$PIN_CK_PATH" | cut -d' ' -f3 | cut -f1)
[ "$ck_have" = "$PIN_CK_REF" ] || die "aiter's gitlink for $PIN_CK_PATH is $ck_have, expected $PIN_CK_REF"
if [ "$(head_of "$ck")" != "$PIN_CK_REF" ]; then
  log "aiter: initialising $PIN_CK_PATH @ $PIN_CK_REF"
  git -C "$RL_ROOT/aiter" submodule update --init --filter=blob:none "$PIN_CK_PATH" 2>/dev/null \
    || git -C "$RL_ROOT/aiter" submodule update --init "$PIN_CK_PATH"
fi
[ "$(head_of "$ck")" = "$PIN_CK_REF" ] || die "$ck is not at $PIN_CK_REF"
log "aiter/$PIN_CK_PATH: $PIN_CK_REF"

clone_pinned "$RL_ROOT/Lumen" "$PIN_LUMEN_URL" "$PIN_LUMEN_REF"
clone_pinned "$RL_ROOT/ATOM" "$PIN_ATOM_URL" "$PIN_ATOM_REF"

log "code ready under $RL_ROOT"
