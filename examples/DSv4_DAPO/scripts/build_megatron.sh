#!/usr/bin/env bash
# Rebuild the DeepSeek-V4-patched Megatron into $WORK_DIR/megatron_dsv4/megatron:
#
#   base   /root/Megatron-LM/megatron from IMAGE (NVIDIA/Megatron-LM dev @ 1dcf0dafa
#          plus the image's own uncommitted changes, megatron-core 0.16.0rc0)
#   patch  megatron/megatron_dsv4.patch (18 files)
#
# The result is checked against MEGATRON_TREE_SHA256, a digest over every file of the
# tree that the 4-layer run used. VERIFY_AGAINST=<dir containing megatron/> adds a
# file-by-file sha256 comparison against another tree.
#
#   bash scripts/build_megatron.sh            # build (no-op if already verified)
#   REBUILD=1 bash scripts/build_megatron.sh  # throw the old tree away first
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
out=$MEGATRON_DSV4_DIR
patch=$DSV4_EXAMPLE_DIR/megatron/megatron_dsv4.patch
image_present "$IMAGE" || die "IMAGE not pulled; run scripts/pull_images.sh"

if [ -f "$out/.tree.sha256" ] && [ "$(cat "$out/.tree.sha256")" = "$MEGATRON_TREE_SHA256" ] && [ "${REBUILD:-0}" != 1 ]; then
  log "megatron_dsv4 already built and verified: $out"
  exit 0
fi
if [ -e "$out" ]; then
  [ "${REBUILD:-0}" = 1 ] || die "$out exists but is not a verified build; set REBUILD=1 to replace it"
  dk_util "$IMAGE" "rm -rf $(printf %q "$out")"
fi
mkdir -p "$(dirname "$out")"

manifest='find megatron -type f ! -path "*/__pycache__/*" ! -name "*.pyc" -print0 | LC_ALL=C sort -z | xargs -0 sha256sum'
log "building $out from IMAGE + $(basename "$patch")"
dk_util "$IMAGE" "
set -euo pipefail
tmp=\$(mktemp -d /tmp/megatron-build.XXXXXX)
cp -a /root/Megatron-LM/megatron \"\$tmp/\"
find \"\$tmp\" -name __pycache__ -prune -exec rm -rf {} +
find \"\$tmp\" -name '*.pyc' -delete
cd \"\$tmp\"
git apply -p1 --whitespace=nowarn $(printf %q "$patch")
($manifest) > \"\$tmp/manifest\"
mkdir -p $(printf %q "$out")
cp -a \"\$tmp/megatron\" $(printf %q "$out")/
cp \"\$tmp/manifest\" $(printf %q "$out")/.manifest
sha256sum < \"\$tmp/manifest\" | cut -d' ' -f1 > $(printf %q "$out")/.tree.sha256
$(own_cmd "$out")
"
got=$(cat "$out/.tree.sha256")
log "tree sha256: $got ($(wc -l <"$out/.manifest") files)"

if [ -n "${VERIFY_AGAINST:-}" ]; then
  ref=$(realpath "$VERIFY_AGAINST")
  log "comparing file by file against $ref"
  dk run --rm -v "$ref:$ref:ro" --entrypoint /bin/bash "$IMAGE" -c "cd $(printf %q "$ref") && $manifest" \
    > "$out/.manifest.ref"
  if [ "$(sha256sum <"$out/.manifest")" = "$(sha256sum <"$out/.manifest.ref")" ]; then
    log "identical to $ref: $(wc -l <"$out/.manifest") files, every sha256 equal"
  else
    echo "  only in reference (left) / only in rebuilt (right):" >&2
    comm -3 <(sort "$out/.manifest.ref") <(sort "$out/.manifest") >&2 || true
    die "rebuilt tree differs from $ref"
  fi
fi

[ "$got" = "$MEGATRON_TREE_SHA256" ] || die "tree sha256 $got != expected $MEGATRON_TREE_SHA256"
log "megatron_dsv4 OK: $out"
