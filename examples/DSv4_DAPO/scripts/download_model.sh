#!/usr/bin/env bash
# Download the DeepSeek-V4-Flash-Base files that DSV4_LAYERS needs, and verify each
# one against the Hub's own checksum (LFS sha256 for shards, git blob sha1 for the
# small files).
#
#   bash scripts/download_model.sh --dry-run   # print the file list, download nothing
#   bash scripts/download_model.sh             # download into $WORK_DIR/models/DeepSeek-V4-Flash-Base
#
# Which shards: from model.safetensors.index.json, every shard that holds
# `layers.<i>` with i < DSV4_LAYERS or a top-level tensor (`mtp.*` excluded); all
# shards when DSV4_LAYERS is the model's full depth. With MODEL_SRC_DIR set nothing
# is downloaded; the same files are checked there instead (VERIFY=0 only checks
# that they exist). The Hub API must be reachable in every mode.
#
# Env: HF_ENDPOINT (mirror), HF_TOKEN (optional), DL_PARALLEL (parallel downloads, 4).
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1
DL_PARALLEL=${DL_PARALLEL:-4}
VERIFY=${VERIFY:-1}
image_present "$IMAGE" || die "IMAGE not pulled; run scripts/pull_images.sh (the file plan is computed inside it)"

meta=$WORK_DIR/cache/hf/$PIN_HF_REV
mkdir -p "$meta"
auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
fetch() {  # fetch <url> <out>
  curl -fsSL --retry 5 --retry-delay 5 "${auth[@]}" -o "$2.part" "$1" && mv "$2.part" "$2"
}
resolve_url() { echo "$HF_ENDPOINT/$PIN_HF_REPO/resolve/$PIN_HF_REV/$1"; }

if [ -n "${MODEL_SRC_DIR:-}" ]; then
  dst=$(realpath "$MODEL_SRC_DIR")
  cp "$dst/model.safetensors.index.json" "$meta/index.json"
  cp "$dst/config.json" "$meta/config.json"
else
  dst=$MODELS_DIR/DeepSeek-V4-Flash-Base
  [ -s "$meta/index.json" ] || fetch "$(resolve_url model.safetensors.index.json)" "$meta/index.json"
  [ -s "$meta/config.json" ] || fetch "$(resolve_url config.json)" "$meta/config.json"
fi
# The Hub's file listing carries the sizes and checksums for the pinned revision.
[ -s "$meta/tree.json" ] || fetch "$HF_ENDPOINT/api/models/$PIN_HF_REPO/tree/$PIN_HF_REV" "$meta/tree.json"

plan=$meta/plan-L$DSV4_LAYERS.tsv
dk_util "$IMAGE" "python3 $(printf %q "$DSV4_EXAMPLE_DIR/tools/hf_plan.py") \
  --index $(printf %q "$meta/index.json") --config $(printf %q "$meta/config.json") \
  --tree $(printf %q "$meta/tree.json") \
  --layers $DSV4_LAYERS --out $(printf %q "$plan") && $(own_cmd "$plan")"

total=0
while IFS=$'\t' read -r path _kind size _hash; do
  printf '  %-45s %14s bytes\n' "$path" "$size"
  total=$((total + size))
done <"$plan"
log "DSV4_LAYERS=$DSV4_LAYERS: $(wc -l <"$plan") files, $total bytes ($(human_gb "$total")) from $PIN_HF_REPO@$PIN_HF_REV"
[ "$DRY" = 1 ] && exit 0

verify_file() {  # verify_file <file> <kind> <size> <hash>
  local f=$1 kind=$2 size=$3 want=$4 got
  [ "$(stat -c %s "$f")" = "$size" ] || return 1
  if [ "$kind" = lfs ]; then
    got=$(sha256sum "$f" | cut -d' ' -f1)
  else
    got=$({ printf 'blob %s\0' "$size"; cat "$f"; } | sha1sum | cut -d' ' -f1)
  fi
  [ "$got" = "$want" ]
}

# Verified-file markers live under WORK_DIR, so MODEL_SRC_DIR may be read-only.
marks=$meta/verified-$(printf '%s' "$dst" | sha1sum | cut -c1-12)
mkdir -p "$dst" "$marks"
one() {  # one <path> <kind> <size> <hash>
  local path=$1 kind=$2 size=$3 hash=$4 f="$dst/$1" mark="$marks/$1"
  if [ -f "$f" ] && [ -f "$mark" ] && [ "$(cat "$mark")" = "$hash" ]; then return 0; fi
  if [ -f "$f" ] && verify_file "$f" "$kind" "$size" "$hash"; then echo "$hash" >"$mark"; return 0; fi
  if [ -n "${MODEL_SRC_DIR:-}" ]; then
    echo "[dsv4 FAIL] $f is missing or does not match the Hub checksum" >&2; return 1
  fi
  local part="$f.part" have=0
  [ -f "$part" ] && have=$(stat -c %s "$part")
  if [ "$have" != "$size" ]; then
    curl -fL --retry 10 --retry-delay 10 --retry-all-errors -sS -C - "${auth[@]}" \
      -o "$part" "$(resolve_url "$path")" || { echo "[dsv4 FAIL] download $path" >&2; return 1; }
  fi
  verify_file "$part" "$kind" "$size" "$hash" || {
    echo "[dsv4 FAIL] checksum mismatch: $path (removing the partial file)" >&2; rm -f "$part"; return 1; }
  mv "$part" "$f"
  echo "$hash" >"$mark"
  echo "[dsv4] ok $path" >&2
}

if [ "$VERIFY" = 0 ]; then
  log "VERIFY=0: checking presence only in $dst"
  while IFS=$'\t' read -r path _ _ _; do [ -f "$dst/$path" ] || die "missing $dst/$path"; done <"$plan"
  exit 0
fi

t0=$(date +%s); failed=0; running=0
while IFS=$'\t' read -r path kind size hash; do
  one "$path" "$kind" "$size" "$hash" &
  running=$((running + 1))
  if [ "$running" -ge "$DL_PARALLEL" ]; then wait -n || failed=1; running=$((running - 1)); fi
done <"$plan"
while [ "$running" -gt 0 ]; do wait -n || failed=1; running=$((running - 1)); done
[ "$failed" = 0 ] || die "some files failed (see FAIL lines); rerun to resume"
log "all $(wc -l <"$plan") files present and verified in $dst ($(($(date +%s) - t0))s)"
