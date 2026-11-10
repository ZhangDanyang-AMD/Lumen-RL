#!/usr/bin/env bash
# Build the two parquets under $DATA_ROOT/data_cached/qwen3-8b-maxprompt1024/ inside IMAGE:
#   dapo-math-17k.filtered.parquet   training prompts
#   aime-2024.filtered.parquet       validation set; loaded at startup, never evaluated here
# Only the head node reads them.
#
#   bash scripts/make_data.sh                       # the files the reference run used
#   DATA_FILTER=tokens bash scripts/make_data.sh    # really drop prompts > 1024 tokens
#   COMPARE_WITH=<dir holding both parquets> bash scripts/make_data.sh   # content diff
#
# See tools/make_dapo_data.py for why the default removes nothing.
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
image_present "$IMAGE" || die "IMAGE not pulled; run scripts/pull_images.sh"
q() { printf %q "$1"; }
out_dir=$(dirname "$TRAIN_FILE")
files=(dapo-math-17k.filtered.parquet aime-2024.filtered.parquet)

if [ -f "$TRAIN_FILE" ] && [ -f "$VAL_FILE_PATH" ] && [ "${REBUILD:-0}" != 1 ]; then
  log "already there: $out_dir (REBUILD=1 to regenerate)"
else
  mkdir -p "$out_dir" "$DATA_ROOT/raw" "$DATA_ROOT/hf_home"
  t0=$(date +%s)
  dk_util --net "$IMAGE" "set -e
    python3 $(q "$DSV4_EXAMPLE_DIR/tools/make_dapo_data.py") --out-dir $(q "$out_dir") --raw-dir $(q "$DATA_ROOT/raw") \
      --filter $(q "${DATA_FILTER:-as-validated}")
    $(own_cmd "$out_dir" "$DATA_ROOT/raw" "$DATA_ROOT/hf_home")"
  log "built in $(($(date +%s) - t0))s"
fi
for f in "${files[@]}"; do log "sha256 $(sha256sum "$out_dir/$f" | cut -d' ' -f1)  $f"; done

if [ -n "${COMPARE_WITH:-}" ]; then
  ref=$(realpath "$COMPARE_WITH")
  for f in "${files[@]}"; do
    log "comparing $f with $ref/$f"
    dk run --rm -v "$ref:$ref:ro" -v "$out_dir:$out_dir:ro" \
      -v "$DSV4_EXAMPLE_DIR:$DSV4_EXAMPLE_DIR:ro" --entrypoint python3 "$IMAGE" \
      "$DSV4_EXAMPLE_DIR/tools/compare_parquet.py" "$out_dir/$f" "$ref/$f"
  done
fi
