#!/usr/bin/env bash
# Everything a node needs on disk for DSV4_LAYERS, under $WORK_DIR/models:
#
#   1  download      DeepSeek-V4-Flash-Base (only the shards N layers need)   download_model.sh
#   2  slice         dsv4-L<N>-fp8          first N layers, FP8, native names  [IMAGE]       (N < 43 only)
#   3  fp8 -> bf16   dsv4-L<N>-bf16         dequantised, DeepSeek-HF names      [MILES_IMAGE, 1 GPU]
#   4  torch_dist    dsv4-L<N>_torch_dist   Megatron checkpoint for training    [MILES_IMAGE]
#   5  reference     transformers logits    optional, RUN_REF=1                 [IMAGE, 1 GPU]
#   6  native names  dsv4-L<N>-bf16-native  what the vLLM rollout loads         [IMAGE]
#   7  check         tensor-name census of stage 6                               [IMAGE]
#
#   bash scripts/rebuild_artifacts.sh             # all stages, skipping finished ones
#   STAGE=3 STOP=4 bash scripts/rebuild_artifacts.sh
#
# A finished stage leaves <dir>/.dsv4-done; delete it (or the dir) to redo a stage.
# /models is node-local by nature: on several nodes, run this on every node.
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
STAGE=${STAGE:-1}
STOP=${STOP:-7}
RUN_REF=${RUN_REF:-0}
N=$DSV4_LAYERS
FULL=0; [ "$N" -eq "$MODEL_LAYERS_FULL" ] && FULL=1
q() { printf %q "$1"; }
done_mark() { [ -f "$1/.dsv4-done" ]; }
in_range() { [ "$STAGE" -le "$1" ] && [ "$STOP" -ge "$1" ]; }
stage_start() { T0=$(date +%s); log "===== $1 ====="; }
stage_end() { log "----- done in $(($(date +%s) - T0))s"; }
mkdir -p "$MODELS_DIR"
log "DSV4_LAYERS=$N  src=$DSV4_SRC_MODEL_DIR  out=$MODELS_DIR"
df -h "$MODELS_DIR" | tail -1 >&2

check_miles() {
  image_present "$MILES_IMAGE" || die "MILES_IMAGE not pulled; run scripts/pull_images.sh"
  dk run --rm --entrypoint /bin/bash "$MILES_IMAGE" -c \
    'test -f /root/miles/tools/fp8_cast_bf16.py && test -f /root/miles/scripts/models/deepseek-v4-flash.sh && test -f /root/miles/tools/convert_hf_to_torch_dist.py' \
    || die "MILES_IMAGE lacks /root/miles/tools/fp8_cast_bf16.py or scripts/models/deepseek-v4-flash.sh"
}

if in_range 1; then
  stage_start "1/7 download"
  bash "$DSV4_EXAMPLE_DIR/scripts/download_model.sh"
  stage_end
fi

if in_range 2; then
  if [ "$FULL" = 1 ]; then
    log "===== 2/7 slice: skipped, DSV4_LAYERS=$N is the full model ($FP8_DIR is used as is) ====="
  elif done_mark "$FP8_DIR"; then
    log "===== 2/7 slice: already done ($FP8_DIR) ====="
  else
    stage_start "2/7 slice: first $N layers -> $FP8_DIR"
    dk_util "$IMAGE" "set -e; rm -rf $(q "$FP8_DIR")
      python3 $(q "$DSV4_EXAMPLE_DIR/tools/make_native_subset.py") --src $(q "$DSV4_SRC_MODEL_DIR") --dst $(q "$FP8_DIR") --layers $N
      touch $(q "$FP8_DIR/.dsv4-done"); $(own_cmd "$FP8_DIR")"
    du -sh "$FP8_DIR" >&2
    stage_end
  fi
fi

if in_range 3; then
  if done_mark "$BF16_DIR"; then
    log "===== 3/7 fp8 -> bf16: already done ($BF16_DIR) ====="
  else
    check_miles
    stage_start "3/7 fp8 -> bf16: $FP8_DIR -> $BF16_DIR"
    dk_util --gpu 0 "$MILES_IMAGE" "set -e; rm -rf $(q "$BF16_DIR")
      python3 /root/miles/tools/fp8_cast_bf16.py --input-fp8-hf-path $(q "$FP8_DIR") --output-bf16-hf-path $(q "$BF16_DIR")
      cp -n $(q "$FP8_DIR")/tokenizer.json $(q "$BF16_DIR")/ 2>/dev/null || true
      touch $(q "$BF16_DIR/.dsv4-done"); $(own_cmd "$BF16_DIR")"
    du -sh "$BF16_DIR" >&2
    stage_end
  fi
fi

if in_range 4; then
  if done_mark "$DIST_DIR"; then
    log "===== 4/7 torch_dist: already done ($DIST_DIR) ====="
  else
    check_miles
    nproc=1 gpus=0
    if [ "$FULL" = 1 ]; then nproc=8 gpus=0,1,2,3,4,5,6,7; fi
    stage_start "4/7 bf16 -> torch_dist ($nproc rank(s)): $BF16_DIR -> $DIST_DIR"
    dk_util --gpu "$gpus" "$MILES_IMAGE" "set -e
      SRC=$(q "$BF16_DIR") DST=$(q "$DIST_DIR") LAYERS=$N MODEL_LAYERS=$MODEL_LAYERS_FULL NPROC=$nproc \
        bash $(q "$DSV4_EXAMPLE_DIR/tools/convert_torch_dist.sh")
      touch $(q "$DIST_DIR/.dsv4-done"); $(own_cmd "$DIST_DIR")"
    du -sh "$DIST_DIR" >&2
    stage_end
  fi
fi

if in_range 5; then
  if [ "$RUN_REF" = 1 ]; then
    stage_start "5/7 transformers reference logits (seq 256)"
    ref=$WORK_DIR/ref/dsv4-L${N}_ref_logits_256.pt
    mkdir -p "$(dirname "$ref")"
    dk_util --gpu 0 "$IMAGE" "set -e; cd $(q "$WORK_DIR/ref")
      python3 $(q "$DSV4_EXAMPLE_DIR/tools/ref_hf_logits.py") --model $(q "$FP8_DIR") --seq 256 --out $(q "$ref")
      $(own_cmd "$ref")"
    stage_end
  else
    log "===== 5/7 reference logits: skipped (RUN_REF=1 to run) ====="
  fi
fi

if in_range 6; then
  if done_mark "$NATIVE_DIR"; then
    log "===== 6/7 native names: already done ($NATIVE_DIR) ====="
  else
    stage_start "6/7 bf16 -> the checkpoint's own tensor names: $NATIVE_DIR"
    # The reference is the FP8 directory the bf16 one came from; its names minus
    # the `.scale` tensors and MTP are exactly what vLLM's DSv4 loader expects.
    dk_util "$IMAGE" "set -e; rm -rf $(q "$NATIVE_DIR")
      python3 $(q "$DSV4_EXAMPLE_DIR/tools/make_native_bf16.py") --src $(q "$BF16_DIR") --dst $(q "$NATIVE_DIR") --reference $(q "$FP8_DIR")
      touch $(q "$NATIVE_DIR/.dsv4-done"); $(own_cmd "$NATIVE_DIR")"
    du -sh "$NATIVE_DIR" >&2
    stage_end
  fi
fi

if in_range 7; then
  stage_start "7/7 check tensor names in $NATIVE_DIR"
  expect=()
  if [ "$FULL" = 1 ]; then expect=(--expect-tensors 34223 --expect-native 34217); fi
  dk_util "$IMAGE" "python3 $(q "$DSV4_EXAMPLE_DIR/tools/check_native_index.py") $(q "$NATIVE_DIR") ${expect[*]}"
  stage_end
fi

log "artifacts for DSV4_LAYERS=$N:"
for d in "$FP8_DIR" "$BF16_DIR" "$DIST_DIR" "$NATIVE_DIR"; do
  [ -e "$d" ] && du -sh "$d" >&2
done
df -h "$MODELS_DIR" | tail -1 >&2
