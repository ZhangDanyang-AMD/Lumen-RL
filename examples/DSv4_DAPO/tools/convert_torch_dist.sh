#!/usr/bin/env bash
# BF16 HF checkpoint -> Megatron torch_dist for DeepSeek-V4-Flash. Runs INSIDE the
# miles image (needs /root/miles and /root/Megatron-LM from it).
#
#   SRC=<bf16 dir> DST=<torch_dist dir> LAYERS=<n> MODEL_LAYERS=<n of the full model> \
#     [NPROC=<ranks>] bash convert_torch_dist.sh
#
# The conversion runs at TP=PP=EP=CP=1. The layout does not constrain training:
# torch_dist reshards on load.
#
# A slice (LAYERS < the model's num_hidden_layers) overrides the model script's layer
# count and compress-ratio schedule with the first LAYERS entries of the checkpoint's
# own `compress_ratios`, so layer types stay as in the full model (for 4 layers:
# 0 0 4 128 = sliding, sliding, compressed+indexed, heavily compressed). NPROC
# defaults to 1 for a slice: convert_hf_to_torch_dist.py rewrites PP to world_size
# when PP==1 and world_size>1, and one rank has nothing to rewrite.
#
# The full model keeps the model script's defaults and converts on 8 ranks. With
# EP=1, convert_hf_to_torch_dist.py's own auto-PP splits the 43 layers over the 8
# ranks (each holds 43/8 layers with all 256 experts, about 71 GB of the 569 GB).
# EP=8 would make 1*8*8 = 64 != world_size and fail.
set -euo pipefail
: "${SRC:?SRC: BF16 HF checkpoint directory}"
: "${DST:?DST: output torch_dist directory}"
: "${LAYERS:?LAYERS: number of transformer layers in SRC}"
: "${MODEL_LAYERS:?MODEL_LAYERS: num_hidden_layers of the original (unsliced) model}"
MILES_DIR=${MILES_DIR:-/root/miles}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}

if [ "$LAYERS" -lt "$MODEL_LAYERS" ]; then
  NPROC=${NPROC:-1}
  export MODEL_ARGS_NUM_LAYERS=$LAYERS
  mapfile -t COMPRESS_RATIOS < <(python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))
for r in c['compress_ratios'][:int(sys.argv[2])]:
    print(r)" "$SRC/config.json" "$LAYERS")
  echo "slice: num_layers=$LAYERS compress_ratios=(${COMPRESS_RATIOS[*]}) nproc=$NPROC"
else
  NPROC=${NPROC:-8}
  echo "full model: num_layers=$LAYERS (model script defaults) nproc=$NPROC"
fi

if [ -f "$DST/latest_checkpointed_iteration.txt" ]; then
  echo "[skip] $DST already converted"
  exit 0
fi
rm -rf "$DST"

cd "$MILES_DIR"
# The model script probes ${#COMPRESS_RATIOS[@]} before defining it, which trips set -u.
set +u
# shellcheck disable=SC1091
source "$MILES_DIR/scripts/models/deepseek-v4-flash.sh"
set -u

PYTHONPATH=$MILES_DIR:$MEGATRON_DIR torchrun --nproc-per-node "$NPROC" \
  "$MILES_DIR/tools/convert_hf_to_torch_dist.py" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$SRC" \
  --save "$DST" \
  --expert-tensor-parallel-size 1 \
  --context-parallel-size 1 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1

echo "===== done ====="
du -sh "$DST"
echo "latest_checkpointed_iteration: $(cat "$DST/latest_checkpointed_iteration.txt")"
