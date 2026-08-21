#!/usr/bin/env bash
# Install the locally built Apex + TransformerEngine wheels into the same NFS
# tree as vLLM, so both nodes get them without touching the primus image.
#
# --no-deps on purpose, twice over: Apex and TE only need torch, which primus
# already ships, and anything --target pulls in would shadow primus's own copy
# (PYTHONPATH always wins over site-packages). megatron-core went in the same
# way -- `pip install --target $SITE --no-deps "megatron-core==0.18.2"`.
#
#   bash ~/4node/install_megatron_primus.sh
#   PYTHONPATH=/home/xysheng/vllm_primus/site   # how to consume it
set -uo pipefail

OUT=${OUT:-/home/xysheng/vllm_primus}
SITE=$OUT/site
NAME=${NAME:-primus-build}

APEX_WHL=$(ls $OUT/wheels/apex-*.whl 2>/dev/null | head -1)
TE_WHLS=$(ls $OUT/wheels/transformer_engine*-*.whl 2>/dev/null)
[ -z "$APEX_WHL" ] && { echo "no apex wheel in $OUT/wheels"; exit 1; }
[ -z "$TE_WHLS" ] && { echo "no transformer_engine wheel in $OUT/wheels"; exit 1; }

echo "=== installing into $SITE"
for w in "$APEX_WHL" $TE_WHLS; do
  echo "  $w"
done

docker exec "$NAME" bash -lc "
  pip install --no-cache-dir --no-warn-script-location --no-deps \
    --target $SITE '$APEX_WHL' $TE_WHLS 2>&1 | tail -5
" || exit 1

echo "=== result"
du -sh "$SITE"
