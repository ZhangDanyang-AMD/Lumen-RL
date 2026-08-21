#!/usr/bin/env bash
# Install the locally built vLLM wheel into an NFS tree that both nodes can see,
# without touching the primus image.
#
# Why not just `pip install --target` the wheel: --target ignores what is already
# in the image and would re-install ~200 packages, every one of which then shadows
# primus's copy (PYTHONPATH always wins over site-packages). So resolve first in a
# pristine primus container, take only the packages pip says are MISSING, and
# install exactly those. Everything primus already ships keeps being used.
#
#   bash ~/4node/install_vllm_primus.sh
#   PYTHONPATH=/home/xysheng/vllm_primus/site  # how to consume it
set -uo pipefail

OUT=${OUT:-/home/xysheng/vllm_primus}
SITE=$OUT/site
PROBE_CONTAINER=${PROBE_CONTAINER:-anp-primus}   # pristine: used only to resolve
WORK_CONTAINER=${WORK_CONTAINER:-primus-build}   # has network + pip, does the work

WHEEL=$(ls $OUT/wheels/vllm-*.whl 2>/dev/null | head -1)
[ -z "$WHEEL" ] && { echo "no wheel in $OUT/wheels"; exit 1; }
echo "wheel: $WHEEL"

echo "=== resolving against a pristine primus ($PROBE_CONTAINER)"
docker exec "$PROBE_CONTAINER" bash -lc "
  timeout 900 pip install --dry-run --quiet --report /tmp/vllm_report.json '$WHEEL' >/dev/null 2>&1
  python3 - <<'PY'
import json
r = json.load(open('/tmp/vllm_report.json'))
out = []
for it in r.get('install', []):
    m = it['metadata']
    if m['name'].lower() == 'vllm':
        continue          # installed separately from the local wheel
    out.append(f\"{m['name']}=={m['version']}\")
open('/tmp/vllm_missing.txt', 'w').write('\n'.join(out) + '\n')
print(f'missing packages: {len(out)}')
PY
  cp /tmp/vllm_missing.txt $OUT/missing.txt
" || exit 1

echo "=== installing $(wc -l < $OUT/missing.txt) deps into $SITE (--no-deps: the set is already closed)"
mkdir -p "$SITE"
docker exec "$WORK_CONTAINER" bash -lc "
  pip install --no-cache-dir --no-warn-script-location --no-deps \
    --target $SITE -r $OUT/missing.txt 2>&1 | tail -3
" || exit 1

echo "=== installing the vLLM wheel itself"
docker exec "$WORK_CONTAINER" bash -lc "
  pip install --no-cache-dir --no-warn-script-location --no-deps \
    --target $SITE '$WHEEL' 2>&1 | tail -3
" || exit 1

echo "=== result"
du -sh "$SITE"
ls "$SITE" | wc -l
