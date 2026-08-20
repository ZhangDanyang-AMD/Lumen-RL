#!/usr/bin/env bash
# Launch Qwen3-8B GRPO on verl with LumenRL plugin (FSDP2 + ATOM).
#
# Prerequisites:
#   - verl and lumenrl both installed (`pip install -e .` in each repo)
#   - The verl.plugins entry-point is active (default: VERL_USE_EXTERNAL_PLUGINS=auto)
#   - Model and data downloaded to $DATA_ROOT
#
# Usage:
#   DATA_ROOT=/data bash examples/plugin/verl/run.sh          # 3-step smoke
#   DATA_ROOT=/data STEPS=200 bash examples/plugin/verl/run.sh  # long run
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing models/ and data_cached/}"
STEPS="${STEPS:-3}"
CONFIG="${CONFIG:-examples/plugin/verl/qwen3_8b_grpo.yaml}"
LOG="${LOG:-${DATA_ROOT}/logs/verl-plugin-qwen3-8b-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=7200

# Verify LumenRL plugin is registered
python3 -c "
import verl  # triggers plugin auto-discovery
from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.engine.base import EngineRegistry
assert ('atom', 'async') in _ROLLOUT_REGISTRY, 'ATOM rollout not registered'
print('LumenRL plugin: ATOM rollout registered')
try:
    cls = EngineRegistry.get_engine_cls('language_model', 'lumenrl_fsdp2')
    print(f'LumenRL plugin: lumenrl_fsdp2 engine registered -> {cls.__name__}')
except Exception as e:
    print(f'Warning: lumenrl_fsdp2 engine not registered: {e}')
"

echo "=== verl + LumenRL plugin: Qwen3-8B GRPO ==="
echo "    STEPS=$STEPS  CONFIG=$CONFIG  LOG=$LOG"

python3 -m verl.trainer.main_ppo \
  --config-path "$(dirname "$CONFIG")" \
  --config-name "$(basename "$CONFIG" .yaml)" \
  trainer.total_training_steps="$STEPS" \
  2>&1 | tee "$LOG"
