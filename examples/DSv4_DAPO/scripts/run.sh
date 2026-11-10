#!/usr/bin/env bash
# Run DSv4 DAPO: check -> containers -> Ray -> GPU count -> driver (in the background).
# Run it on the head node (the first entry of NODES, or the only machine).
#
#   bash scripts/run.sh                 # STEPS from env.sh (default 3)
#   STEPS=1 bash scripts/run.sh
#   EXTRA="policy.foo=bar" bash scripts/run.sh   # extra Hydra overrides, appended last
#
# The driver runs detached inside the container (`docker exec -d`), so this script
# returns once training has started. Poll with scripts/status.sh.
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
detect_head_ip
q() { printf %q "$1"; }

if [ -z "${CONFIG:-}" ]; then
  if [ "$DSV4_LAYERS" -lt "$MODEL_LAYERS_FULL" ]; then
    CONFIG=examples/DAPO/configs/dapo_dsv4_flash_ray_vllm_megatron_1node_smoke.yaml
  else
    die "DSV4_LAYERS=$DSV4_LAYERS: this branch has no full-model training config or the code it needs (see README, 全量 43 层). Set CONFIG=<config relative to the Lumen-RL root> to try one anyway."
  fi
fi

# ---- 1. preflight -------------------------------------------------------------
[ "${SKIP_CHECK:-0}" = 1 ] || bash "$DSV4_EXAMPLE_DIR/scripts/check_env.sh"
for f in "$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh" "$RL_ROOT/Lumen-RL/$CONFIG" \
         "$RL_ROOT/aiter/aiter/__init__.py" "$RL_ROOT/Lumen/lumen/config.py" \
         "$MEGATRON_DSV4_DIR/.tree.sha256" "$TRAIN_FILE" "$VAL_FILE_PATH" \
         "$NATIVE_DIR/.dsv4-done" "$DIST_DIR/.dsv4-done"; do
  [ -e "$f" ] || die "missing $f -- run the setup steps in the README first"
done
[ "$(cat "$MEGATRON_DSV4_DIR/.tree.sha256")" = "$MEGATRON_TREE_SHA256" ] \
  || die "$MEGATRON_DSV4_DIR does not match the expected tree; rerun scripts/build_megatron.sh"

# ---- 2/3. containers and Ray --------------------------------------------------
bash "$DSV4_EXAMPLE_DIR/scripts/start_container.sh"
bash "$DSV4_EXAMPLE_DIR/scripts/start_ray.sh" head
probe_gpus
GPUS_PER_NODE=$GPU_COUNT
if is_multi_node; then
  while read -r host; do
    log "---- $host: container + ray worker ----"
    remote_script "$host" start_container.sh
    remote_script "$host" start_ray.sh worker
  done < <(remote_nodes)
fi

# ---- 4. GPU count -------------------------------------------------------------
TOTAL=$((NUM_NODES * GPUS_PER_NODE))
NUM_WORKERS=$TOTAL
log "waiting for $NUM_NODES node(s) / $TOTAL GPU(s) to join Ray at $HEAD_IP:$RAY_PORT"
dk exec "$CONTAINER" python3 - "$HEAD_IP:$RAY_PORT" "$TOTAL" <<'PY'
import sys, time
import ray
addr, want = sys.argv[1], int(sys.argv[2])
ray.init(address=addr, logging_level="ERROR")
for _ in range(60):
    n = int(ray.cluster_resources().get("GPU", 0))
    if n >= want:
        break
    time.sleep(2)
nodes = sum(1 for x in ray.nodes() if x["Alive"])
print(f"ray cluster: {nodes} node(s), {n} GPU(s)")
sys.exit(0 if n == want else 1)
PY

# ---- 5. driver ----------------------------------------------------------------
OVERRIDE="cluster.num_nodes=$NUM_NODES cluster.gpus_per_node=$GPUS_PER_NODE"
OVERRIDE+=" cluster.ray_address=$HEAD_IP:$RAY_PORT controller.ray.actor.num_workers=$NUM_WORKERS"
if is_multi_node; then
  # One bundle per node (a STRICT_SPREAD placement group), so a TP=8 rollout replica
  # never straddles two nodes -- VLLMReplicaManager rejects that.
  layout=""
  for ((i = 0; i < NUM_NODES; i++)); do layout+="$GPUS_PER_NODE,"; done
  layout="[${layout%,}]"
  OVERRIDE+=" controller.ray.actor.process_on_nodes=$layout"
fi
# Neither path is taken from the YAML: both follow from DSV4_LAYERS.
OVERRIDE+=" policy.training.megatron_cfg.dist_checkpoint_path=$DIST_DIR"
[ -n "${EXTRA:-}" ] && OVERRIDE+=" $EXTRA"

mkdir -p "$LOG_DIR"
RUN_ID=${RUN_ID:-dsv4-L$DSV4_LAYERS-${NUM_NODES}x$GPUS_PER_NODE-$(date -u +%Y%m%dT%H%M%SZ)}
LOG=$LOG_DIR/$RUN_ID.log
echo "$RUN_ID" >"$LOG_DIR/latest"
cat >"$LOG.cmd" <<EOF
CONFIG_OVERRIDE=$CONFIG
MODEL_PATH=$NATIVE_DIR
TRAIN_FILE=$TRAIN_FILE
STEPS=$STEPS
MODE=bf16
EXTRA_OVERRIDE=$OVERRIDE
EOF

dk exec -d -e NET_IF="$NET_IF" -e DSV4_EXAMPLE_DIR="$DSV4_EXAMPLE_DIR" \
  -e MEGATRON_DSV4_DIR="$MEGATRON_DSV4_DIR" -e DSV4_SRC_MODEL_DIR="$DSV4_SRC_MODEL_DIR" \
  "$CONTAINER" bash -lc "
source \"\$DSV4_EXAMPLE_DIR/runtime/ray_env.sh\"
# Swallows run_dapo.sh's opening 'ray stop --force'.
export PATH=\"\$DSV4_EXAMPLE_DIR/runtime/bin:\$PATH\"
export SCRATCH_ROOT=$(q "$WORK_DIR")
# Both are '\${VAR-default}' inside run_dapo.sh: an explicit empty value turns
# expandable_segments off (ROCm has none).
export PYTORCH_CUDA_ALLOC_CONF=
CONFIG_OVERRIDE=$(q "$CONFIG") \
MODEL_PATH=$(q "$NATIVE_DIR") \
TRAIN_FILE=$(q "$TRAIN_FILE") \
VAL_FILE='' \
STEPS=$(q "$STEPS") \
MODE=bf16 \
LOG=$(q "$LOG") \
EXTRA_OVERRIDE=$(q "$OVERRIDE") \
  bash \"\$RL_ROOT/Lumen-RL/examples/DAPO/run_dapo.sh\" > $(q "$LOG.launch") 2>&1
echo \$? > $(q "$LOG.exit")
"

log "driver started: $RUN_ID"
log "  training log: $LOG"
log "  launcher log: $LOG.launch   exit code -> $LOG.exit when it ends"
log "poll with:  bash $DSV4_EXAMPLE_DIR/scripts/status.sh        (or: tail -f $LOG)"
