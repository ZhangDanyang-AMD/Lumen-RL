#!/usr/bin/env bash
# Benchmark a Kimi-K3 DSpark draft on ATOM and record acceptance statistics.
#
# One invocation = one fresh server for one draft checkpoint/config. The
# /debug/mtp_stats counters are cumulative since server start, so the server has
# to be restarted between configurations for the numbers to be comparable.
#
#   ./benchmark_dspark.sh <draft_dir> <label> [num_prompts] [max_tokens]
#
# Env overrides:
#   TARGET_MODEL   target checkpoint dir            (default /home/jimguo12/k3-target-official)
#   IMAGE          docker image                     (default rocm/atom-dev:latest)
#   RESULT_DIR     where JSON + logs land           (default /home/jimguo12/k3-bench/results)
#   BENCH_DIR      dir holding run_client.py        (default /home/jimguo12/k3-bench)
#   SERVER_PORT    HTTP port                        (default 8000)
#   NUM_SPEC       --num-speculative-tokens         (default 7)
#   MAX_NUM_SEQS   --max-num-seqs                   (default 8)
#   QUESTION_SETS  space-separated name:path pairs  (default mtbench:$BENCH_DIR/mt_bench_question.jsonl)
#
# max_num_seqs defaults to 8 because that is what the 2026-08-14 runs used and
# because K3's KDA recurrent state is allocated per slot: at 64 the state pool
# alone asks for 28.91 GB against a ~19 GB KV budget and the engine refuses to
# start. The client is serial, so nothing above 1 is exercised anyway.
#
# Multiple question sets share one server on purpose. Loading K3 costs ~20
# minutes, and /debug/mtp_stats is cumulative but the client records the counters
# before and after its own run and reports the delta, so each set still gets an
# isolated measurement. Restarting per set would buy nothing and cost an hour.

set -uo pipefail

DRAFT_DIR="${1:?usage: benchmark_dspark.sh <draft_dir> <label> [num_prompts] [max_tokens]}"
LABEL="${2:?missing label}"
NUM_PROMPTS="${3:-20}"
MAX_TOKENS="${4:-256}"

TARGET_MODEL="${TARGET_MODEL:-/home/jimguo12/k3-target-official}"
IMAGE="${IMAGE:-rocm/atom-dev:latest}"
BENCH_DIR="${BENCH_DIR:-/home/jimguo12/k3-bench}"
RESULT_DIR="${RESULT_DIR:-$BENCH_DIR/results}"
SERVER_PORT="${SERVER_PORT:-8000}"
NUM_SPEC="${NUM_SPEC:-7}"
# recipes/Kimi-K3.md and recipes/DSpark.md both serve K3 with fp8 KV.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"

QUESTION_SETS="${QUESTION_SETS:-mtbench:${BENCH_DIR}/mt_bench_question.jsonl}"

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server_${LABEL}.log"
CONTAINER="atom-dspark-${LABEL}"

echo "=== benchmark_dspark: $LABEL ==="
echo "  target : $TARGET_MODEL"
echo "  draft  : $DRAFT_DIR"
echo "  kv     : $KV_CACHE_DTYPE"
python3 -m json.tool "$DRAFT_DIR/config.json" | grep -E 'rope|mscale' || true

docker rm -f "$CONTAINER" >/dev/null 2>&1

EXTRA_ARGS=()
# ONLINE_QUANT_CONFIG is the recipes/Kimi-K3.md ptpc_fp8 spec. recipes/DSpark.md
# launches K3+DSpark without it, so it stays off unless asked for.
if [[ -n "${ONLINE_QUANT_CONFIG:-}" ]]; then
  SPEC_ARGS_EXTRA=(--online_quant_config "$ONLINE_QUANT_CONFIG")
else
  SPEC_ARGS_EXTRA=()
fi

EAGER_ARGS=()
if [[ -n "${CUDAGRAPH_MODE:-}" ]]; then
  EAGER_ARGS+=(--cudagraph-mode "$CUDAGRAPH_MODE")
fi
# recipes/DSpark.md: "Eager also works for correctness checks" -- the knob that
# separates a real numerics bug from a CUDA-graph replay bug.
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  EAGER_ARGS=(--enforce-eager)
fi

SPEC_ARGS=(--method dspark --draft-model /draft --num-speculative-tokens "$NUM_SPEC")
# The confidence-scheduled ragged verify is the path recipes/DSpark.md actually
# measured; the default here is the batch-uniform one.
if [[ -n "${DSPARK_CONFIG:-}" ]]; then
  SPEC_ARGS+=(--dspark-config "$DSPARK_CONFIG")
fi
# DISABLE_SPEC=1 serves the target alone: the ground truth that greedy
# speculative decoding must reproduce token for token.
if [[ "${DISABLE_SPEC:-0}" == "1" ]]; then
  SPEC_ARGS=()
fi

# No --rm: the container has to survive its own exit so `docker logs` can still
# explain a failed startup.
docker run -d --name "$CONTAINER" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  --ipc=host --shm-size 128g --network host \
  -v "$TARGET_MODEL":/target:ro \
  -v "$DRAFT_DIR":/draft:ro \
  "${EXTRA_ARGS[@]}" \
  "$IMAGE" \
  python -m atom.entrypoints.openai_server \
    --model /target \
    --served-model-name Kimi-K3 \
    "${SPEC_ARGS[@]}" \
    "${SPEC_ARGS_EXTRA[@]}" \
    "${EAGER_ARGS[@]}" \
    --kv_cache_dtype "$KV_CACHE_DTYPE" \
    -tp 8 \
    --trust-remote-code \
    --max-model-len 16384 \
    --max-num-seqs "${MAX_NUM_SEQS:-8}" \
    --max-num-batched-tokens 10240 \
    --gpu-memory-utilization 0.93 \
    --block-size 128 \
    --no-enable_prefix_caching \
    --server-port "$SERVER_PORT" \
  >/dev/null || { echo "docker run failed"; exit 1; }

cleanup() {
  echo "--- stopping server ---"
  docker logs "$CONTAINER" >"$SERVER_LOG" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "--- waiting for /health (timeout ${STARTUP_TIMEOUT}s) ---"
deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null; do
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "server container exited early; last 60 log lines:"
    docker logs "$CONTAINER" 2>&1 | tail -60
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "startup timed out"
    exit 1
  fi
  sleep 10
done
echo "--- server up after ${SECONDS}s ---"

# Which aux-capture path the serving side took, plus the draft's rope decision.
docker logs "$CONTAINER" 2>&1 \
  | grep -aiE "dspark|hidden states extraction|rope|aux" \
  | tail -40 >"$RESULT_DIR/startup_${LABEL}.log"

rc=0
for entry in $QUESTION_SETS; do
  set_name="${entry%%:*}"
  set_path="${entry#*:}"
  if [[ ! -f "$set_path" ]]; then
    echo "!!! question set $set_name not found at $set_path, skipping"
    rc=1
    continue
  fi
  echo
  echo "=== question set: $set_name ($set_path) ==="
  python3 "$BENCH_DIR/run_client.py" \
    --base-url "http://127.0.0.1:${SERVER_PORT}" \
    --model Kimi-K3 \
    --questions "$set_path" \
    --num-prompts "$NUM_PROMPTS" \
    --max-tokens "$MAX_TOKENS" \
    --label "${LABEL}-${set_name}" \
    --out "$RESULT_DIR/bench_${LABEL}-${set_name}.json" || rc=$?
  echo "result -> $RESULT_DIR/bench_${LABEL}-${set_name}.json"
done

exit $rc
