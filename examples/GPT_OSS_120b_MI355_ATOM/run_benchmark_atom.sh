#!/usr/bin/env bash
# Benchmark Eagle3 speculative decoding for gpt-oss-120b using ATOM on MI350
# Uses BF16 target model + trained Eagle3 draft model (spec_length=3)
#
# Usage:
#   bash examples/GPT_OSS_120b_MI355_ATOM/run_benchmark_atom.sh
#   bash examples/GPT_OSS_120b_MI355_ATOM/run_benchmark_atom.sh --benchmarks mtbench_categories
#   bash examples/GPT_OSS_120b_MI355_ATOM/run_benchmark_atom.sh --benchmarks all
set -uo pipefail

CONTAINER_NAME="gpt_oss_120b_export"
DRAFT_MODEL="/home/danyzhan/gpt_oss_120b_eagle3_HF"
BASE_MODEL="/dev/shm/gpt-oss-120b"
BENCH_SCRIPT="/home/danyzhan/Lumen-RL/examples/GPT_OSS_120b_MI355_ATOM/bench_eagle3_atom.py"
OUTPUT_DIR="/home/danyzhan/Lumen-RL/examples/GPT_OSS_120b_MI355_ATOM/benchmark_results"
PORT=8000
SPEC_LENGTH=3
STEP=15800

# Forward extra args (e.g. --benchmarks all)
EXTRA_ARGS="${@}"
if [ -z "${EXTRA_ARGS}" ]; then
    EXTRA_ARGS="--benchmarks mtbench_categories"
fi

echo "=== ATOM Eagle3 Benchmark (gpt-oss-120b) ==="
echo "Container: ${CONTAINER_NAME}"
echo "Target model: ${BASE_MODEL} (BF16)"
echo "Draft model: ${DRAFT_MODEL}"
echo "Spec length: ${SPEC_LENGTH}"

# Kill any existing server in the container
docker exec "${CONTAINER_NAME}" bash -c "pkill -f 'openai_server\|atom.entrypoints' 2>/dev/null || true"
sleep 2

# Clear compile cache
docker exec "${CONTAINER_NAME}" bash -c "rm -rf /root/.cache/atom/* 2>/dev/null || true"

echo "Starting ATOM server with Eagle3 (TP=8, spec_length=${SPEC_LENGTH})..."
docker exec -d "${CONTAINER_NAME}" bash -c "
cd /root/ATOM && \
AITER_LOG_LEVEL=WARNING \
ATOM_LOGGING_LEVEL=WARNING \
python3 -m atom.entrypoints.openai.api_server \
    --model ${BASE_MODEL} \
    --trust-remote-code \
    --kv_cache_dtype fp8 \
    -tp 8 \
    --method eagle3 \
    --num-speculative-tokens ${SPEC_LENGTH} \
    --draft-model ${DRAFT_MODEL} \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port ${PORT} \
    --level 0 \
    2>&1 | tee /tmp/atom_eagle3_gpt_oss_bench.log
"

echo "Waiting for ATOM server to be ready..."
for i in $(seq 1 180); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after $((i*5)) seconds!"
        break
    fi
    if [ "$i" -eq 180 ]; then
        echo "ERROR: Server failed to start after 15 minutes"
        echo "Last log lines:"
        docker exec "${CONTAINER_NAME}" tail -30 /tmp/atom_eagle3_gpt_oss_bench.log
        exit 1
    fi
    sleep 5
done

sleep 5

echo "=== Running benchmarks ==="
python3 "${BENCH_SCRIPT}" \
    --base-url "http://localhost:${PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --step "${STEP}" \
    --draft-model "${DRAFT_MODEL}" \
    --spec-length "${SPEC_LENGTH}" \
    ${EXTRA_ARGS} 2>&1

echo "=== Benchmark complete ==="
echo "Stopping ATOM server..."
docker exec "${CONTAINER_NAME}" bash -c "pkill -f 'openai_server\|atom.entrypoints' 2>/dev/null || true"
