#!/usr/bin/env bash
# Benchmark Eagle3 Phase 2 model using ATOM with speculative decoding on MI350
# Uses MXFP4 target model + trained Eagle3 phase2 draft model
set -uo pipefail

CONTAINER_NAME="kimi_k25_eagle3_v2_phase2_bench"
DRAFT_MODEL="/dev/shm/Kimi_K25_eagle3_v2_phase2_HF"
BASE_MODEL="/dev/shm/Kimi-K2.5-MXFP4"
BENCH_SCRIPT="/root/lumenrl/examples/Kimi_K25_SDDD_MI350_vLLM/bench_eagle3_atom.py"
OUTPUT_DIR="/root/lumenrl/examples/Kimi_K25_SDDD_MI350_vLLM/benchmark_results"
PORT=8000

echo "=== ATOM Eagle3 Phase2 Benchmark ==="
echo "Container: ${CONTAINER_NAME}"
echo "Target model: ${BASE_MODEL} (MXFP4)"
echo "Draft model: ${DRAFT_MODEL}"

# Kill any existing server in the container
docker exec "${CONTAINER_NAME}" bash -c "pkill -f 'openai_server\|atom.entrypoints\|api_server' 2>/dev/null || true"
sleep 2

# Clear compile cache and old stats
docker exec "${CONTAINER_NAME}" bash -c "rm -rf /root/.cache/atom/* /tmp/atom_mtp_stats.json 2>/dev/null || true"

echo "Starting ATOM server with Eagle3..."
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
    --num-speculative-tokens 3 \
    --draft-model ${DRAFT_MODEL} \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port ${PORT} \
    --level 0 \
    2>&1 | tee /tmp/atom_eagle3_bench.log
"

echo "Waiting for ATOM server to be ready..."
for i in $(seq 1 180); do
    if docker exec "${CONTAINER_NAME}" curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after $((i*5)) seconds!"
        break
    fi
    if [ "$i" -eq 180 ]; then
        echo "ERROR: Server failed to start after 15 minutes"
        echo "Last log lines:"
        docker exec "${CONTAINER_NAME}" tail -30 /tmp/atom_eagle3_bench.log
        exit 1
    fi
    sleep 5
done

# Verify GPU is loaded
sleep 5

echo "=== Running benchmarks ==="
docker exec "${CONTAINER_NAME}" python3 "${BENCH_SCRIPT}" \
    --base-url "http://localhost:${PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --phase phase2 \
    --draft-model "${DRAFT_MODEL}" \
    --benchmarks mtbench ceval gsm8k humaneval math500 aime 2>&1

echo "=== Benchmark complete ==="
echo "Stopping ATOM server..."
docker exec "${CONTAINER_NAME}" bash -c "pkill -f 'openai_server\|atom.entrypoints\|api_server' 2>/dev/null || true"
