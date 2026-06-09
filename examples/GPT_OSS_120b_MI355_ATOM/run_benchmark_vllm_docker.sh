#!/usr/bin/env bash
# Benchmark GPT-OSS-120B Eagle3 draft using vLLM with speculative decoding on MI350.
#
# Prerequisites:
#   1. Trained draft checkpoint exported to HuggingFace safetensors at ${DRAFT_MODEL}.
#   2. Base model present at ${BASE_MODEL} (native MXFP4 ~196 GB).
#
# Usage:
#   bash examples/GPT_OSS_120b_MI355_ATOM/run_benchmark_vllm_docker.sh
#   DRAFT_MODEL=/path/to/draft_HF bash examples/GPT_OSS_120b_MI355_ATOM/run_benchmark_vllm_docker.sh
set -uo pipefail

DOCKER_IMAGE="${DOCKER_IMAGE:-lumenrl-vllm-mi350:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-gpt_oss_120b_eagle3_benchmark}"
DRAFT_MODEL="${DRAFT_MODEL:-/dev/shm/gpt_oss_120b_eagle3_HF}"
BASE_MODEL="${BASE_MODEL:-/dev/shm/gpt-oss-120b}"
LUMENRL_DIR="${LUMENRL_DIR:-/home/danyzhan/Lumen-RL}"
BENCH_SCRIPT="${BENCH_SCRIPT:-${LUMENRL_DIR}/examples/Kimi_K25_SDDD_MI350_ATOM/bench_eagle3_vllm.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${LUMENRL_DIR}/examples/GPT_OSS_120b_MI355_ATOM/benchmark_results}"

TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
CONCURRENCY="${CONCURRENCY:-1}"

mkdir -p "${OUTPUT_DIR}"

for path in "${BASE_MODEL}" "${DRAFT_MODEL}" "${BENCH_SCRIPT}"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: required path missing: ${path}" >&2
        exit 1
    fi
done

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "═══════════════════════════════════════════════════════════════"
echo "  GPT-OSS-120B Eagle3 vLLM Benchmark — MI350"
echo "  Image:       ${DOCKER_IMAGE}"
echo "  Container:   ${CONTAINER_NAME}"
echo "  Base model:  ${BASE_MODEL} (native MXFP4)"
echo "  Draft model: ${DRAFT_MODEL}"
echo "  TP / spec:   ${TP_SIZE} / ${NUM_SPEC_TOKENS}"
echo "  Max seq:    ${MAX_MODEL_LEN}"
echo "  Output:      ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

docker run -d \
    --name "${CONTAINER_NAME}" \
    --network host \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size 64G \
    -v /dev/shm:/dev/shm \
    -v "${LUMENRL_DIR}:${LUMENRL_DIR}" \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e PYTORCH_ROCM_ARCH=gfx950 \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e HSA_COREDUMP_FILE=/dev/null \
    -e HSA_ENABLE_COREDUMP=0 \
    -e VLLM_PLUGINS='' \
    -e CONCURRENCY="${CONCURRENCY}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "${DOCKER_IMAGE}" \
    bash -c "
set -e

echo '=== Starting vLLM server with Eagle3 speculative decoding (gpt-oss-120b, MXFP4) ==='
vllm serve ${BASE_MODEL} \
    --speculative-config '{\"model\": \"${DRAFT_MODEL}\", \"method\": \"eagle3\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}}' \
    --tensor-parallel-size ${TP_SIZE} \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --host 0.0.0.0 \
    --port 8000 &

SERVER_PID=\$!

echo 'Waiting for server to be ready (up to 30 min)...'
for i in \$(seq 1 360); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo 'Server ready!'
        break
    fi
    if ! kill -0 \$SERVER_PID 2>/dev/null; then
        echo 'ERROR: Server process died before becoming ready'
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo 'ERROR: Server failed to start after 30 minutes'
    exit 1
fi

echo '=== Running benchmarks ==='
pip install requests pyarrow 2>/dev/null
rm -rf ~/.cache/benchmark_data

python3 ${BENCH_SCRIPT} \
    --base-url http://localhost:8000 \
    --output-dir ${OUTPUT_DIR} \
    --concurrency ${CONCURRENCY} \
    --benchmarks all 2>&1

echo '=== All benchmarks complete ==='
kill \${SERVER_PID} 2>/dev/null || true
"

echo "Container started. Monitor with:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo "Results land in ${OUTPUT_DIR}/"
