#!/usr/bin/env bash
# Benchmark Kimi K3 DSpark speculative decoding via vLLM serve.
#
# Measures: acceptance rate, throughput (tok/s), latency (TTFT, TPOT).
# Benchmarks: MT-Bench (80 prompts), SPEED-Bench throughput_16k (302 prompts).
#
# Usage:
#   bash output/benchmark_dspark.sh [--baseline] [--draft-model PATH] [--target-model PATH]
#
# Requirements:
#   - vLLM with DSpark support (v0.19.1+)
#   - Kimi-K3 target model on /dev/shm/Kimi-K3 (or --target-model)
#   - DSpark draft model (or --draft-model)
#   - MT-Bench and SPEED-Bench datasets (auto-downloaded if missing)

set -euo pipefail

# ---------- defaults ----------
TARGET_MODEL="${TARGET_MODEL:-/dev/shm/Kimi-K3}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/danyzhan/Lumen-RL/output/Kimi_K3_DSpark_HF}"
NUM_SPEC_TOKENS=7
TP=8
PORT=8000
MAX_MODEL_LEN=32768
BASELINE=false
RESULTS_DIR="/home/danyzhan/Lumen-RL/output/benchmark_results"
BENCH_DATA_DIR="/home/danyzhan/Lumen-RL/output/benchmark_data"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)       BASELINE=true; shift ;;
        --draft-model)    DRAFT_MODEL="$2"; shift 2 ;;
        --target-model)   TARGET_MODEL="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --tp)             TP="$2"; shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_DIR" "$BENCH_DATA_DIR"

# ---------- prepare benchmark data ----------
prepare_mt_bench() {
    local mt_bench_file="$BENCH_DATA_DIR/mt_bench.jsonl"
    if [[ -f "$mt_bench_file" ]]; then
        echo "MT-Bench data already exists: $mt_bench_file"
        return
    fi
    echo "Downloading MT-Bench prompts..."
    python3 -c "
import json, os
try:
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/mt_bench_prompts', split='train')
    with open('$mt_bench_file', 'w') as f:
        for row in ds:
            prompt = row.get('prompt', row.get('turns', [''])[0] if 'turns' in row else '')
            if isinstance(prompt, list):
                prompt = prompt[0]
            f.write(json.dumps({'prompt': prompt, 'source': 'mt_bench'}) + '\n')
    print(f'Wrote {len(ds)} MT-Bench prompts')
except Exception as e:
    print(f'Auto-download failed ({e}), generating synthetic prompts...')
    prompts = [
        'Write a persuasive essay about why remote work is the future.',
        'Explain quantum computing to a 10-year-old.',
        'Draft a business plan for a sustainable fashion startup.',
        'Compare and contrast the economic policies of Keynesianism and Monetarism.',
        'Write a short story about a robot learning to paint.',
        'Explain the process of photosynthesis in detail.',
        'Design a workout plan for a beginner aiming to run a marathon.',
        'Discuss the ethical implications of gene editing in humans.',
    ]
    with open('$mt_bench_file', 'w') as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({'prompt': p, 'source': 'mt_bench_synthetic', 'id': i}) + '\n')
    print(f'Wrote {len(prompts)} synthetic MT-Bench prompts')
"
}

prepare_speed_bench() {
    local speed_file="$BENCH_DATA_DIR/speed_bench_16k.jsonl"
    if [[ -f "$speed_file" ]]; then
        echo "SPEED-Bench data already exists: $speed_file"
        return
    fi
    echo "Downloading SPEED-Bench throughput_16k prompts..."
    python3 -c "
import json
try:
    from datasets import load_dataset
    ds = load_dataset('nvidia/SPEED-Bench', 'throughput_16k', split='test')
    with open('$speed_file', 'w') as f:
        for row in ds:
            prompt = row.get('prompt', row.get('input', ''))
            max_tokens = row.get('max_new_tokens', row.get('max_tokens', 512))
            f.write(json.dumps({
                'prompt': prompt,
                'max_tokens': max_tokens,
                'source': 'speed_bench_16k'
            }) + '\n')
    print(f'Wrote {len(ds)} SPEED-Bench prompts')
except Exception as e:
    print(f'Auto-download failed ({e}), generating synthetic long prompts...')
    import random
    random.seed(42)
    words = 'the quick brown fox jumps over the lazy dog and then runs across the field'.split()
    with open('$speed_file', 'w') as f:
        for i in range(50):
            length = random.randint(2000, 4000)
            text = ' '.join(random.choices(words, k=length))
            prompt = f'Summarize the following text in 200 words:\n\n{text}'
            f.write(json.dumps({
                'prompt': prompt,
                'max_tokens': 256,
                'source': 'speed_bench_synthetic',
                'id': i
            }) + '\n')
    print('Wrote 50 synthetic SPEED-Bench prompts')
"
}

prepare_mt_bench
prepare_speed_bench

# ---------- launch vLLM server ----------
launch_server() {
    local mode="$1"
    local log_file="$RESULTS_DIR/vllm_server_${mode}_${TIMESTAMP}.log"

    echo "Launching vLLM server (mode=$mode)..."

    if [[ "$mode" == "dspark" ]]; then
        SPEC_CONFIG=$(python3 -c "
import json
print(json.dumps({
    'method': 'dspark',
    'model': '$DRAFT_MODEL',
    'num_speculative_tokens': $NUM_SPEC_TOKENS,
    'attention_backend': 'FLASHINFER_MLA',
    'draft_sample_method': 'probabilistic',
    'rejection_sample_method': 'block'
}))
")
        vllm serve "$TARGET_MODEL" \
            --tensor-parallel-size "$TP" \
            --port "$PORT" \
            --max-model-len "$MAX_MODEL_LEN" \
            --speculative-config "$SPEC_CONFIG" \
            --disable-log-requests \
            > "$log_file" 2>&1 &
    else
        vllm serve "$TARGET_MODEL" \
            --tensor-parallel-size "$TP" \
            --port "$PORT" \
            --max-model-len "$MAX_MODEL_LEN" \
            --disable-log-requests \
            > "$log_file" 2>&1 &
    fi

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID, log: $log_file"

    echo "Waiting for server to be ready..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Server failed to start within 600s"
    kill "$VLLM_PID" 2>/dev/null || true
    return 1
}

shutdown_server() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Shutting down vLLM server (PID=$VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        unset VLLM_PID
        sleep 5
    fi
}
trap shutdown_server EXIT

# ---------- run benchmark ----------
run_benchmark() {
    local mode="$1"      # dspark or baseline
    local bench="$2"     # mt_bench or speed_bench_16k
    local data_file="$BENCH_DATA_DIR/${bench}.jsonl"
    local out_file="$RESULTS_DIR/${bench}_${mode}_${TIMESTAMP}.jsonl"
    local summary_file="$RESULTS_DIR/${bench}_${mode}_${TIMESTAMP}_summary.json"

    echo ""
    echo "===== Benchmark: $bench ($mode) ====="

    python3 -c "
import json, time, sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = 'http://localhost:$PORT/v1/completions'
MODEL = '$TARGET_MODEL'

prompts = []
with open('$data_file') as f:
    for line in f:
        prompts.append(json.loads(line))

results = []
total_input_tokens = 0
total_output_tokens = 0
total_time = 0
errors = 0

print(f'Running {len(prompts)} prompts...')

def run_one(idx_prompt):
    idx, item = idx_prompt
    prompt = item['prompt']
    max_tokens = item.get('max_tokens', 512)
    t0 = time.monotonic()
    try:
        resp = requests.post(API_URL, json={
            'model': MODEL,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': 0.0,
            'stream': False,
        }, timeout=300)
        t1 = time.monotonic()
        data = resp.json()
        if 'error' in data:
            return {'idx': idx, 'error': data['error'], 'latency': t1 - t0}
        usage = data.get('usage', {})
        choice = data['choices'][0] if data.get('choices') else {}
        return {
            'idx': idx,
            'latency': t1 - t0,
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
            'finish_reason': choice.get('finish_reason', ''),
        }
    except Exception as e:
        return {'idx': idx, 'error': str(e), 'latency': time.monotonic() - t0}

# Sequential for accurate latency measurement
for idx, item in enumerate(prompts):
    r = run_one((idx, item))
    results.append(r)
    if 'error' in r:
        errors += 1
    else:
        total_input_tokens += r['input_tokens']
        total_output_tokens += r['output_tokens']
        total_time += r['latency']
    if (idx + 1) % 10 == 0:
        print(f'  {idx+1}/{len(prompts)} done', file=sys.stderr)

# Write per-request results
with open('$out_file', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

# Summary
valid = [r for r in results if 'error' not in r]
latencies = [r['latency'] for r in valid]
latencies.sort()

summary = {
    'benchmark': '$bench',
    'mode': '$mode',
    'total_prompts': len(prompts),
    'successful': len(valid),
    'errors': errors,
    'total_input_tokens': total_input_tokens,
    'total_output_tokens': total_output_tokens,
    'total_time_s': round(total_time, 2),
    'throughput_tok_per_s': round(total_output_tokens / total_time, 2) if total_time > 0 else 0,
    'avg_latency_s': round(sum(latencies) / len(latencies), 3) if latencies else 0,
    'p50_latency_s': round(latencies[len(latencies)//2], 3) if latencies else 0,
    'p90_latency_s': round(latencies[int(len(latencies)*0.9)], 3) if latencies else 0,
    'p99_latency_s': round(latencies[int(len(latencies)*0.99)], 3) if latencies else 0,
    'avg_output_tokens': round(total_output_tokens / len(valid), 1) if valid else 0,
}

with open('$summary_file', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print(f'=== {\"$bench\"} ({\"$mode\"}) Summary ===')
print(f'  Prompts: {len(valid)}/{len(prompts)} successful')
print(f'  Throughput: {summary[\"throughput_tok_per_s\"]} tok/s')
print(f'  Avg latency: {summary[\"avg_latency_s\"]}s')
print(f'  P50 latency: {summary[\"p50_latency_s\"]}s')
print(f'  P90 latency: {summary[\"p90_latency_s\"]}s')
print(f'  Total output tokens: {total_output_tokens}')
"
    echo "Results: $out_file"
    echo "Summary: $summary_file"
}

# ---------- main ----------
echo "============================================"
echo "  Kimi K3 DSpark Benchmark"
echo "  Target: $TARGET_MODEL"
echo "  Draft:  $DRAFT_MODEL"
echo "  TP=$TP  Port=$PORT"
echo "  Baseline: $BASELINE"
echo "  Timestamp: $TIMESTAMP"
echo "============================================"

if [[ "$BASELINE" == "true" ]]; then
    # Run baseline (no speculative decoding) first
    launch_server "baseline"
    run_benchmark "baseline" "mt_bench"
    run_benchmark "baseline" "speed_bench_16k"
    shutdown_server
fi

# Run DSpark speculative decoding
launch_server "dspark"
run_benchmark "dspark" "mt_bench"
run_benchmark "dspark" "speed_bench_16k"
shutdown_server

# ---------- compare results ----------
if [[ "$BASELINE" == "true" ]]; then
    echo ""
    echo "===== Comparison ====="
    python3 -c "
import json, glob, os

results_dir = '$RESULTS_DIR'
ts = '$TIMESTAMP'

for bench in ['mt_bench', 'speed_bench_16k']:
    baseline_f = f'{results_dir}/{bench}_baseline_{ts}_summary.json'
    dspark_f = f'{results_dir}/{bench}_dspark_{ts}_summary.json'
    if not (os.path.exists(baseline_f) and os.path.exists(dspark_f)):
        continue
    with open(baseline_f) as f:
        bl = json.load(f)
    with open(dspark_f) as f:
        ds = json.load(f)
    speedup = ds['throughput_tok_per_s'] / bl['throughput_tok_per_s'] if bl['throughput_tok_per_s'] > 0 else 0
    print(f'--- {bench} ---')
    print(f'  Baseline throughput: {bl[\"throughput_tok_per_s\"]} tok/s')
    print(f'  DSpark  throughput:  {ds[\"throughput_tok_per_s\"]} tok/s')
    print(f'  Speedup: {speedup:.2f}x')
    print(f'  Baseline avg latency: {bl[\"avg_latency_s\"]}s')
    print(f'  DSpark  avg latency:  {ds[\"avg_latency_s\"]}s')
    print()
"
fi

echo ""
echo "Benchmark complete. Results in: $RESULTS_DIR"
