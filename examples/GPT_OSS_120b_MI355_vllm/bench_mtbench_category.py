"""MT-Bench benchmark by category for Eagle3 speculative decoding.

Matches NVIDIA's evaluation format for gpt-oss-120b-Eagle3-long-context:
reports mean acceptance length per MT-Bench category (writing, roleplay,
reasoning, math, coding, extraction, stem, humanities).

Usage:
    python3 bench_mtbench_category.py [--base-url http://localhost:8000]
"""
import argparse
import json
import re
import sys
import time

import requests


def get_model_name(base_url):
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        return resp.json()["data"][0]["id"]
    except Exception:
        return "/dev/shm/gpt-oss-120b"


def get_spec_metrics(base_url):
    try:
        resp = requests.get(f"{base_url}/metrics", timeout=10)
        metrics = {}
        for line in resp.text.split("\n"):
            if "spec_decode" in line and not line.startswith("#"):
                match = re.match(r'^([^\s{]+)(?:\{[^}]*\})?\s+([\d.eE+-]+)', line)
                if match:
                    key = match.group(1)
                    val = float(match.group(2))
                    metrics[key] = metrics.get(key, 0) + val
        return metrics
    except Exception:
        return {}


def load_mtbench():
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    cache_path = "/tmp/mtbench_questions.jsonl"

    import os
    if not os.path.exists(cache_path):
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(r.content)

    questions = []
    with open(cache_path) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            questions.append(q)
    return questions


def query_vllm(base_url, model_name, messages, max_tokens=2048):
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="./benchmark_results")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = get_model_name(args.base_url)
    print(f"Model: {model_name}")

    questions = load_mtbench()
    print(f"Loaded {len(questions)} MT-Bench questions")

    categories = {}
    for q in questions:
        cat = q.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(q)

    print(f"Categories: {sorted(categories.keys())}")
    print()

    results = {}

    for cat in sorted(categories.keys()):
        cat_questions = categories[cat]
        print(f"{'='*60}")
        print(f"Category: {cat} ({len(cat_questions)} questions)")
        print(f"{'='*60}")

        m_before = get_spec_metrics(args.base_url)
        drafts_before = m_before.get("vllm:spec_decode_num_drafts_total", 0)
        accepted_before = m_before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        draft_tokens_before = m_before.get("vllm:spec_decode_num_draft_tokens_total", 0)

        total_output_tokens = 0
        start = time.time()

        for i, q in enumerate(cat_questions):
            msgs = [{"role": "user", "content": q["turns"][0]}]
            try:
                result = query_vllm(args.base_url, model_name, msgs, max_tokens=2048)
                usage = result.get("usage", {})
                total_output_tokens += usage.get("completion_tokens", 0)
            except Exception as e:
                print(f"  ERROR on {i}: {e}")

        elapsed = time.time() - start

        m_after = get_spec_metrics(args.base_url)
        drafts_after = m_after.get("vllm:spec_decode_num_drafts_total", 0)
        accepted_after = m_after.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        draft_tokens_after = m_after.get("vllm:spec_decode_num_draft_tokens_total", 0)

        num_drafts = drafts_after - drafts_before
        num_accepted = accepted_after - accepted_before
        num_draft_tokens = draft_tokens_after - draft_tokens_before

        accept_length = 1.0 + (num_accepted / num_drafts) if num_drafts > 0 else 1.0
        acceptance_rate = (num_accepted / num_draft_tokens * 100) if num_draft_tokens > 0 else 0
        throughput = total_output_tokens / elapsed if elapsed > 0 else 0

        results[cat] = {
            "accept_length": round(accept_length, 2),
            "acceptance_rate": round(acceptance_rate, 1),
            "num_drafts": int(num_drafts),
            "num_accepted": int(num_accepted),
            "total_output_tokens": total_output_tokens,
            "throughput_tps": round(throughput, 1),
            "latency_s": round(elapsed, 1),
        }

        print(f"  Accept length: {accept_length:.2f}")
        print(f"  Acceptance rate: {acceptance_rate:.1f}%")
        print(f"  Throughput: {throughput:.1f} tok/s")
        print(f"  Time: {elapsed:.1f}s")
        print()
        sys.stdout.flush()

    # Summary table
    print()
    print(f"{'='*70}")
    print(f"  MT-Bench Results by Category (checkpoint_5500, step 5500)")
    print(f"{'='*70}")
    print(f"{'Category':<20} {'Accept Length':>14} {'NVIDIA Ref':>12} {'Gap':>8}")
    print(f"{'-'*70}")

    nvidia_ref = {
        "writing": 2.24, "roleplay": 2.25, "reasoning": 2.47,
        "math": 2.83, "coding": 2.51, "extraction": 2.53,
        "stem": 2.17, "humanities": 1.95,
    }

    total_drafts = 0
    total_accepted = 0
    for cat in sorted(results.keys()):
        r = results[cat]
        ref = nvidia_ref.get(cat, None)
        gap_str = f"{r['accept_length'] - ref:+.2f}" if ref else "N/A"
        ref_str = f"{ref:.2f}" if ref else "N/A"
        print(f"{cat:<20} {r['accept_length']:>14.2f} {ref_str:>12} {gap_str:>8}")
        total_drafts += r["num_drafts"]
        total_accepted += r["num_accepted"]

    overall = 1.0 + (total_accepted / total_drafts) if total_drafts > 0 else 1.0
    nvidia_avg = sum(nvidia_ref.values()) / len(nvidia_ref)
    print(f"{'-'*70}")
    print(f"{'OVERALL':<20} {overall:>14.2f} {nvidia_avg:>12.2f} {overall - nvidia_avg:>+8.2f}")
    print(f"{'='*70}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.output_dir, f"mtbench_category_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump({"checkpoint": "checkpoint_5500", "step": 5500,
                    "draft_length": 3, "results": results,
                    "nvidia_ref": nvidia_ref}, f, indent=2)
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
