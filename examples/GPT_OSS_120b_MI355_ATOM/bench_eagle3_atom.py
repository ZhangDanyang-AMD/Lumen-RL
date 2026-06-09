"""
Benchmark Eagle3 speculative decoding with ATOM for gpt-oss-120b.

Measures accept_length (average accepted tokens per speculation step) across
MT-Bench categories (matching NVIDIA's gpt-oss-120b-Eagle3-long-context evaluation)
and additional benchmarks using ATOM's /debug/mtp_stats endpoint.

NVIDIA reference (draft_length=3, TensorRT-LLM on B200):
  writing=2.24, roleplay=2.25, reasoning=2.47, math=2.83,
  coding=2.51, extraction=2.53, stem=2.17, humanities=1.95
"""

import argparse
import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests

CACHE_DIR = os.path.expanduser("~/.cache/benchmark_data")


def download_file(url, filename):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        return path
    print(f"Downloading {filename}...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def get_model_name(base_url):
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        data = resp.json()
        return data["data"][0]["id"]
    except Exception:
        return "/dev/shm/gpt-oss-120b"


def load_mtbench_by_category():
    """Load MT-Bench questions grouped by category (8 categories, 10 each)."""
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    path = download_file(url, "mtbench.jsonl")
    by_cat = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            cat = q.get("category", "unknown")
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append([{"role": "user", "content": q["turns"][0]}])
    return by_cat


def load_mtbench_all(n):
    path = download_file(
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
        "mtbench.jsonl",
    )
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            prompts.append([{"role": "user", "content": q["turns"][0]}])
    return prompts[:n]


def load_gsm8k(n):
    path = download_file(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "gsm8k_test.jsonl",
    )
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            prompts.append([{"role": "user", "content": "Question: " + q["question"] + "\nAnswer:"}])
    return prompts[:n]


def load_humaneval(n):
    path = download_file(
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "humaneval.jsonl.gz",
    )
    prompts = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            prompts.append([{"role": "user", "content": "Complete the following Python function:\n\n" + q["prompt"]}])
    return prompts[:n]


def load_math500(n):
    path = download_file(
        "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
        "math500_test.jsonl",
    )
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            prompts.append([{"role": "user", "content": q["problem"]}])
    return prompts[:n]


def load_aime(n):
    path = download_file(
        "https://huggingface.co/datasets/HuggingFaceH4/aime_2024/resolve/main/data/train-00000-of-00001.parquet",
        "aime2024.parquet",
    )
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    prompts = []
    for row in table.to_pydict()["problem"][:n]:
        prompts.append([{"role": "user", "content": row}])
    if len(prompts) < n:
        prompts = prompts * (n // len(prompts) + 1)
    return prompts[:n]


def query_server(base_url, model_name, messages, max_tokens=2048):
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


def get_atom_mtp_stats(base_url):
    try:
        resp = requests.get(f"{base_url}/debug/mtp_stats", timeout=10)
        return resp.json()
    except Exception:
        return {}


def run_benchmark(base_url, model_name, name, prompts, max_tokens=2048, batch_size=1, spec_length=3):
    print(f"\n{'='*60}")
    print(f"Running: {name} ({len(prompts)} samples, BS={batch_size})")
    print(f"{'='*60}")

    stats_before = get_atom_mtp_stats(base_url)
    draft_before = stats_before.get("total_draft_tokens", 0)
    accepted_before = stats_before.get("total_accepted_tokens", 0)

    total_output_tokens = 0
    errors = 0
    done_count = 0
    lock = Lock()
    start = time.time()

    def _worker(msgs):
        return query_server(base_url, model_name, msgs, max_tokens=max_tokens)

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_worker, msgs): i for i, msgs in enumerate(prompts)}
        for future in as_completed(futures):
            try:
                result = future.result()
                usage = result.get("usage", {})
                with lock:
                    total_output_tokens += usage.get("completion_tokens", 0)
                    done_count += 1
                    if done_count % 20 == 0:
                        elapsed = time.time() - start
                        print(f"  Progress: {done_count}/{len(prompts)} ({elapsed:.1f}s)")
            except Exception as e:
                with lock:
                    errors += 1
                    done_count += 1
                    if errors <= 3:
                        print(f"  ERROR on sample {futures[future]}: {e}")

    elapsed = time.time() - start
    if errors > 3:
        print(f"  ... {errors} total errors")

    stats_after = get_atom_mtp_stats(base_url)
    draft_after = stats_after.get("total_draft_tokens", 0)
    accepted_after = stats_after.get("total_accepted_tokens", 0)

    num_draft_tokens = draft_after - draft_before
    num_accepted = accepted_after - accepted_before

    num_steps = num_draft_tokens // spec_length if spec_length > 0 else 0
    accept_length = 1.0 + (num_accepted / num_steps) if num_steps > 0 else 1.0
    acceptance_rate = (num_accepted / num_draft_tokens * 100) if num_draft_tokens > 0 else 0

    dist_before = stats_before.get("distribution", {})
    dist_after = stats_after.get("distribution", {})
    distribution = {}
    for k in dist_after:
        distribution[k] = dist_after[k] - dist_before.get(k, 0)

    throughput = total_output_tokens / elapsed if elapsed > 0 else 0

    result = {
        "benchmark": name,
        "num_samples": len(prompts),
        "accept_length": round(accept_length, 3),
        "acceptance_rate": round(acceptance_rate, 1),
        "total_output_tokens": total_output_tokens,
        "throughput_tps": round(throughput, 1),
        "latency_s": round(elapsed, 1),
        "num_draft_tokens": int(num_draft_tokens),
        "num_accepted": int(num_accepted),
        "distribution": distribution,
        "errors": errors,
    }

    print(f"  Accept length: {accept_length:.3f}")
    print(f"  Acceptance rate: {acceptance_rate:.1f}%")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  Total time: {elapsed:.1f}s")
    if distribution:
        print(f"  Distribution: {distribution}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--benchmarks", nargs="+", default=["mtbench_categories"])
    parser.add_argument("--step", default="15800", help="Checkpoint step number")
    parser.add_argument("--draft-model", default="/home/danyzhan/gpt_oss_120b_eagle3_HF")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--spec-length", type=int, default=3, help="Speculative tokens (draft_length)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = get_model_name(args.base_url)
    print(f"Using model: {model_name}")
    print(f"Backend: ATOM")
    print(f"Spec length (draft_length): {args.spec_length}")
    print(f"Batch size: {args.batch_size}")

    initial_stats = get_atom_mtp_stats(args.base_url)
    if not initial_stats.get("enabled", False):
        print("WARNING: MTP/Eagle3 stats not enabled — acceptance metrics will be unavailable")

    mtp_k = len(initial_stats.get("distribution", {})) - 1
    if mtp_k <= 0:
        mtp_k = args.spec_length
    print(f"Detected num_speculative_tokens: {mtp_k}")

    all_results = {}

    if "mtbench_categories" in args.benchmarks or "all" in args.benchmarks:
        print("\n" + "=" * 70)
        print("MT-BENCH PER-CATEGORY EVALUATION (matching NVIDIA evaluation)")
        print("=" * 70)
        by_cat = load_mtbench_by_category()
        mtbench_cat_results = {}
        for cat in sorted(by_cat.keys()):
            prompts = by_cat[cat]
            result = run_benchmark(
                args.base_url, model_name, f"mtbench_{cat}", prompts,
                max_tokens=args.max_tokens, batch_size=args.batch_size,
                spec_length=args.spec_length,
            )
            mtbench_cat_results[cat] = result
            all_results[f"mtbench_{cat}"] = result

        print(f"\n{'='*70}")
        print(f"MT-BENCH SUMMARY (NVIDIA comparison: draft_length={args.spec_length})")
        print(f"{'='*70}")
        nvidia_ref = {
            "writing": 2.24, "roleplay": 2.25, "reasoning": 2.47,
            "math": 2.83, "coding": 2.51, "extraction": 2.53,
            "stem": 2.17, "humanities": 1.95,
        }
        print(f"{'Category':<15} {'Ours':>10} {'NVIDIA':>10} {'Delta':>10}")
        print(f"{'-'*45}")
        total_ours = 0
        total_nvidia = 0
        for cat in sorted(nvidia_ref.keys()):
            ours = mtbench_cat_results.get(cat, {}).get("accept_length", 0)
            nv = nvidia_ref[cat]
            delta = ours - nv
            total_ours += ours
            total_nvidia += nv
            print(f"{cat:<15} {ours:>10.3f} {nv:>10.2f} {delta:>+10.3f}")
        avg_ours = total_ours / len(nvidia_ref)
        avg_nv = total_nvidia / len(nvidia_ref)
        print(f"{'-'*45}")
        print(f"{'AVERAGE':<15} {avg_ours:>10.3f} {avg_nv:>10.2f} {avg_ours - avg_nv:>+10.3f}")

    extra_benchmarks = {
        "gsm8k": (load_gsm8k, 500, 512),
        "humaneval": (load_humaneval, 164, 1024),
        "math500": (load_math500, 500, 2048),
        "aime": (load_aime, 30, 4096),
    }

    run_extra = [b for b in args.benchmarks if b in extra_benchmarks]
    if "all" in args.benchmarks:
        run_extra = list(extra_benchmarks.keys())

    for name in run_extra:
        loader, n, max_tok = extra_benchmarks[name]
        try:
            prompts = loader(n)
            result = run_benchmark(
                args.base_url, model_name, name, prompts,
                max_tokens=max_tok, batch_size=args.batch_size,
                spec_length=args.spec_length,
            )
            all_results[name] = result
        except Exception as e:
            print(f"ERROR running {name}: {e}")
            import traceback
            traceback.print_exc()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        args.output_dir,
        f"atom_eagle3_step{args.step}_{timestamp}.json",
    )
    with open(result_file, "w") as f:
        json.dump({
            "backend": "atom",
            "target_model": "/dev/shm/gpt-oss-120b",
            "draft_model": args.draft_model,
            "checkpoint_step": args.step,
            "spec_length": args.spec_length,
            "benchmarks": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {result_file}")

    print(f"\n{'='*70}")
    print(f"{'Benchmark':<25} {'n':>5} {'Accept Len':>12} {'Acc Rate':>10} {'Tok/s':>8}")
    print(f"{'='*70}")
    for name, r in all_results.items():
        print(f"{name:<25} {r['num_samples']:>5} {r['accept_length']:>12.3f} {r['acceptance_rate']:>9.1f}% {r['throughput_tps']:>8.1f}")
    print(f"{'='*70}")

    if all_results:
        avg_accept = sum(r["accept_length"] for r in all_results.values()) / len(all_results)
        avg_rate = sum(r["acceptance_rate"] for r in all_results.values()) / len(all_results)
        print(f"{'AVERAGE':<25} {'':>5} {avg_accept:>12.3f} {avg_rate:>9.1f}%")


if __name__ == "__main__":
    main()
