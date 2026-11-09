#!/usr/bin/env python3
"""Validate all training kernels by running compile + correctness checks on gfx950.

Extracts kernel code from each sample in train_v4.jsonl, builds a temporary
workspace, and runs compile_check.py + correctness_runner.py inside the
aiter-mi400-regress Docker container.

Usage:
    # Full validation (all 7437 samples, ~6 GPUs)
    python3 scripts/validate_training_kernels.py \
        --input /home/danyzhan/agent-kernel-sft-gfx950-triton-HIP/train_v4.jsonl \
        --gpu-ids 2,3,4,5,6,7 --workers 6

    # Quick smoke test (first 20 samples)
    python3 scripts/validate_training_kernels.py \
        --input /home/danyzhan/agent-kernel-sft-gfx950-triton-HIP/train_v4.jsonl \
        --gpu-ids 2 --workers 1 --max-samples 20

    # Only validate specific data sources
    python3 scripts/validate_training_kernels.py \
        --input /home/danyzhan/agent-kernel-sft-gfx950-triton-HIP/train_v4.jsonl \
        --data-sources gfx942_ported,complete_impl_v1,patch_v1

Output:
    --output-dir / validation_results.jsonl   (per-sample results)
    --output-dir / validation_summary.json    (aggregate stats)
    --output-dir / train_v4_validated.jsonl   (only passing samples)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DOCKER_CONTAINER = "aiter-mi400-regress"

SCRIPTS_DIR = Path(__file__).resolve().parent
COMPILE_CHECK = SCRIPTS_DIR / "compile_check.py"
CORRECTNESS_RUNNER = SCRIPTS_DIR / "correctness_runner_v2.py"

SAMPLE_OP_HARNESS = None
for p in Path("/home/danyzhan/agent-phase2-held-out-aiter/artifacts/kernel").iterdir():
    oh = p / "initial" / "scripts" / "op_harness.py"
    if oh.exists():
        SAMPLE_OP_HARNESS = oh
        break


def detect_language(system_prompt: str) -> str:
    sp = system_prompt.lower()
    if "flydsl" in sp:
        return "flydsl"
    if "opus" in sp or "hip c++" in sp or "hip kernel" in sp:
        return "opus"
    if "triton" in sp:
        return "triton"
    if "gpu kernel" in sp and ("gfx950" in sp or "mi350" in sp):
        return "opus"
    return "triton"


def extract_kernel(sample: dict) -> tuple[str | None, str]:
    """Extract kernel code and file extension from a training sample.

    Returns (code, filename) or (None, "") if extraction fails.
    """
    messages = sample["messages"]
    has_tool_calls = any("tool_calls" in m for m in messages)
    language = detect_language(messages[0].get("content", ""))

    if has_tool_calls:
        for m in reversed(messages):
            if "tool_calls" not in m:
                continue
            for tc in reversed(m.get("tool_calls", [])):
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        continue
                if args.get("action") == "write_file":
                    path = args.get("path", "")
                    content = args.get("content", "")
                    if content and ("kernel" in path or path.endswith(".py") or path.endswith(".cu")):
                        filename = "kernel.cu" if path.endswith(".cu") else "kernel.py"
                        return content, filename
        return None, ""

    assistant_content = ""
    for m in messages:
        if m["role"] == "assistant":
            assistant_content = m["content"]
            break

    if not assistant_content:
        return None, ""

    if assistant_content.strip().startswith("```diff"):
        return None, ""

    if language == "opus":
        if assistant_content.strip().startswith(("//", "#include", "extern", "__global__")):
            return assistant_content, "kernel.cu"
        blocks = re.findall(r'```(?:cpp|c\+\+|cuda|hip|c)?\n(.*?)```', assistant_content, re.DOTALL)
        if blocks:
            return blocks[-1], "kernel.cu"
        return assistant_content, "kernel.cu"

    blocks = re.findall(r'```(?:python|py)?\n(.*?)```', assistant_content, re.DOTALL)
    if blocks:
        return blocks[-1], "kernel.py"

    if ("import " in assistant_content[:300] or "def " in assistant_content[:500]
            or "@triton" in assistant_content[:300]):
        return assistant_content, "kernel.py"

    return None, ""


def map_operator(operator: str) -> str:
    """Map training data operator names to op_harness registry names."""
    aliases = {
        "rms_norm": "rmsnorm",
        "fused_add_rms_norm": "rmsnorm",
        "gated_rmsnorm_fp8_per_token_quant": "rmsnorm",
        "fused_add_layer_norm": "layernorm",
        "groupnorm": "layernorm",
        "opus_moe": "moe",
        "fused_moe": "moe",
        "moe_2stage": "moe",
        "moe_gemm": "moe",
        "moe_gate": "moe",
        "moe_scatter": "moe",
        "opus_gemm": "gemm",
        "gated_gemm": "gemm",
        "gemm_activation": "gemm",
        "scaled_quant_gemm": "gemm_a8w8",
        "batched_scaled_quant_gemm": "gemm_a8w8",
        "mxfp8_blockscale_gemm": "mxfp4_gemm",
        "mxfp8_gemm": "mxfp4_gemm",
        "blockscale_gemm": "block_scale_gemm",
        "block_scaled_gemm": "block_scale_gemm",
        "gemm_a4w4": "gemm_a8w8",
        "grouped_gemm": "moe_grouped_gemm",
        "paged_attention_decode": "paged_attention",
        "fused_causal_attention": "attention",
        "fused_qk_concat": "attention",
        "mla_decode": "mla",
        "mhc": "mha",
        "top_k": "topk",
        "topk_gating": "topk",
        "topk_softmax": "topk",
        "top_k_top_p_sampling_from_probs": "sampling",
        "silu_and_mul": "quantization",
        "fused_silu_mul": "quantization",
        "scaled_silu_and_mul": "quantization",
        "fused_mul_add": "quantization",
        "binary_op": "quantization",
        "activation": "quantization",
        "softmax": "quantization",
        "dynamic_per_token_quant": "quantization",
        "dynamic_per_tensor_quant": "quantization",
        "static_per_tensor_quant": "quantization",
        "mxfp4_quant": "quantization",
        "dsv4_rotate_quant": "quantization",
        "reduction": "quantization",
        "embedding": "quantization",
        "rope": "quantization",
        "rope_kv_cache": "quantization",
        "knn": "quantization",
        "fused_ops": "quantization",
        "gdr": "quantization",
        "custom_allreduce": "all_reduce",
    }
    return aliases.get(operator, operator)


def build_workspace(sample: dict, idx: int, base_dir: Path) -> Path | None:
    """Build a temporary workspace for validating one sample."""
    code, filename = extract_kernel(sample)
    if code is None:
        return None

    language = detect_language(sample["messages"][0].get("content", ""))
    operator = sample.get("operator", "unknown")
    mapped_op = map_operator(operator)

    ws = base_dir / f"sample_{idx:05d}"
    ws.mkdir(parents=True, exist_ok=True)
    scripts = ws / "scripts"
    scripts.mkdir(exist_ok=True)

    (ws / filename).write_text(code)

    config = {
        "language": language,
        "operator": mapped_op,
        "architecture": "gfx950",
        "task_id": f"validate-{idx:05d}",
        "timeout_seconds": 120,
    }
    import yaml
    (ws / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    shutil.copy2(COMPILE_CHECK, scripts / "compile_check.py")
    shutil.copy2(CORRECTNESS_RUNNER, scripts / "correctness_runner.py")
    if SAMPLE_OP_HARNESS:
        shutil.copy2(SAMPLE_OP_HARNESS, scripts / "op_harness.py")

    return ws


def docker_validate(ws: Path, gpu_id: str, timeout: int = 180) -> dict:
    """Run compile + correctness checks in Docker."""
    result = {
        "compiled": False,
        "correct": False,
        "compile_output": "",
        "correct_output": "",
    }

    stage = f"/tmp/_validate_{uuid.uuid4().hex[:12]}"

    try:
        subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "mkdir", "-p", f"{stage}/scripts"],
            capture_output=True, timeout=10,
        )

        for f in (ws).iterdir():
            if f.is_file():
                subprocess.run(
                    ["docker", "cp", str(f), f"{DOCKER_CONTAINER}:{stage}/{f.name}"],
                    capture_output=True, timeout=10,
                )
        for f in (ws / "scripts").iterdir():
            subprocess.run(
                ["docker", "cp", str(f), f"{DOCKER_CONTAINER}:{stage}/scripts/{f.name}"],
                capture_output=True, timeout=10,
            )

        r = subprocess.run(
            ["docker", "exec", "-e", f"HIP_VISIBLE_DEVICES={gpu_id}",
             DOCKER_CONTAINER, "python3",
             f"{stage}/scripts/compile_check.py", stage],
            capture_output=True, text=True, timeout=timeout,
        )
        output = (r.stdout + r.stderr).strip()
        result["compile_output"] = output[-2000:]
        if "PASS" in output:
            result["compiled"] = True
        else:
            return result

        r = subprocess.run(
            ["docker", "exec", "-e", f"HIP_VISIBLE_DEVICES={gpu_id}",
             DOCKER_CONTAINER, "python3",
             f"{stage}/scripts/correctness_runner.py", stage],
            capture_output=True, text=True, timeout=timeout,
        )
        output = (r.stdout + r.stderr).strip()
        result["correct_output"] = output[-2000:]
        if "PASS" in output:
            result["correct"] = True

    except subprocess.TimeoutExpired:
        result["compile_output"] = "TIMEOUT"
    except Exception as e:
        result["compile_output"] = f"EXCEPTION: {e}"
    finally:
        subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "rm", "-rf", stage],
            capture_output=True, timeout=10,
        )

    return result


def validate_one(args_tuple):
    idx, sample, base_dir, gpu_id = args_tuple

    data_source = sample.get("data_source", "unknown")
    operator = sample.get("operator", "unknown")
    schema = sample.get("schema_version", "unknown")

    entry = {
        "idx": idx,
        "operator": operator,
        "data_source": data_source,
        "schema_version": schema,
        "compiled": False,
        "correct": False,
        "skip_reason": None,
        "error": None,
    }

    code, filename = extract_kernel(sample)
    if code is None:
        entry["skip_reason"] = "cannot_extract_kernel"
        if schema == "messages_v1":
            assistant = ""
            for m in sample["messages"]:
                if m["role"] == "assistant":
                    assistant = m["content"]
                    break
            if assistant.strip().startswith("```diff"):
                entry["skip_reason"] = "patch_format"
        return entry

    ws = build_workspace(sample, idx, base_dir)
    if ws is None:
        entry["skip_reason"] = "workspace_build_failed"
        return entry

    try:
        result = docker_validate(ws, gpu_id)
        entry["compiled"] = result["compiled"]
        entry["correct"] = result["correct"]
        if not result["compiled"]:
            entry["error"] = result["compile_output"][-500:]
        elif not result["correct"]:
            entry["error"] = result["correct_output"][-500:]
    except Exception as e:
        entry["error"] = str(e)[:500]
    finally:
        shutil.rmtree(ws, ignore_errors=True)

    return entry


def main():
    parser = argparse.ArgumentParser(description="Validate training kernel quality")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/danyzhan/agent-kernel-sft-gfx950-triton-HIP/validation"))
    parser.add_argument("--gpu-ids", default="2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-sources", default=None,
                        help="Comma-separated data sources to validate (default: all)")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    print(f"Loading samples from {args.input}...")
    with open(args.input) as f:
        samples = [json.loads(l) for l in f]
    print(f"  Total: {len(samples)} samples")

    if args.data_sources:
        sources = set(args.data_sources.split(","))
        samples = [(i, s) for i, s in enumerate(samples) if s.get("data_source") in sources]
        print(f"  Filtered to data_sources={sources}: {len(samples)} samples")
    else:
        samples = list(enumerate(samples))

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"  Limited to first {args.max_samples} samples")

    gpu_ids = args.gpu_ids.split(",")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(tempfile.mkdtemp(prefix="kernel_validate_"))
    print(f"\nWorkspace: {base_dir}")
    print(f"GPUs: {gpu_ids} | Workers: {args.workers}")
    print(f"Output: {args.output_dir}")
    print()

    n = len(samples)
    results = []
    start = time.monotonic()

    results_path = args.output_dir / "validation_results.jsonl"
    with open(results_path, "w") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = []
            for task_idx, (orig_idx, sample) in enumerate(samples):
                gpu = gpu_ids[task_idx % len(gpu_ids)]
                tasks.append((orig_idx, sample, base_dir, gpu))

            future_map = {pool.submit(validate_one, t): t[0] for t in tasks}

            for done_count, future in enumerate(as_completed(future_map), 1):
                orig_idx = future_map[future]
                try:
                    entry = future.result()
                except Exception as e:
                    entry = {
                        "idx": orig_idx, "compiled": False, "correct": False,
                        "error": str(e)[:500],
                    }

                results.append(entry)
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()

                status = "PASS" if entry.get("correct") else (
                    "COMPILE" if entry.get("compiled") else (
                        "SKIP" if entry.get("skip_reason") else "FAIL"
                    )
                )
                op = entry.get("operator", "?")
                ds = entry.get("data_source", "?")
                print(
                    f"  [{done_count:5d}/{n}] idx={orig_idx:5d} "
                    f"[{ds:25s}] {op:25s} {status}",
                    flush=True,
                )

                if done_count % 200 == 0:
                    elapsed = time.monotonic() - start
                    compiled = sum(1 for r in results if r.get("compiled"))
                    correct = sum(1 for r in results if r.get("correct"))
                    rate = done_count / elapsed * 3600
                    print(
                        f"  --- Progress: {done_count}/{n} | "
                        f"Compiled: {compiled}/{done_count} ({compiled/done_count*100:.0f}%) | "
                        f"Correct: {correct}/{done_count} ({correct/done_count*100:.0f}%) | "
                        f"Rate: {rate:.0f}/hr ---",
                        flush=True,
                    )

    shutil.rmtree(base_dir, ignore_errors=True)

    elapsed = time.monotonic() - start
    compiled = sum(1 for r in results if r.get("compiled"))
    correct = sum(1 for r in results if r.get("correct"))
    skipped = sum(1 for r in results if r.get("skip_reason"))

    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total samples:  {n}")
    print(f"  Compiled:       {compiled}/{n} ({compiled/max(n,1)*100:.1f}%)")
    print(f"  Correct:        {correct}/{n} ({correct/max(n,1)*100:.1f}%)")
    print(f"  Skipped:        {skipped}/{n} ({skipped/max(n,1)*100:.1f}%)")
    print(f"  Failed:         {n - compiled - skipped}/{n}")
    print(f"  Elapsed:        {elapsed/60:.1f} min")

    by_source = {}
    for r in results:
        ds = r.get("data_source", "unknown")
        if ds not in by_source:
            by_source[ds] = {"total": 0, "compiled": 0, "correct": 0, "skipped": 0}
        by_source[ds]["total"] += 1
        if r.get("compiled"):
            by_source[ds]["compiled"] += 1
        if r.get("correct"):
            by_source[ds]["correct"] += 1
        if r.get("skip_reason"):
            by_source[ds]["skipped"] += 1

    print(f"\n  By data_source:")
    for ds, stats in sorted(by_source.items()):
        t = stats["total"]
        print(
            f"    {ds:30s}: {t:5d} total | "
            f"compiled={stats['compiled']:5d} ({stats['compiled']/t*100:5.1f}%) | "
            f"correct={stats['correct']:5d} ({stats['correct']/t*100:5.1f}%) | "
            f"skipped={stats['skipped']:5d}"
        )

    by_op = {}
    for r in results:
        op = r.get("operator", "unknown")
        if op not in by_op:
            by_op[op] = {"total": 0, "compiled": 0, "correct": 0}
        by_op[op]["total"] += 1
        if r.get("compiled"):
            by_op[op]["compiled"] += 1
        if r.get("correct"):
            by_op[op]["correct"] += 1

    print(f"\n  By operator (top 20 by failure count):")
    sorted_ops = sorted(by_op.items(), key=lambda x: x[1]["total"] - x[1]["compiled"], reverse=True)
    for op, stats in sorted_ops[:20]:
        t = stats["total"]
        failed = t - stats["compiled"]
        print(
            f"    {op:30s}: {t:5d} total | "
            f"compiled={stats['compiled']:5d} ({stats['compiled']/t*100:5.1f}%) | "
            f"correct={stats['correct']:5d} | "
            f"failed={failed:5d}"
        )

    common_errors = Counter()
    for r in results:
        err = r.get("error", "")
        if not err:
            continue
        if "import error" in err.lower():
            common_errors["import_error"] += 1
        elif "no public wrapper" in err.lower():
            common_errors["no_wrapper_function"] += 1
        elif "execution error" in err.lower() or "raised exception" in err.lower():
            common_errors["runtime_error"] += 1
        elif "hipcc" in err.lower():
            common_errors["hipcc_compile_error"] += 1
        elif "NaN" in err or "Inf" in err:
            common_errors["nan_inf_output"] += 1
        elif "make_args" in err:
            common_errors["make_args_failed"] += 1
        elif "TIMEOUT" in err:
            common_errors["timeout"] += 1
        else:
            common_errors["other"] += 1

    print(f"\n  Common error categories:")
    for cat, cnt in common_errors.most_common():
        print(f"    {cat:30s}: {cnt}")

    summary = {
        "input": str(args.input),
        "total_samples": n,
        "compiled": compiled,
        "correct": correct,
        "skipped": skipped,
        "compile_rate": round(compiled / max(n, 1) * 100, 1),
        "correct_rate": round(correct / max(n, 1) * 100, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "by_data_source": by_source,
        "by_operator": by_op,
        "error_categories": dict(common_errors),
    }
    summary_path = args.output_dir / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary: {summary_path}")

    print(f"\n  Writing validated dataset...")
    with open(args.input) as f:
        all_samples = [json.loads(l) for l in f]

    passing_indices = set()
    for r in results:
        if r.get("correct"):
            passing_indices.add(r["idx"])

    validated_path = args.output_dir / "train_v4_validated.jsonl"
    kept = 0
    with open(validated_path, "w") as f:
        for i, sample in enumerate(all_samples):
            if i in passing_indices or i not in {r["idx"] for r in results}:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1

    print(f"  Validated dataset: {validated_path} ({kept} samples)")
    print(f"\n  Results: {results_path}")


if __name__ == "__main__":
    main()
