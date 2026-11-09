#!/usr/bin/env python3
"""Clean train_v6: remove correct=False samples, then re-validate all triton via Docker.

Steps:
1. Parse all samples, remove any with correct=False in eval tool response
2. Fix eval tool response for samples that had correct=False (set ok=False)
3. For multi-turn samples: extract kernel from write_file tool call
4. For single-turn samples: extract kernel from code block
5. Run Docker compile+correctness check on each kernel
6. Write only passing samples to output

Usage:
    python3 scripts/clean_and_revalidate.py \
        --input train_v6.jsonl \
        --output train_v6_clean.jsonl \
        --gpu-ids 0,1,2,3,4,5 --workers 6
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import shutil
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


def detect_language(msgs: list[dict]) -> str:
    sys_msg = msgs[0].get("content", "").lower() if msgs else ""
    if "flydsl" in sys_msg:
        return "flydsl"
    if "opus" in sys_msg or "hip c++" in sys_msg or "hip kernel" in sys_msg:
        return "opus"
    return "triton"


def extract_kernel_from_sample(sample: dict) -> str | None:
    msgs = sample.get("messages", [])
    kernel = None
    for m in msgs:
        if m.get("role") == "assistant" and "tool_calls" in m:
            for tc in m["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if args.get("action") == "write_file" and "kernel" in args.get("path", ""):
                        kernel = args["content"]
                except (json.JSONDecodeError, KeyError):
                    pass

    if kernel:
        return kernel

    for m in msgs:
        if m.get("role") == "assistant":
            content = m.get("content", "")
            blocks = re.findall(r'```(?:python|py)?\n(.*?)```', content, re.DOTALL)
            for b in blocks:
                if "import " in b or "def " in b or "@triton" in b or "tl." in b:
                    kernel = b
    return kernel


def check_eval_correct(sample: dict) -> bool | None:
    for m in sample.get("messages", []):
        if m.get("role") == "tool":
            try:
                content = json.loads(m["content"])
                if "evaluation" in content:
                    return content["evaluation"].get("correct", None)
            except (json.JSONDecodeError, KeyError):
                pass
    return None


MAP_OP = {
    "rms_norm": "rmsnorm", "fused_add_rms_norm": "rmsnorm",
    "opus_moe": "moe", "fused_moe": "moe",
    "opus_gemm": "gemm", "gated_gemm": "gemm", "gemm_activation": "gemm",
    "scaled_quant_gemm": "gemm_a8w8",
    "fused_silu_mul": "quantization", "silu_and_mul": "quantization",
    "binary_op": "quantization", "activation": "quantization",
    "softmax": "quantization", "rope": "quantization",
    "rope_kv_cache": "quantization", "knn": "quantization",
    "fused_mul_add": "quantization", "sampling": "quantization",
    "dynamic_per_token_quant": "quantization",
    "dynamic_per_tensor_quant": "quantization",
    "static_per_tensor_quant": "quantization",
}


def docker_validate(kernel_code: str, language: str, operator: str, gpu_id: str) -> dict:
    import yaml
    result = {"compiled": False, "correct": False, "error": ""}
    stage = f"/tmp/_rv_{uuid.uuid4().hex[:12]}"

    try:
        subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "mkdir", "-p", f"{stage}/scripts"],
            capture_output=True, timeout=10,
        )

        tmp = Path(tempfile.mkdtemp(prefix="rv_"))
        filename = "kernel.cu" if language == "opus" else "kernel.py"
        (tmp / filename).write_text(kernel_code)

        mapped_op = MAP_OP.get(operator, operator)
        config = {
            "language": language,
            "operator": mapped_op,
            "architecture": "gfx950",
            "task_id": f"revalidate-{uuid.uuid4().hex[:8]}",
            "timeout_seconds": 120,
        }
        (tmp / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

        for f in [filename, "config.yaml"]:
            subprocess.run(
                ["docker", "cp", str(tmp / f), f"{DOCKER_CONTAINER}:{stage}/{f}"],
                capture_output=True, timeout=10,
            )
        for script in [COMPILE_CHECK, CORRECTNESS_RUNNER, SAMPLE_OP_HARNESS]:
            if script and script.exists():
                dst_name = "compile_check.py" if "compile" in script.name else (
                    "correctness_runner.py" if "correct" in script.name else script.name
                )
                subprocess.run(
                    ["docker", "cp", str(script), f"{DOCKER_CONTAINER}:{stage}/scripts/{dst_name}"],
                    capture_output=True, timeout=10,
                )
        shutil.rmtree(tmp, ignore_errors=True)

        for step in ["compile_check.py", "correctness_runner.py"]:
            r = subprocess.run(
                ["docker", "exec", "-e", f"HIP_VISIBLE_DEVICES={gpu_id}",
                 DOCKER_CONTAINER, "python3", f"{stage}/scripts/{step}", stage],
                capture_output=True, text=True, timeout=180,
            )
            output = (r.stdout + r.stderr).strip()
            if "PASS" in output:
                if "compile" in step:
                    result["compiled"] = True
                else:
                    result["correct"] = True
            else:
                result["error"] = output[-500:]
                break

    except subprocess.TimeoutExpired:
        result["error"] = "TIMEOUT"
    except Exception as e:
        result["error"] = str(e)[:500]
    finally:
        subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "rm", "-rf", stage],
            capture_output=True, timeout=10,
        )

    return result


def validate_one(args_tuple):
    idx, sample, gpu_id = args_tuple
    msgs = sample.get("messages", [])
    language = detect_language(msgs)
    operator = sample.get("operator", "unknown")

    if language != "triton":
        return idx, sample, "non_triton_pass"

    kernel = extract_kernel_from_sample(sample)
    if kernel is None:
        return idx, None, "no_kernel_extracted"

    result = docker_validate(kernel, language, operator, gpu_id)
    if result["correct"]:
        return idx, sample, "pass"
    elif result["compiled"]:
        return idx, None, f"correctness_fail: {result['error'][:100]}"
    else:
        return idx, None, f"compile_fail: {result['error'][:100]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    with open(args.input) as f:
        all_samples = [json.loads(l) for l in f]

    print(f"Total input samples: {len(all_samples)}")

    triton_samples = []
    non_triton_samples = []
    for i, s in enumerate(all_samples):
        lang = detect_language(s.get("messages", []))
        if lang == "triton":
            triton_samples.append((i, s))
        else:
            non_triton_samples.append(s)

    print(f"Triton samples to validate: {len(triton_samples)}")
    print(f"Non-triton (pass-through): {len(non_triton_samples)}")

    eval_status = Counter()
    for _, s in triton_samples:
        ec = check_eval_correct(s)
        if ec is True:
            eval_status["eval_correct_true"] += 1
        elif ec is False:
            eval_status["eval_correct_false"] += 1
        else:
            eval_status["no_eval_field"] += 1
    print(f"Eval field status: {dict(eval_status)}")

    gpu_ids = args.gpu_ids.split(",")
    results = []
    passed_triton = []
    start = time.monotonic()
    n = len(triton_samples)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [(idx, sample, gpu_ids[i % len(gpu_ids)])
                 for i, (idx, sample) in enumerate(triton_samples)]
        future_map = {pool.submit(validate_one, t): t[0] for t in tasks}

        for done, future in enumerate(as_completed(future_map), 1):
            idx, sample, status = future.result()
            results.append((idx, status))

            if sample is not None and "triton" in status or status == "pass":
                passed_triton.append(sample)

            if done % 50 == 0 or done == n:
                elapsed = time.monotonic() - start
                pass_count = sum(1 for _, s in results if s == "pass")
                print(f"  [{done:5d}/{n}] pass={pass_count} "
                      f"({elapsed/60:.1f}min)", flush=True)

    elapsed = time.monotonic() - start
    status_counts = Counter(s.split(":")[0] for _, s in results)

    print(f"\nTriton validation done in {elapsed/60:.1f} min")
    print(f"Status breakdown: {dict(status_counts)}")

    final = non_triton_samples + passed_triton
    print(f"\nFinal dataset: {len(final)} samples")
    print(f"  Non-triton: {len(non_triton_samples)}")
    print(f"  Triton (validated pass): {len(passed_triton)}")

    with open(args.output, "w") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(final)} samples to {args.output}")


if __name__ == "__main__":
    main()
