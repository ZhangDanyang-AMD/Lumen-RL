#!/usr/bin/env python3
"""Re-validate all flydsl (and opus) samples via Docker.

Takes train_v6_clean.jsonl (already has validated triton), extracts flydsl/opus
kernel code, runs Docker compile+correctness, keeps only passing samples.

Usage:
    python3 scripts/revalidate_flydsl.py \
        --input train_v6_clean.jsonl \
        --output train_v6_verified.jsonl \
        --gpu-ids 0,1,2,3,4,5 --workers 6
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
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

AITER_INCLUDE_DIRS = [
    "/home/jimguo12/aiter/csrc/include",
    "/home/jimguo12/aiter/csrc/opus_gemm/include",
    "/home/jimguo12/aiter/csrc/opus_gemm/include/gfx950",
    "/home/jimguo12/aiter/csrc/opus_gemm/include/gfx942",
    "/home/jimguo12/aiter/csrc/opus_gemm/include/gfx942/a8w8",
    "/home/jimguo12/aiter/csrc/opus_gemm/include/gfx942/a16w16",
    "/home/jimguo12/aiter/csrc/opus_moe/include",
    "/home/jimguo12/aiter/csrc/opus_moe/include/gfx950",
    "/home/jimguo12/aiter/csrc/opus_moe/include/gfx950/a8w4",
    "/home/jimguo12/aiter/csrc/opus_moe/include/gfx950/a16w16",
    "/home/jimguo12/aiter/csrc/kernels/rope",
    "/home/jimguo12/aiter/csrc/cpp_itfs/pa",
    "/home/jimguo12/aiter/csrc/cpp_itfs",
    "/opt/venv/lib/python3.12/site-packages/torch/include",
    "/opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include",
]


def detect_language(msgs: list[dict]) -> str:
    sys_msg = msgs[0].get("content", "").lower() if msgs else ""
    if "flydsl" in sys_msg:
        return "flydsl"
    if "opus" in sys_msg or "hip c++" in sys_msg or "hip kernel" in sys_msg:
        return "opus"
    return "triton"


def extract_kernel_from_sample(sample: dict, language: str) -> str | None:
    msgs = sample.get("messages", [])
    kernel = None
    ext = ".cu" if language == "opus" else ".py"

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
            if language == "opus":
                blocks = re.findall(r'```(?:cpp|c\+\+|cuda|hip)?\n(.*?)```', content, re.DOTALL)
                for b in blocks:
                    if "__global__" in b or "__device__" in b or "#include" in b:
                        kernel = b
                if kernel is None and ("__global__" in content or "__device__" in content or "#include" in content):
                    kernel = content
            else:
                blocks = re.findall(r'```(?:python|py)?\n(.*?)```', content, re.DOTALL)
                for b in blocks:
                    if "import " in b or "def " in b or "fx." in b or "FlyDSL" in b.upper():
                        kernel = b
    return kernel


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
    stage = f"/tmp/_rfv_{uuid.uuid4().hex[:12]}"

    try:
        subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "mkdir", "-p", f"{stage}/scripts"],
            capture_output=True, timeout=10,
        )

        tmp = Path(tempfile.mkdtemp(prefix="rfv_"))
        filename = "kernel.cu" if language == "opus" else "kernel.py"
        (tmp / filename).write_text(kernel_code)

        mapped_op = MAP_OP.get(operator, operator)
        config = {
            "language": language,
            "operator": mapped_op,
            "architecture": "gfx950",
            "task_id": f"flydsl-rv-{uuid.uuid4().hex[:8]}",
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
    idx, sample, gpu_id, language = args_tuple
    operator = sample.get("operator", "unknown")

    kernel = extract_kernel_from_sample(sample, language)
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
    flydsl_samples = []
    opus_samples = []

    for i, s in enumerate(all_samples):
        lang = detect_language(s.get("messages", []))
        if lang == "flydsl":
            flydsl_samples.append((i, s))
        elif lang == "opus":
            opus_samples.append((i, s))
        else:
            triton_samples.append(s)

    print(f"Triton (already validated, pass-through): {len(triton_samples)}")
    print(f"FlyDSL to validate: {len(flydsl_samples)}")
    print(f"Opus to validate: {len(opus_samples)}")

    to_validate = [(i, s, "flydsl") for i, s in flydsl_samples] + \
                  [(i, s, "opus") for i, s in opus_samples]

    gpu_ids = args.gpu_ids.split(",")
    results = []
    passed_non_triton = []
    start = time.monotonic()
    n = len(to_validate)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [(idx, sample, gpu_ids[i % len(gpu_ids)], lang)
                 for i, (idx, sample, lang) in enumerate(to_validate)]
        future_map = {pool.submit(validate_one, t): t[0] for t in tasks}

        for done, future in enumerate(as_completed(future_map), 1):
            idx, sample, status = future.result()
            results.append((idx, status))

            if sample is not None:
                passed_non_triton.append(sample)

            if done % 50 == 0 or done == n:
                elapsed = time.monotonic() - start
                pass_count = sum(1 for _, s in results if s == "pass")
                print(f"  [{done:5d}/{n}] pass={pass_count} "
                      f"({elapsed/60:.1f}min)", flush=True)

    elapsed = time.monotonic() - start
    status_counts = Counter(s.split(":")[0] for _, s in results)

    lang_pass = Counter()
    for s in passed_non_triton:
        lang = detect_language(s.get("messages", []))
        lang_pass[lang] += 1

    print(f"\nFlyDSL/Opus validation done in {elapsed/60:.1f} min")
    print(f"Status breakdown: {dict(status_counts)}")
    print(f"Passed by language: {dict(lang_pass)}")

    final = triton_samples + passed_non_triton
    print(f"\nFinal dataset: {len(final)} samples")
    print(f"  Triton (pass-through): {len(triton_samples)}")
    print(f"  FlyDSL+Opus (validated pass): {len(passed_non_triton)}")

    lang_final = Counter()
    for s in final:
        lang = detect_language(s.get("messages", []))
        lang_final[lang] += 1
    print(f"  By language: {dict(lang_final)}")

    with open(args.output, "w") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(final)} samples to {args.output}")


if __name__ == "__main__":
    main()
