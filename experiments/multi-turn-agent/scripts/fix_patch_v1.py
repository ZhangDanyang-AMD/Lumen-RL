#!/usr/bin/env python3
"""Fix patch_v1 training samples by applying diffs to base kernels.

Extracts base kernel from user message code block, applies the diff patch,
validates the result via Docker, and outputs passing samples.

Usage:
    python3 scripts/fix_patch_v1.py \
        --input /home/danyzhan/agent-kernel-sft-gfx950-triton-HIP/train_v4.jsonl \
        --output /home/danyzhan/agent-kernel-sft-gfx950-triton-HIP/patch_v1_fixed.jsonl \
        --gpu-ids 0,1,2,3,4,5 --workers 6
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
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


def extract_base_kernel(user_msg: str) -> str | None:
    blocks = re.findall(r'```(?:python|py)?\n(.*?)```', user_msg, re.DOTALL)
    if blocks:
        return blocks[0]
    return None


def extract_diff(assistant_msg: str) -> str | None:
    blocks = re.findall(r'```(?:diff)?\n(.*?)```', assistant_msg, re.DOTALL)
    if blocks:
        return blocks[0]
    if assistant_msg.strip().startswith(("---", "diff ")):
        return assistant_msg.strip()
    return None


def apply_unified_diff(base: str, diff_text: str) -> str | None:
    """Apply a unified diff to base text. Returns patched text or None on failure."""
    tmp = Path(tempfile.mkdtemp(prefix="patch_"))
    try:
        (tmp / "kernel.py").write_text(base)
        (tmp / "patch.diff").write_text(diff_text)
        r = subprocess.run(
            ["patch", "-p1", "--no-backup-if-mismatch", "-i", str(tmp / "patch.diff"),
             str(tmp / "kernel.py")],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return (tmp / "kernel.py").read_text()
        r2 = subprocess.run(
            ["patch", "-p0", "--no-backup-if-mismatch", "-i", str(tmp / "patch.diff"),
             str(tmp / "kernel.py")],
            capture_output=True, text=True, timeout=10,
        )
        if r2.returncode == 0:
            return (tmp / "kernel.py").read_text()
        return None
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def detect_language(system_prompt: str) -> str:
    sp = system_prompt.lower()
    if "flydsl" in sp:
        return "flydsl"
    if "opus" in sp or "hip c++" in sp or "hip kernel" in sp:
        return "opus"
    return "triton"


def map_operator(operator: str) -> str:
    aliases = {
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
    return aliases.get(operator, operator)


def docker_validate(kernel_code: str, language: str, operator: str, gpu_id: str) -> dict:
    result = {"compiled": False, "correct": False, "error": ""}
    stage = f"/tmp/_patchfix_{uuid.uuid4().hex[:12]}"

    try:
        subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "mkdir", "-p", f"{stage}/scripts"],
            capture_output=True, timeout=10,
        )

        tmp = Path(tempfile.mkdtemp(prefix="pf_"))
        filename = "kernel.cu" if language == "opus" else "kernel.py"
        (tmp / filename).write_text(kernel_code)

        import yaml
        config = {
            "language": language,
            "operator": map_operator(operator),
            "architecture": "gfx950",
            "task_id": f"patchfix-{uuid.uuid4().hex[:8]}",
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


def process_one(args_tuple):
    idx, sample, gpu_id = args_tuple
    msgs = sample["messages"]
    operator = sample.get("operator", "unknown")
    language = detect_language(msgs[0].get("content", ""))

    user_msg = ""
    assistant_msg = ""
    for m in msgs:
        if m["role"] == "user":
            user_msg = m["content"]
        elif m["role"] == "assistant":
            assistant_msg = m["content"]

    base = extract_base_kernel(user_msg)
    diff = extract_diff(assistant_msg)

    if base is None:
        return idx, None, "no_base_kernel"
    if diff is None:
        return idx, None, "no_diff"

    patched = apply_unified_diff(base, diff)
    if patched is None:
        return idx, None, "patch_failed"

    result = docker_validate(patched, language, operator, gpu_id)
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

    patches = [(i, s) for i, s in enumerate(all_samples) if s.get("data_source") == "patch_v1"]
    print(f"Found {len(patches)} patch_v1 samples")

    gpu_ids = args.gpu_ids.split(",")
    n = len(patches)
    results = []
    start = time.monotonic()

    with open(args.output, "w") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = [(idx, sample, gpu_ids[i % len(gpu_ids)]) for i, (idx, sample) in enumerate(patches)]
            future_map = {pool.submit(process_one, t): t[0] for t in tasks}

            for done, future in enumerate(as_completed(future_map), 1):
                idx, sample, status = future.result()
                results.append((idx, status))

                if sample is not None:
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    out_f.flush()

                print(f"  [{done:4d}/{n}] idx={idx:5d} {status}", flush=True)

    elapsed = time.monotonic() - start
    status_counts = Counter(s for _, s in results)
    passed = sum(1 for _, s in results if s == "pass")
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Passed: {passed}/{n}")
    print(f"Status: {dict(status_counts)}")


if __name__ == "__main__":
    main()
