"""GEAK kernel sandbox reward for Lumen-RL GRPO training.

Evaluates model-generated kernel patches/code on real AMD GPUs via
the GEAK task_runner pipeline (compile -> correctness -> performance).

Compatible with Lumen-RL's batch reward interface:
    reward_fn(batch_responses, batch_prompts, **kwargs) -> Tensor[B]

Delegates to sandbox.reward.sandbox_reward_batch when available,
falls back to local implementation otherwise.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

logger = logging.getLogger(__name__)

GEAK_ROOT = Path(os.environ.get("GEAK_ROOT", "/home/danyzhan/GEAK"))
HELD_OUT_ROOT = Path(os.environ.get("HELD_OUT_ROOT", "/home/danyzhan/held-out-benchmark"))
EVAL_GPU_IDS = [int(g) for g in os.environ.get("EVAL_GPU_IDS", "4,5,6,7").split(",")]
COMMAND_TIMEOUT = int(os.environ.get("COMMAND_TIMEOUT", "120"))
BASELINE_REPEATS = int(os.environ.get("BASELINE_REPEATS", "3"))


def _run_eval(workspace: Path, mode: str, gpu_id: int) -> dict:
    """Run compile/correctness/performance in a workspace."""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        proc = subprocess.run(
            ["python3", "scripts/task_runner.py", mode],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            env=env,
        )
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-1000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _extract_perf_ms(stdout: str) -> float:
    match = re.search(r"Perf:\s+([\d.]+)\s+ms", stdout)
    return float(match.group(1)) if match else 0.0


def _extract_code(response: str) -> str:
    """Extract code from model response (handles code blocks)."""
    blocks = re.findall(r"```(?:python|cpp|c\+\+|hip)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    if "```" in response:
        return response.split("```", 1)[1].split("```", 1)[0].strip()
    return response.strip()


def _fix_arch_gate(code: str) -> str:
    """Remove broken architecture gates."""
    code = re.sub(
        r"if\s+(?:not\s+)?.*(?:gcnArchName|get_device_arch).*?:\s*\n\s*raise\s+RuntimeError\(.*?\)\s*\n",
        "# Architecture gate removed (gfx942 verified)\n",
        code,
        flags=re.DOTALL,
    )
    code = re.sub(
        r"raise\s+NotImplementedError\(.*?(?:NVIDIA|CUDA|not AMD).*?\)\s*\n",
        "# NVIDIA gate removed\npass\n",
        code,
        flags=re.IGNORECASE,
    )
    code = code.replace("hip_cxxflags=", "extra_cflags=")
    return code


def _add_function_alias(code: str, expected_name: str) -> str:
    """Add function alias if expected name not found."""
    if not expected_name or re.search(rf"def\s+{expected_name}\s*\(", code):
        return code
    if re.search(rf"^{expected_name}\s*=", code, re.MULTILINE):
        return code
    defs = re.findall(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    public = [d for d in defs if not d.startswith("_")]
    if len(public) == 1:
        code += f"\n{expected_name} = {public[0]}\n"
    elif public:
        for d in public:
            if expected_name.replace("_", "") in d.replace("_", "").lower():
                code += f"\n{expected_name} = {d}\n"
                break
        else:
            code += f"\n{expected_name} = {public[-1]}\n"
    return code


def eval_single_kernel(
    response: str,
    task_id: str,
    task_dir: Path,
    baseline_ms: float,
    family: str,
    gpu_id: int,
) -> dict[str, Any]:
    """Evaluate a single kernel response on GPU.

    Returns dict with reward, stage, speedup, error info.
    """
    has_patch = "---" in response and "+++" in response
    code = _extract_code(response)
    has_code = len(code.strip()) > 50 and ("import" in code[:100] or "def " in code[:200])

    workdir = Path(tempfile.mkdtemp(prefix=f"rl_eval_{task_id[:15]}_"))
    try:
        shutil.copytree(task_dir, workdir / "w", dirs_exist_ok=True)
        ws = workdir / "w"

        if has_patch:
            # Try fuzzy patch apply
            try:
                from scripts.fuzzy_patch import apply_patch as fuzzy_apply

                original = (ws / "kernel.py").read_text()
                result = fuzzy_apply(original, _clean_patch(response))
                if result and result != original:
                    (ws / "kernel.py").write_text(result)
                else:
                    return {"reward": -1.0, "stage": "patch_apply", "error": "apply_failed"}
            except Exception:
                return {"reward": -1.0, "stage": "patch_apply", "error": "apply_exception"}
        elif has_code:
            code = _fix_arch_gate(code)
            code = _add_function_alias(code, family)
            (ws / "kernel.py").write_text(code)
        else:
            return {"reward": -1.0, "stage": "no_output", "error": "no_patch_or_code"}

        # Compile
        r = _run_eval(ws, "compile", gpu_id)
        if not r["ok"]:
            return {"reward": -1.0, "stage": "compile", "error": r.get("stderr", "")[-200:]}

        # Correctness
        r = _run_eval(ws, "correctness", gpu_id)
        if not r["ok"]:
            return {"reward": -0.5, "stage": "correctness", "error": r.get("stderr", "")[-200:]}

        # Performance
        r = _run_eval(ws, "performance", gpu_id)
        if not r["ok"]:
            return {"reward": 0.0, "stage": "performance", "error": "perf_failed"}

        perf_ms = _extract_perf_ms(r["stdout"])
        if perf_ms <= 0 or baseline_ms <= 0:
            return {"reward": 0.0, "stage": "performance", "error": "no_perf_data"}

        speedup = baseline_ms / perf_ms
        reward = 1.0 + min(max(math.log(speedup), 0), math.log(3.0))
        return {"reward": reward, "stage": "success", "speedup": speedup, "perf_ms": perf_ms}

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _clean_patch(raw: str) -> str:
    """Strip code fences, normalize git diff."""
    text = raw.strip()
    block = re.search(r"```(?:diff|patch|)\s*\n(.*?)```", text, re.DOTALL)
    if block:
        text = block.group(1).strip()
    else:
        start = re.search(r"^(diff --git|--- )", text, re.MULTILINE)
        if start:
            text = text[start.start():].strip()
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if line.startswith("diff --git") or line.startswith("index "):
            continue
        if line.startswith("--- a/"):
            cleaned.append("--- " + line[6:])
        elif line.startswith("+++ b/"):
            cleaned.append("+++ " + line[6:])
        else:
            cleaned.append(line)
    return "\n".join(cleaned) + "\n"


def kernel_reward_batch(
    responses: Sequence[str],
    ground_truths: Sequence[str],
    *,
    task_ids: Sequence[str] | None = None,
    task_dirs: Sequence[str | Path] | None = None,
    baseline_ms_list: Sequence[float] | None = None,
    families: Sequence[str] | None = None,
    **kwargs,
) -> torch.Tensor:
    """Batch kernel reward computation for Lumen-RL GRPO.

    Compatible with Lumen-RL's reward interface: fn(responses, ground_truths) -> Tensor.
    Task metadata is passed via ground_truths as JSON strings with fields:
      task_id, task_dir, baseline_ms, family

    Also supports explicit kwargs for direct invocation.

    Returns:
        Tensor of rewards, shape [batch_size]
    """
    batch_size = len(responses)
    rewards = torch.zeros(batch_size, dtype=torch.float32)

    if task_ids is None:
        task_ids_resolved = []
        task_dirs_resolved = []
        baseline_ms_resolved = []
        families_resolved = []
        for gt in ground_truths:
            try:
                meta = json.loads(gt)
                task_ids_resolved.append(meta.get("task_id", ""))
                task_dirs_resolved.append(meta.get("task_dir", ""))
                baseline_ms_resolved.append(float(meta.get("baseline_ms", 1.0)))
                families_resolved.append(meta.get("family", ""))
            except (json.JSONDecodeError, TypeError):
                task_ids_resolved.append("")
                task_dirs_resolved.append("")
                baseline_ms_resolved.append(1.0)
                families_resolved.append("")
        task_ids = task_ids_resolved
        task_dirs = task_dirs_resolved
        baseline_ms_list = baseline_ms_resolved
        families = families_resolved

    if not any(task_ids):
        logger.warning("No task metadata provided; returning zero rewards")
        return rewards

    with ThreadPoolExecutor(max_workers=len(EVAL_GPU_IDS)) as pool:
        futures = {}
        for i, (resp, tid) in enumerate(zip(responses, task_ids)):
            if not tid:
                continue
            gpu_id = EVAL_GPU_IDS[i % len(EVAL_GPU_IDS)]
            td = Path(task_dirs[i]) if task_dirs else HELD_OUT_ROOT / "artifacts" / "kernel" / tid / "initial"
            bl = baseline_ms_list[i] if baseline_ms_list else 1.0
            fam = families[i] if families else ""
            f = pool.submit(eval_single_kernel, resp, tid, td, bl, fam, gpu_id)
            futures[f] = i

        for f in as_completed(futures):
            idx = futures[f]
            try:
                result = f.result()
                rewards[idx] = result["reward"]
                if result.get("speedup"):
                    logger.info(
                        "task=%s reward=%.2f speedup=%.3f",
                        task_ids[idx][:30],
                        result["reward"],
                        result["speedup"],
                    )
            except Exception as e:
                logger.error("eval failed for task %s: %s", task_ids[idx][:30], e)
                rewards[idx] = -1.0

    return rewards
