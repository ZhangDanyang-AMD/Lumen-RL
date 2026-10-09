"""Generic batch reward function dispatching to any sandbox backend."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import torch

from .base import SandboxBackend
from .registry import get_backend

logger = logging.getLogger(__name__)

EVAL_GPU_IDS = [int(g) for g in os.environ.get("EVAL_GPU_IDS", "4,5,6,7").split(",")]


def _extract_code(response: str) -> str:
    blocks = re.findall(r"```(?:python|cpp|c\+\+|hip)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    if "```" in response:
        return response.split("```", 1)[1].split("```", 1)[0].strip()
    return response.strip()


def _fix_arch_gate(code: str) -> str:
    code = re.sub(
        r"if\s+(?:not\s+)?.*(?:gcnArchName|get_device_arch).*?:\s*\n\s*raise\s+RuntimeError\(.*?\)\s*\n",
        "# Architecture gate removed (gfx942 verified)\n",
        code, flags=re.DOTALL,
    )
    code = re.sub(
        r"raise\s+NotImplementedError\(.*?(?:NVIDIA|CUDA|not AMD).*?\)\s*\n",
        "# NVIDIA gate removed\npass\n",
        code, flags=re.IGNORECASE,
    )
    code = code.replace("hip_cxxflags=", "extra_cflags=")
    return code


def _add_function_alias(code: str, expected_name: str) -> str:
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


def _clean_patch(raw: str) -> str:
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


def _reconstruct_from_diff(response: str, original: str) -> str | None:
    """Extract the new file content from a unified diff by applying +/- lines."""
    patch_text = _clean_patch(response)
    lines = patch_text.split("\n")
    result_lines = original.split("\n")

    try:
        new_lines = []
        in_hunk = False
        for line in lines:
            if line.startswith("@@"):
                in_hunk = True
                continue
            if not in_hunk:
                continue
            if line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith("-"):
                pass
            elif line.startswith(" "):
                new_lines.append(line[1:])
            else:
                new_lines.append(line)

        if not new_lines:
            return None
        result = "\n".join(new_lines)
        try:
            compile(result, "<reconstructed>", "exec")
            return result
        except SyntaxError:
            return None
    except Exception:
        return None


def eval_single(
    backend: SandboxBackend,
    response: str,
    task_id: str,
    task_dir: Path,
    baseline_ms: float,
    family: str,
    gpu_id: int,
) -> dict[str, Any]:
    """Evaluate a single response using the given sandbox backend."""
    has_patch = "---" in response and "+++" in response
    code = _extract_code(response)
    has_code = len(code.strip()) > 50 and ("import" in code[:100] or "def " in code[:200])

    if has_patch:
        patch_ok = False
        try:
            import sys
            mta = Path(__file__).resolve().parents[2] / "multi-tune-agent" / "scripts"
            if str(mta) not in sys.path:
                sys.path.insert(0, str(mta))
            from fuzzy_patch import apply_patch as fuzzy_apply

            original = (task_dir / "kernel.py").read_text()
            result = fuzzy_apply(original, _clean_patch(response))
            if result and result != original:
                try:
                    compile(result, "<patch>", "exec")
                    kernel_source = result
                    patch_ok = True
                except SyntaxError:
                    pass
        except Exception:
            pass

        if not patch_ok:
            # Try GNU patch with fuzz factor as fallback
            try:
                import subprocess, tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
                    tf.write((task_dir / "kernel.py").read_text())
                    tf_path = tf.name
                with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as pf:
                    pf.write(_clean_patch(response))
                    pf_path = pf.name
                cp = subprocess.run(
                    ["patch", "--fuzz=3", "-s", tf_path, pf_path],
                    capture_output=True, text=True, timeout=10
                )
                if cp.returncode == 0:
                    patched = Path(tf_path).read_text()
                    try:
                        compile(patched, "<gnu_patch>", "exec")
                        kernel_source = patched
                        patch_ok = True
                    except SyntaxError:
                        pass
                Path(tf_path).unlink(missing_ok=True)
                Path(pf_path).unlink(missing_ok=True)
            except Exception:
                pass

        if not patch_ok:
            try:
                from sandbox.robust_patch import apply_robust_patch
                rp = apply_robust_patch(original, _clean_patch(response))
                if rp:
                    try:
                        compile(rp, "<robust_patch>", "exec")
                        kernel_source = rp
                        patch_ok = True
                    except SyntaxError:
                        pass
            except Exception:
                pass

        if not patch_ok:
            reconstructed = _reconstruct_from_diff(response, (task_dir / "kernel.py").read_text())
            if reconstructed:
                kernel_source = _fix_arch_gate(reconstructed)
                kernel_source = _add_function_alias(kernel_source, family)
            elif has_code:
                kernel_source = _fix_arch_gate(code)
                kernel_source = _add_function_alias(kernel_source, family)
            else:
                return {"reward": -1.0, "stage": "patch_apply", "error": "apply_failed_syntax"}
    elif has_code:
        kernel_source = _fix_arch_gate(code)
        kernel_source = _add_function_alias(kernel_source, family)
    else:
        return {"reward": -1.0, "stage": "no_output", "error": "no_patch_or_code"}

    eval_result = backend.evaluate(
        task_id, kernel_source, task_dir, gpu_id,
        baseline_ms=baseline_ms,
    )

    if not eval_result.compiled:
        return {"reward": -1.0, "stage": "compile", "error": eval_result.error}
    if not eval_result.correct:
        return {"reward": -0.5, "stage": "correctness", "error": eval_result.error}
    if eval_result.speedup <= 0:
        return {"reward": 0.0, "stage": "performance", "error": "no_perf_data"}

    reward = 1.0 + min(max(math.log(eval_result.speedup), 0), math.log(3.0))
    return {
        "reward": reward,
        "stage": "success",
        "speedup": eval_result.speedup,
        "perf_ms": eval_result.perf_ms,
    }


def sandbox_reward_batch(
    responses: Sequence[str],
    ground_truths: Sequence[str],
    *,
    backend_name: str = "geak",
    backend_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Batch reward computation using any registered sandbox backend.

    Task metadata is passed via ground_truths as JSON strings with fields:
      task_id, task_dir, baseline_ms, family
    """
    batch_size = len(responses)
    rewards = torch.zeros(batch_size, dtype=torch.float32)

    backend = get_backend(backend_name, **(backend_kwargs or {}))

    task_metas = []
    for gt in ground_truths:
        try:
            task_metas.append(json.loads(gt))
        except (json.JSONDecodeError, TypeError):
            task_metas.append({})

    if not any(m.get("task_id") for m in task_metas):
        logger.warning("No task metadata in ground_truths; returning zero rewards")
        return rewards

    with ThreadPoolExecutor(max_workers=len(EVAL_GPU_IDS)) as pool:
        futures = {}
        for i, (resp, meta) in enumerate(zip(responses, task_metas)):
            tid = meta.get("task_id", "")
            if not tid:
                continue
            gpu_id = EVAL_GPU_IDS[i % len(EVAL_GPU_IDS)]
            td = Path(meta.get("task_dir", ""))
            bl = float(meta.get("baseline_ms", 1.0))
            fam = meta.get("family", "")
            f = pool.submit(eval_single, backend, resp, tid, td, bl, fam, gpu_id)
            futures[f] = i

        for f in as_completed(futures):
            idx = futures[f]
            try:
                result = f.result()
                rewards[idx] = result["reward"]
                if result.get("speedup"):
                    logger.info(
                        "task=%s reward=%.2f speedup=%.3f",
                        task_metas[idx].get("task_id", "?")[:30],
                        result["reward"],
                        result["speedup"],
                    )
            except Exception as e:
                logger.error("eval failed: %s", e)
                rewards[idx] = -1.0

    return rewards
