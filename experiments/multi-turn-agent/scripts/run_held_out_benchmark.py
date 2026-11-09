"""Held-out benchmark: patch optimization + from-scratch generation.

Runs both benchmark modes against a model served via vLLM,
using the 99 passing held-out tasks (53 Triton + 46 HIP).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as http_requests


BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
HELD_OUT_ROOT = Path("/home/danyzhan/held-out-benchmark")
BASELINE_RESULTS = HELD_OUT_ROOT / "receipts" / "baseline_results.json"

# Backend selection: "vllm" (default) or "anthropic"
MODEL_BACKEND = os.environ.get("BENCHMARK_BACKEND", "vllm")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-20250514")


def load_passing_tasks() -> list[dict]:
    """Load tasks that passed baseline verification."""
    with open(BASELINE_RESULTS) as f:
        baselines = json.load(f)
    passing = {r["task"] for r in baselines if r["passed"]}

    tasks = []
    with open(HELD_OUT_ROOT / "tasks" / "kernel.jsonl") as f:
        for line in f:
            task = json.loads(line)
            if task["task_id"] in passing:
                tasks.append(task)
    return tasks


MAX_TURNS = 5  # max retry turns for agent loop


def _call_vllm(messages: list[dict], max_tokens: int, temperature: float) -> tuple[str, dict]:
    resp = http_requests.post(
        f"{BASE_URL}/chat/completions",
        json={"model": MODEL_NAME, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


def _call_anthropic(messages: list[dict], max_tokens: int, temperature: float) -> tuple[str, dict]:
    import anthropic
    client = anthropic.Anthropic()
    # Convert OpenAI format to Anthropic format
    system_msg = ""
    api_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            api_messages.append({"role": m["role"], "content": m["content"]})
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_msg,
        messages=api_messages,
    )
    content = resp.content[0].text
    return content, {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


def call_model(messages: list[dict], max_tokens: int = 8192, temperature: float = 0.2) -> tuple[str, dict]:
    """Call model and return (content, usage_dict)."""
    if MODEL_BACKEND == "anthropic":
        return _call_anthropic(messages, max_tokens, temperature)
    return _call_vllm(messages, max_tokens, temperature)


def build_patch_prompt(task: dict, kernel_source: str) -> list[dict]:
    """Build SFT-format prompt for patch optimization (cold_start)."""
    contract = task.get("contract", {})
    frozen_input = json.dumps({
        "input": {
            "contract": contract,
            "parent_source": {"kernel.py": kernel_source},
            "baseline": {"commands": {
                "compile": "python3 scripts/task_runner.py compile",
                "correctness": "python3 scripts/task_runner.py correctness",
                "performance": "python3 scripts/task_runner.py performance",
            }},
            "direction": None,
            "profile": None,
            "error_feedback": None,
        },
        "task_type": "cold_start",
    })
    return [
        {"role": "system", "content": "You are an expert software engineer. Return only the requested patch."},
        {"role": "user", "content": frozen_input},
    ]


def _extract_interface_spec(task: dict) -> str:
    """Extract the function interface from task_runner.py and initial_source.py."""
    task_id = task["task_id"]
    task_dir = HELD_OUT_ROOT / "artifacts" / "kernel" / task_id / "initial"
    runner_path = task_dir / "scripts" / "task_runner.py"
    source_path = task_dir / "kernel.py"

    import re
    spec_parts = []

    # Extract FAMILY and calling pattern from task_runner.py
    family = task.get("top10_family", "kernel_fn")
    make_case = ""
    call_pattern = ""
    if runner_path.exists():
        content = runner_path.read_text()
        family_match = re.search(r"FAMILY\s*=\s*['\"](\w+)['\"]", content)
        if family_match:
            family = family_match.group(1)

        lines = content.split("\n")
        in_fn = False
        for line in lines:
            if "def make_case" in line:
                in_fn = True
            if in_fn:
                make_case += line + "\n"
                if line.strip().startswith("return "):
                    break

        for line in lines:
            stripped = line.strip()
            if "kernel." in stripped and "import" not in stripped and "#" not in stripped:
                call_pattern = stripped
                break

    # Extract the EXACT function signature from initial_source.py
    func_signature = ""
    if source_path.exists():
        source = source_path.read_text()
        # Find the public function definition (the wrapper, not the @triton.jit kernel)
        func_defs = re.findall(r'^(def\s+\w+\s*\([^)]*\)(?:\s*->.*?)?:)', source, re.MULTILINE)
        for fd in func_defs:
            func_name = re.match(r'def\s+(\w+)', fd).group(1)
            if not func_name.startswith('_'):
                func_signature = fd
                break

    spec_parts.append(f"CRITICAL: Your module MUST export a function named exactly '{family}'.")
    spec_parts.append(f"")
    spec_parts.append(f"The evaluation harness calls your function as:")
    spec_parts.append(f"  from kernel import {family}")
    spec_parts.append(f"  args = make_case(...)  # returns a tuple")
    spec_parts.append(f"  result = kernel.{family}(*args)")
    if call_pattern:
        spec_parts.append(f"  Actual call: {call_pattern}")
    spec_parts.append(f"")

    if func_signature:
        spec_parts.append(f"REFERENCE function signature (your function MUST match this EXACT signature):")
        spec_parts.append(f"  {func_signature}")
        spec_parts.append(f"")

    if make_case:
        spec_parts.append(f"The harness creates test inputs with:")
        spec_parts.append(make_case)
        spec_parts.append(f"Your function receives make_case()'s return value unpacked as *args.")
        spec_parts.append(f"Match the number and order of positional arguments EXACTLY.")

    return "\n".join(spec_parts)


def build_generation_prompt(task: dict) -> list[dict]:
    """Build prompt for from-scratch kernel generation with interface spec."""
    contract = task.get("contract", {}).get("kernel_contract", {})
    lane = task.get("lane", "")
    language = "Triton" if "triton" in lane else "HIP"
    family = task.get("top10_family", contract.get("operator", "kernel_fn"))

    interface_spec = _extract_interface_spec(task)

    prompt = (
        f"Generate a standalone {language} {contract.get('operator', 'kernel')} kernel for AMD gfx942.\n"
        f"Contract: {json.dumps(contract)}\n\n"
        f"Implementation requirements:\n"
        f"- For Triton: write a @triton.jit kernel + a Python wrapper function\n"
        f"- For HIP: use torch.utils.cpp_extension.load_inline with C++ source + a Python wrapper\n"
        f"- Do not add architecture gates (the harness already verifies gfx942)\n"
        f"- Do not call AITER, CK, ASM, or reference implementations at runtime\n\n"
    )
    if interface_spec:
        prompt += f"{interface_spec}\n"
    prompt += "\nReturn ONLY the complete kernel.py source code. No explanations, no markdown code fences."

    return [
        {"role": "system", "content": "You are an expert GPU kernel engineer. Return only the requested code."},
        {"role": "user", "content": prompt},
    ]


def _clean_patch(raw: str) -> str:
    """Strip code fences, explanatory text, normalize git diff to standard unified diff."""
    text = raw.strip()

    # Find the code block containing the patch (may have prose before it)
    import re
    block_match = re.search(r'```(?:diff|patch|)\s*\n(.*?)```', text, re.DOTALL)
    if block_match:
        text = block_match.group(1).strip()
    else:
        # No code block - try to find the patch starting from first ---
        diff_start = re.search(r'^(diff --git|--- )', text, re.MULTILINE)
        if diff_start:
            text = text[diff_start.start():].strip()
        else:
            # Strip any leading code fences without closing
            for fence in ["```diff", "```patch", "```python", "```"]:
                if text.startswith(fence):
                    text = text[len(fence):]
                    break
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # Convert git diff header to standard unified diff
        if line.startswith("diff --git"):
            continue
        if line.startswith("index "):
            continue
        # Normalize a/kernel.py -> kernel.py for -p0
        if line.startswith("--- a/"):
            cleaned.append("--- " + line[6:])
        elif line.startswith("+++ b/"):
            cleaned.append("+++ " + line[6:])
        else:
            cleaned.append(line)
    return "\n".join(cleaned) + "\n"


def apply_patch(workspace: Path, patch_text: str) -> bool:
    """Apply unified diff patch to kernel.py with fuzzy matching."""
    from scripts.fuzzy_patch import apply_patch as fuzzy_apply

    clean = _clean_patch(patch_text)
    kernel_path = workspace / "kernel.py"
    if not kernel_path.exists():
        return False
    original_content = kernel_path.read_text()

    # Method 1: Python fuzzy patch (handles context mismatches)
    for text in [clean, patch_text]:
        result = fuzzy_apply(original_content, text)
        if result is not None and result != original_content:
            kernel_path.write_text(result)
            return True

    # Method 2: GNU patch with fuzz as fallback
    for text in [clean, patch_text]:
        for strip_level in [0, 1]:
            for fuzz in [0, 3, 10, 100]:
                try:
                    kernel_path.write_text(original_content)
                    for suffix in [".rej", ".orig"]:
                        p = workspace / f"kernel.py{suffix}"
                        if p.exists():
                            p.unlink()
                    cmd = ["patch", f"-p{strip_level}", "--no-backup-if-mismatch"]
                    if fuzz > 0:
                        cmd.append(f"--fuzz={fuzz}")
                    proc = subprocess.run(
                        cmd, input=text, capture_output=True, text=True,
                        timeout=10, cwd=str(workspace),
                    )
                    if proc.returncode == 0:
                        return True
                except Exception:
                    pass

    kernel_path.write_text(original_content)
    return False


def run_eval(workspace: Path, mode: str, gpu_id: int = 1) -> dict:
    """Run compile/correctness/performance."""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        proc = subprocess.run(
            ["python3", "scripts/task_runner.py", mode],
            cwd=str(workspace), capture_output=True, text=True, timeout=120, env=env,
        )
        return {"ok": proc.returncode == 0, "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def extract_perf_ms(stdout: str) -> float:
    """Extract performance in ms from task_runner output."""
    import re
    match = re.search(r"Perf:\s+([\d.]+)\s+ms", stdout)
    return float(match.group(1)) if match else 0.0


def benchmark_task_patch(task: dict, gpu_id: int) -> dict:
    """Run multi-turn patch optimization benchmark."""
    task_id = task["task_id"]
    task_dir = HELD_OUT_ROOT / "artifacts" / "kernel" / task_id / "initial"
    kernel_source = (task_dir / "kernel.py").read_text()

    result = {"task_id": task_id, "mode": "patch", "lane": task["lane"],
              "operator": task.get("top10_family", ""), "turns": [],
              "total_input_tokens": 0, "total_output_tokens": 0}

    current_source = kernel_source
    messages = build_patch_prompt(task, current_source)

    for turn in range(MAX_TURNS):
        turn_result = {"turn": turn + 1}
        t0 = time.time()
        try:
            response, usage = call_model(messages)
        except Exception as e:
            turn_result["error"] = str(e)
            result["turns"].append(turn_result)
            break
        turn_result["gen_time"] = time.time() - t0
        result["total_input_tokens"] += usage.get("input_tokens", 0)
        result["total_output_tokens"] += usage.get("output_tokens", 0)

        has_patch = "---" in response and "+++" in response
        turn_result["patch_generated"] = has_patch

        # Extract full code if model output code instead of patch
        full_code = None
        if not has_patch:
            extracted = _extract_code(response)
            if len(extracted.strip()) > 50 and ("import" in extracted[:100] or "def " in extracted[:200]):
                full_code = extracted
                turn_result["full_code_output"] = True

        workdir = Path(tempfile.mkdtemp(prefix=f"bench_patch_{task_id[:20]}_t{turn}_"))
        try:
            shutil.copytree(task_dir, workdir / "workspace", dirs_exist_ok=True)
            ws = workdir / "workspace"
            (ws / "kernel.py").write_text(current_source)

            if has_patch:
                applied = apply_patch(ws, response)
                turn_result["patch_applied"] = applied
            elif full_code is not None:
                # Model output full code instead of patch — use it directly
                code = _fix_arch_gate(full_code)
                family = task.get("top10_family", "")
                code = _add_function_alias(code, family)
                (ws / "kernel.py").write_text(code)
                applied = True
                turn_result["patch_applied"] = True
                turn_result["applied_as_full_code"] = True
            else:
                # Model output explanation text — retry
                turn_result["response_preview"] = response[:200]
                result["turns"].append(turn_result)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content":
                    "You must return ONLY a unified diff patch (starting with --- and +++), not an explanation. "
                    "Generate the patch now."
                })
                shutil.rmtree(workdir, ignore_errors=True)
                continue

            if not applied:
                result["turns"].append(turn_result)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": json.dumps({
                    "error_feedback": "Patch failed to apply. Please regenerate with correct context.",
                    "parent_source": {"kernel.py": current_source},
                })})
                shutil.rmtree(workdir, ignore_errors=True)
                continue

            for mode in ["compile", "correctness", "performance"]:
                r = run_eval(ws, mode, gpu_id)
                turn_result[mode] = r["ok"]
                if mode == "performance" and r["ok"]:
                    turn_result["perf_ms"] = extract_perf_ms(r["stdout"])
                if not r["ok"]:
                    turn_result[f"{mode}_error"] = r.get("stderr", r.get("error", ""))[-300:]
                    break

            result["turns"].append(turn_result)

            if turn_result.get("correctness"):
                break  # Success!

            # Error feedback for next turn
            error_msg = ""
            if not turn_result.get("compile"):
                error_msg = f"Compilation failed: {turn_result.get('compile_error', '')[:500]}"
                current_source = (ws / "kernel.py").read_text()
            elif not turn_result.get("correctness"):
                error_msg = f"Correctness check failed: {turn_result.get('correctness_error', '')[:500]}"
                current_source = (ws / "kernel.py").read_text()

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": json.dumps({
                "error_feedback": error_msg,
                "parent_source": {"kernel.py": current_source},
                "task_type": "error_recovery",
            })})
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # Aggregate turn-level metrics
    turns = result["turns"]
    result["total_turns"] = len(turns)
    result["patch_generated"] = any(t.get("patch_generated") for t in turns)
    result["patch_applied"] = any(t.get("patch_applied") for t in turns)
    result["compile"] = any(t.get("compile") for t in turns)
    result["correctness"] = any(t.get("correctness") for t in turns)
    result["performance"] = any(t.get("performance") for t in turns)
    result["first_pass"] = turns[0].get("correctness", False) if turns else False
    # Turns to pass (first turn that gets correctness)
    result["turns_to_pass"] = next((t["turn"] for t in turns if t.get("correctness")), None)
    # Turns to compile (first turn that compiles)
    result["turns_to_compile"] = next((t["turn"] for t in turns if t.get("compile")), None)
    # Best perf
    perfs = [t.get("perf_ms", 0) for t in turns if t.get("perf_ms")]
    result["best_perf_ms"] = min(perfs) if perfs else None
    # Error recovery: did model fix an error after feedback?
    result["error_recovered"] = (
        not turns[0].get("correctness", False) and
        any(t.get("correctness") for t in turns[1:])
    ) if len(turns) > 1 else False
    # Monotonic: did each compile turn have better/equal perf than previous?
    perf_sequence = [t.get("perf_ms") for t in turns if t.get("perf_ms")]
    result["monotonic_improvement"] = all(
        perf_sequence[i] <= perf_sequence[i-1] for i in range(1, len(perf_sequence))
    ) if len(perf_sequence) > 1 else None

    return result

    return result


def _fix_arch_gate(code: str) -> str:
    """Fix common incorrect gfx942 architecture gates."""
    import re
    # Pattern 1: gcnArchName check that fails on ROCm
    # Replace various broken arch checks with a working one
    broken_patterns = [
        r'torch\.cuda\.get_device_properties\(\d+\)\.gcnArchName\s*[!=]=\s*["\']gfx942["\']',
        r'torch\.cuda\.get_device_properties\(\d+\)\.gcnArchName\s*[!=]=\s*["\']gfx94',
        r'torch\.cuda\.get_device_arch_name\(\d+\)',
    ]
    for pat in broken_patterns:
        if re.search(pat, code):
            # Remove the entire if-block that raises RuntimeError about architecture
            code = re.sub(
                r'if\s+(?:not\s+)?.*(?:gcnArchName|get_device_arch).*?:\s*\n\s*raise\s+RuntimeError\(.*?(?:architecture|gfx942).*?\)\s*\n',
                '# Architecture gate removed by benchmark harness (gfx942 verified)\n',
                code, flags=re.DOTALL,
            )
    # Pattern 2: simple string check that might fail
    code = re.sub(
        r'assert\s+["\']gfx942["\']\s+in\s+.*?,\s*["\'].*?architecture.*?["\']',
        '# Architecture assertion removed (gfx942 verified)',
        code,
    )
    # Pattern 3: "This kernel is for NVIDIA GPUs" NotImplementedError
    code = re.sub(
        r'raise\s+NotImplementedError\(.*?(?:NVIDIA|CUDA|not AMD).*?\)\s*\n',
        '# NVIDIA-only gate removed (AMD gfx942 verified)\npass\n',
        code, flags=re.IGNORECASE,
    )
    # Pattern 4: Unsupported architecture assertion
    code = re.sub(
        r'raise\s+RuntimeError\(.*?(?:Unsupported|unsupported).*?architecture.*?\)\s*\n',
        '# Architecture gate removed (gfx942 verified)\npass\n',
        code, flags=re.DOTALL,
    )
    # Fix hip_cxxflags -> extra_cflags for load_inline compatibility
    code = code.replace("hip_cxxflags=", "extra_cflags=")
    return code


def _add_function_alias(code: str, expected_name: str) -> str:
    """If the expected function name doesn't exist, add an alias."""
    if not expected_name:
        return code
    # Check if the expected name already exists as a def or assignment
    import re
    if re.search(rf'def\s+{expected_name}\s*\(', code):
        return code
    if re.search(rf'^{expected_name}\s*=', code, re.MULTILINE):
        return code

    # Find candidate function names (top-level defs that aren't private)
    defs = re.findall(r'^def\s+(\w+)\s*\(', code, re.MULTILINE)
    public_defs = [d for d in defs if not d.startswith('_')]

    if len(public_defs) == 1:
        # Only one public function, alias it
        code += f"\n{expected_name} = {public_defs[0]}\n"
    elif public_defs:
        # Try to find one that matches the operator name partially
        for d in public_defs:
            if expected_name.replace('_', '') in d.replace('_', '').lower():
                code += f"\n{expected_name} = {d}\n"
                break
        else:
            # Use the last public def as it's likely the main entry point
            code += f"\n{expected_name} = {public_defs[-1]}\n"
    return code


def _extract_code(response: str) -> str:
    """Extract code from model response, handling code blocks and prose."""
    import re
    code = response
    blocks = re.findall(r"```(?:python|cpp|c\+\+|hip)?\s*\n(.*?)```", code, re.DOTALL)
    if blocks:
        code = max(blocks, key=len)
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0]
    code = code.strip()
    if code.startswith("python\n"):
        code = code[7:]
    return code


def benchmark_task_generation(task: dict, gpu_id: int) -> dict:
    """Run multi-turn from-scratch generation benchmark."""
    task_id = task["task_id"]
    task_dir = HELD_OUT_ROOT / "artifacts" / "kernel" / task_id / "initial"

    result = {"task_id": task_id, "mode": "generation", "lane": task["lane"],
              "operator": task.get("top10_family", ""), "turns": [],
              "total_input_tokens": 0, "total_output_tokens": 0}

    messages = build_generation_prompt(task)
    family = task.get("top10_family", "")

    for turn in range(MAX_TURNS):
        turn_result = {"turn": turn + 1}
        t0 = time.time()
        try:
            response, usage = call_model(messages, max_tokens=16384)
        except Exception as e:
            turn_result["error"] = str(e)
            result["turns"].append(turn_result)
            break
        turn_result["gen_time"] = time.time() - t0
        result["total_input_tokens"] += usage.get("input_tokens", 0)
        result["total_output_tokens"] += usage.get("output_tokens", 0)

        code = _extract_code(response)
        turn_result["code_generated"] = len(code.strip()) > 50

        code = _fix_arch_gate(code)
        code = _add_function_alias(code, family)

        workdir = Path(tempfile.mkdtemp(prefix=f"bench_gen_{task_id[:20]}_t{turn}_"))
        try:
            shutil.copytree(task_dir, workdir / "workspace", dirs_exist_ok=True)
            ws = workdir / "workspace"
            (ws / "kernel.py").write_text(code)

            for mode in ["compile", "correctness", "performance"]:
                r = run_eval(ws, mode, gpu_id)
                turn_result[mode] = r["ok"]
                if mode == "performance" and r["ok"]:
                    turn_result["perf_ms"] = extract_perf_ms(r["stdout"])
                if not r["ok"]:
                    turn_result[f"{mode}_error"] = r.get("stderr", r.get("error", ""))[-300:]
                    break

            result["turns"].append(turn_result)

            if turn_result.get("correctness"):
                break  # Success!

            # Error feedback for retry
            error_msg = ""
            if not turn_result.get("compile"):
                error_msg = f"Compilation failed: {turn_result.get('compile_error', '')[:500]}"
            elif not turn_result.get("correctness"):
                error_msg = f"Correctness check failed: {turn_result.get('correctness_error', '')[:500]}"

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                f"Your kernel had an error. Please fix it and return the complete corrected kernel.py.\n"
                f"Error: {error_msg}\n"
                f"Return only the complete kernel.py source code, no explanations."
            })
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # Aggregate metrics
    turns = result["turns"]
    result["total_turns"] = len(turns)
    result["code_generated"] = any(t.get("code_generated") for t in turns)
    result["compile"] = any(t.get("compile") for t in turns)
    result["correctness"] = any(t.get("correctness") for t in turns)
    result["performance"] = any(t.get("performance") for t in turns)
    result["first_pass"] = turns[0].get("correctness", False) if turns else False
    result["turns_to_pass"] = next((t["turn"] for t in turns if t.get("correctness")), None)
    result["turns_to_compile"] = next((t["turn"] for t in turns if t.get("compile")), None)
    perfs = [t.get("perf_ms", 0) for t in turns if t.get("perf_ms")]
    result["best_perf_ms"] = min(perfs) if perfs else None
    result["error_recovered"] = (
        not turns[0].get("correctness", False) and
        any(t.get("correctness") for t in turns[1:])
    ) if len(turns) > 1 else False
    perf_sequence = [t.get("perf_ms") for t in turns if t.get("perf_ms")]
    result["monotonic_improvement"] = all(
        perf_sequence[i] <= perf_sequence[i-1] for i in range(1, len(perf_sequence))
    ) if len(perf_sequence) > 1 else None

    return result


def run_benchmark(label: str, benchmark_mode: str = "both"):
    tasks = load_passing_tasks()
    print(f"Loaded {len(tasks)} passing held-out tasks")

    triton_tasks = [t for t in tasks if "triton" in t["lane"]]
    hip_tasks = [t for t in tasks if "hip" in t["lane"]]
    print(f"  Triton: {len(triton_tasks)}, HIP: {len(hip_tasks)}")

    all_results = []

    for mode_name, bench_fn in [("patch", benchmark_task_patch), ("generation", benchmark_task_generation)]:
        if benchmark_mode != "both" and benchmark_mode != mode_name:
            continue
        print(f"\n=== Running {mode_name} benchmark ({label}) ===")

        for i, task in enumerate(tasks):
            gpu_id = (i % 7) + 1
            print(f"  [{i+1}/{len(tasks)}] {task['task_id']} (GPU {gpu_id})...", end=" ", flush=True)
            r = bench_fn(task, gpu_id)
            all_results.append(r)

            status = "PASS" if r.get("correctness") else ("COMPILE" if r.get("compile") else "FAIL")
            perf = f" {r.get('best_perf_ms', 0):.3f}ms" if r.get("performance") else ""
            turns = r.get("total_turns", 1)
            recovered = " RECOVERED" if r.get("error_recovered") else ""
            print(f"{status}{perf} (turns={turns}{recovered})")

    # Save results
    output_dir = HELD_OUT_ROOT / "receipts" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary with agent loop metrics
    for mode_name in ["patch", "generation"]:
        mode_results = [r for r in all_results if r.get("mode") == mode_name]
        if not mode_results:
            continue
        total = len(mode_results)
        patch_gen = sum(1 for r in mode_results if r.get("patch_generated") or r.get("code_generated"))
        compiled = sum(1 for r in mode_results if r.get("compile"))
        correct = sum(1 for r in mode_results if r.get("correctness"))
        perf_ok = sum(1 for r in mode_results if r.get("performance"))
        first_pass = sum(1 for r in mode_results if r.get("first_pass"))
        error_recovered = sum(1 for r in mode_results if r.get("error_recovered"))

        print(f"\n=== {label} {mode_name} Summary ===")
        print(f"  --- Result Quality ---")
        print(f"  Total tasks:    {total}")
        print(f"  Generated:      {patch_gen}/{total} ({100*patch_gen/total:.0f}%)")
        print(f"  Compiled:       {compiled}/{total} ({100*compiled/total:.0f}%)")
        print(f"  Correct:        {correct}/{total} ({100*correct/total:.0f}%)")
        print(f"  Perf OK:        {perf_ok}/{total} ({100*perf_ok/total:.0f}%)")

        print(f"  --- Agent Loop Efficiency ---")
        print(f"  First-pass rate:     {first_pass}/{total} ({100*first_pass/total:.0f}%)")
        print(f"  Pass@{MAX_TURNS} (multi-turn): {correct}/{total} ({100*correct/total:.0f}%)")
        print(f"  Error recovery:      {error_recovered}/{total} ({100*error_recovered/total:.0f}%)")

        # Turns to pass (avg, among successful)
        ttp = [r["turns_to_pass"] for r in mode_results if r.get("turns_to_pass")]
        if ttp:
            print(f"  Avg turns to pass:   {sum(ttp)/len(ttp):.1f} (among {len(ttp)} successes)")

        # Turns to compile
        ttc = [r["turns_to_compile"] for r in mode_results if r.get("turns_to_compile")]
        if ttc:
            print(f"  Avg turns to compile: {sum(ttc)/len(ttc):.1f} (among {len(ttc)} compiled)")

        # Token efficiency
        total_in = sum(r.get("total_input_tokens", 0) for r in mode_results)
        total_out = sum(r.get("total_output_tokens", 0) for r in mode_results)
        total_tok = total_in + total_out
        if total_tok > 0:
            print(f"  --- Token Efficiency ---")
            print(f"  Total tokens:        {total_tok:,} ({total_in:,} in + {total_out:,} out)")
            print(f"  Avg tokens/task:     {total_tok/total:,.0f}")
            if correct > 0:
                cost_per_pass = total_tok / correct
                print(f"  Cost-of-Pass:        {cost_per_pass:,.0f} tokens/correct")
            print(f"  Output ratio:        {100*total_out/total_tok:.1f}%")

        # Compile/correctness error rates (turn-level)
        all_turns = [t for r in mode_results for t in r.get("turns", [])]
        if all_turns:
            turns_with_code = [t for t in all_turns if t.get("patch_generated") or t.get("code_generated") or t.get("patch_applied")]
            compile_errors = sum(1 for t in turns_with_code if not t.get("compile", True))
            correctness_errors = sum(1 for t in turns_with_code if t.get("compile") and not t.get("correctness", True))
            print(f"  --- Trajectory Quality ---")
            print(f"  Total turns:         {len(all_turns)}")
            print(f"  Avg turns/task:      {len(all_turns)/total:.1f}")
            if turns_with_code:
                print(f"  Compile error rate:  {100*compile_errors/len(turns_with_code):.0f}%")
                compiled_turns = sum(1 for t in turns_with_code if t.get("compile"))
                if compiled_turns:
                    print(f"  Correctness error:   {100*correctness_errors/compiled_turns:.0f}%")

        # Per-lane breakdown
        print(f"  --- Per-Lane ---")
        for lane in ["triton_gfx942", "hip_gfx942"]:
            lane_r = [r for r in mode_results if r.get("lane") == lane]
            if lane_r:
                l_correct = sum(1 for r in lane_r if r.get("correctness"))
                l_first = sum(1 for r in lane_r if r.get("first_pass"))
                l_recovered = sum(1 for r in lane_r if r.get("error_recovered"))
                print(f"  {lane}: {l_correct}/{len(lane_r)} correct, {l_first} first-pass, {l_recovered} recovered")

        # Per-operator breakdown
        print(f"  --- Per-Operator ---")
        from collections import Counter
        ops = Counter()
        ops_correct = Counter()
        ops_first = Counter()
        ops_recovered = Counter()
        for r in mode_results:
            op = r.get("operator", "?")
            ops[op] += 1
            if r.get("correctness"): ops_correct[op] += 1
            if r.get("first_pass"): ops_first[op] += 1
            if r.get("error_recovered"): ops_recovered[op] += 1
        for op in sorted(ops):
            print(f"  {op:25s} correct={ops_correct[op]:2d}/{ops[op]:2d}  first={ops_first[op]:2d}  recovered={ops_recovered[op]:2d}")

    print(f"\nResults saved to {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "base"
    mode = sys.argv[2] if len(sys.argv) > 2 else "both"
    run_benchmark(label, mode)
