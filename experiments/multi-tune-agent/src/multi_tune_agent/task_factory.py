"""Generate supported GEAK task directories from structured user requests."""

from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from geak_utils.paths import example_task_path
from geak_utils.local_templates import VerifiedTemplateRecord
from geak_utils.templates import (
    FORMAT_ALIASES,
    get_gemm_template,
    normalize_gemm_format,
    validate_template_target,
)

from .config import MultiTuneConfig


@dataclass(frozen=True)
class GeneratedGemmTask:
    case_id: str
    task_dir: Path
    target_gpu: str
    architecture: str
    dtype: str
    case_type: str
    backend: str
    m: int
    n: int
    k: int
    request: str

    @property
    def format(self) -> str:
        return self.dtype


@dataclass(frozen=True)
class GeneratedKernelTask:
    """A permanent task backed by an already promoted, verified template."""

    case_id: str
    task_dir: Path
    case_type: str
    operator: str
    architecture: str
    backend: str
    request: str
    contract_hash: str
    provenance: Any


def parse_gemm_request(request: str) -> dict[str, Any]:
    """Extract one dense-GEMM shape from English or Chinese request text."""

    text = unicodedata.normalize("NFKC", request).strip()
    if not text:
        raise ValueError("request cannot be empty")

    values: dict[str, int] = {}
    for name in ("M", "N", "K"):
        match = re.search(
            r"(?<![A-Za-z])%s\s*(?:=|:|：|为|是)?\s*(\d+)" % name,
            text,
            re.IGNORECASE,
        )
        if match:
            values[name.lower()] = int(match.group(1))
    if len(values) != 3:
        marker = re.search(
            r"(?<![A-Za-z])M\s*[/,，、\s]*N\s*[/,，、\s]*K(?![A-Za-z])"
            r"|分别\s*(?:是|为|shi|sh|are|=|:)?"
            r"|(?:dimensions?|shape)\s*(?:are|is|=|:)?",
            text,
            re.IGNORECASE,
        )
        trailing_numbers = (
            re.findall(
                r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])",
                text[marker.end() :],
            )
            if marker
            else []
        )
        if len(trailing_numbers) >= 3:
            values = {
                "m": int(trailing_numbers[0]),
                "n": int(trailing_numbers[1]),
                "k": int(trailing_numbers[2]),
            }
    if len(values) != 3 and re.search(r"\bgemm\b", text, re.IGNORECASE):
        request_numbers = re.findall(
            r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])", text
        )
        if len(request_numbers) == 3:
            values = {
                "m": int(request_numbers[0]),
                "n": int(request_numbers[1]),
                "k": int(request_numbers[2]),
            }
    if len(values) != 3:
        bare = re.fullmatch(
            r"\s*(\d+)\s*[/,，、xX*×\s]\s*"
            r"(\d+)\s*[/,，、xX*×\s]\s*(\d+)\s*",
            text,
        )
        if bare:
            values = {
                "m": int(bare.group(1)),
                "n": int(bare.group(2)),
                "k": int(bare.group(3)),
            }
    missing = [name.upper() for name in ("m", "n", "k") if name not in values]
    if missing:
        raise ValueError(
            "could not parse %s; write dimensions as M=1234, N=2048, K=4096 "
            "or MNK分别是/shi 1234、2048、4096" % ", ".join(missing)
        )
    if any(values[name] < 1 for name in ("m", "n", "k")):
        raise ValueError("M, N, and K must be positive")

    alias_pattern = "|".join(
        sorted((re.escape(alias) for alias in FORMAT_ALIASES), key=len, reverse=True)
    )
    dtype_match = re.search(
        r"(?<![A-Za-z0-9])(%s)(?![A-Za-z0-9])" % alias_pattern,
        text,
        re.IGNORECASE,
    )
    dtype = normalize_gemm_format(dtype_match.group(1) if dtype_match else "fp16")

    gpu_match = re.search(r"MI\s*-?\s*(\d{3,4}X?)", text, re.IGNORECASE)
    gfx_match = re.search(r"\bgfx\s*-?\s*(942|950)\b", text, re.IGNORECASE)
    if gpu_match:
        target_gpu = "MI" + gpu_match.group(1).upper()
    elif gfx_match:
        target_gpu = "gfx" + gfx_match.group(1)
    else:
        target_gpu = "AMD-Instinct"
    return {
        "target_gpu": target_gpu,
        "dtype": dtype,
        "m": values["m"],
        "n": values["n"],
        "k": values["k"],
        "request": text,
    }


def default_generated_case_id(spec: Mapping[str, Any]) -> str:
    gpu = re.sub(r"[^a-z0-9]+", "-", str(spec["target_gpu"]).lower()).strip("-")
    return "%s-gemm-m%d-n%d-k%d-%s" % (
        gpu,
        int(spec["m"]),
        int(spec["n"]),
        int(spec["k"]),
        str(spec["dtype"]).lower(),
    )


def generate_gemm_task(
    config: MultiTuneConfig,
    request: str,
    *,
    case_id: Optional[str] = None,
    task_root: Optional[Path] = None,
    parsed_spec: Optional[Mapping[str, Any]] = None,
) -> GeneratedGemmTask:
    """Materialize one GEMM task from the registered format template."""

    spec = (
        _validated_gemm_spec(parsed_spec, request)
        if parsed_spec is not None
        else parse_gemm_request(request)
    )
    descriptor, architecture = validate_template_target(
        spec["dtype"], spec["target_gpu"]
    )
    if descriptor.format == "mxfp4" and int(spec["k"]) % 32:
        raise ValueError("MXFP4 GEMM requires K to be divisible by 32")
    resolved_case_id = case_id or default_generated_case_id(spec)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", resolved_case_id):
        raise ValueError("invalid generated case ID: %s" % resolved_case_id)

    template = example_task_path(descriptor.template_dir)
    if not (template / "kernel.py").is_file():
        raise FileNotFoundError("trusted GEMM seed is missing: %s" % template)
    root = (
        Path(task_root).expanduser().resolve()
        if task_root
        else Path(__file__).resolve().parents[2] / "examples" / "tasks"
    )
    root.mkdir(parents=True, exist_ok=True)
    task_dir = root / resolved_case_id
    if task_dir.exists():
        raise FileExistsError(
            "generated task already exists: %s; choose another case ID" % task_dir
        )
    shutil.copytree(
        template,
        task_dir,
        ignore=shutil.ignore_patterns("build", "__pycache__", "*.pyc"),
    )

    if descriptor.format == "fp16":
        task_runner = _render_task_runner(
            int(spec["m"]), int(spec["n"]), int(spec["k"])
        )
        scripts_dir = task_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "task_runner.py").write_text(task_runner, encoding="utf-8")

    config_path = task_dir / "config.yaml"
    task_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    task_config["prompt"] = {"instructions": request.strip()}
    config_path.write_text(
        yaml.safe_dump(task_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest = {
        "case_id": resolved_case_id,
        "target_gpu": spec["target_gpu"],
        "architecture": architecture,
        "operator": "gemm",
        "dtype": spec["dtype"],
        "format": descriptor.format,
        "case_type": descriptor.case_type,
        "backend": descriptor.backend,
        "input_contract": descriptor.input_contract,
        "scale_contract": descriptor.scale_contract,
        "output_contract": descriptor.output_contract,
        "shapes": [{"M": spec["m"], "N": spec["n"], "K": spec["k"]}],
        "request": request.strip(),
        "generated_from": str(template),
    }
    (task_dir / "task_spec.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return GeneratedGemmTask(
        case_id=resolved_case_id,
        task_dir=task_dir,
        target_gpu=str(spec["target_gpu"]),
        architecture=architecture,
        dtype=str(spec["dtype"]),
        case_type=descriptor.case_type,
        backend=descriptor.backend,
        m=int(spec["m"]),
        n=int(spec["n"]),
        k=int(spec["k"]),
        request=request.strip(),
    )


def _validated_gemm_spec(
    value: Mapping[str, Any], request: str
) -> dict[str, Any]:
    required = ("target_gpu", "dtype", "m", "n", "k")
    missing = [
        name for name in required if value.get(name) is None or value.get(name) == ""
    ]
    if missing:
        raise ValueError("recognized GEMM request is missing: %s" % ", ".join(missing))
    try:
        dimensions = {name: int(value[name]) for name in ("m", "n", "k")}
    except (TypeError, ValueError) as exc:
        raise ValueError("recognized M, N, and K must be integers") from exc
    if any(number < 1 for number in dimensions.values()):
        raise ValueError("M, N, and K must be positive")
    dtype = normalize_gemm_format(value["dtype"])
    return {
        "target_gpu": str(value["target_gpu"]).strip(),
        "dtype": dtype,
        **dimensions,
        "request": request.strip(),
    }


def register_generated_case(
    catalog_path: Path, task: GeneratedGemmTask | GeneratedKernelTask
) -> None:
    """Insert or update one generated task in the permanent case catalog."""

    path = Path(catalog_path).expanduser().resolve()
    if path.is_file():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("case catalog root must be a mapping")
    tasks = payload.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError("case catalog 'tasks' must be a list")
    relative_path = os.path.relpath(task.task_dir, path.parent)
    if isinstance(task, GeneratedKernelTask):
        entry = {
            "id": task.case_id,
            "type": task.case_type,
            "kernel_path": relative_path,
            "direction": task.request,
            "operator": task.operator,
            "backend": task.backend,
            "architecture": task.architecture,
            "contract_hash": task.contract_hash,
            "provenance": task.provenance,
        }
    else:
        entry = {
            "id": task.case_id,
            "type": task.case_type,
            "kernel_path": relative_path,
            "direction": task.request,
            "format": task.format,
            "backend": task.backend,
            "target_gpu": task.target_gpu,
            "architecture": task.architecture,
            "scale_contract": get_gemm_template(task.format).scale_contract,
        }
    for index, value in enumerate(tasks):
        if isinstance(value, dict) and value.get("id") == task.case_id:
            tasks[index] = entry
            break
    else:
        tasks.append(entry)
    payload["tasks"] = tasks
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_verified_template_task(
    config: MultiTuneConfig,
    record: VerifiedTemplateRecord,
    *,
    request: Optional[str] = None,
    case_id: Optional[str] = None,
) -> GeneratedKernelTask:
    """Register an already-promoted verified template in the main case catalog."""

    if not isinstance(record, VerifiedTemplateRecord):
        raise TypeError("record must be a VerifiedTemplateRecord")
    resolved_case_id = case_id or _generated_kernel_case_id(record)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", resolved_case_id):
        raise ValueError("invalid generated case ID: %s" % resolved_case_id)
    task_dir = record.template_path.expanduser().resolve(strict=True)
    metadata_path = task_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("verified template metadata.json must contain a mapping")
    if metadata.get("contract_hash") != record.contract_hash:
        raise ValueError("verified template contract hash does not match registry")
    task = GeneratedKernelTask(
        case_id=resolved_case_id,
        task_dir=task_dir,
        case_type=record.case_type,
        operator=record.operator,
        architecture=record.architecture,
        backend=record.backend,
        request=(request or record.direction).strip(),
        contract_hash=record.contract_hash,
        provenance=record.provenance,
    )
    register_generated_case(config.cases_path, task)
    return task


def materialize_and_register_generated_task(
    config: MultiTuneConfig,
    record: VerifiedTemplateRecord,
    *,
    request: Optional[str] = None,
    case_id: Optional[str] = None,
) -> GeneratedKernelTask:
    """Compatibility alias for materializing a verified generated task."""

    return materialize_verified_template_task(
        config, record, request=request, case_id=case_id
    )


def _generated_kernel_case_id(record: VerifiedTemplateRecord) -> str:
    operator = re.sub(r"[^a-z0-9]+", "-", record.operator.lower()).strip("-")
    return "%s-%s" % (operator or "kernel", record.contract_hash[:12])


def _render_task_runner(m: int, n: int, k: int) -> str:
    case_name = "m%d_n%d_k%d" % (m, n, k)
    return '''#!/usr/bin/env python3
"""Generated compile, correctness, and benchmark harness for one FP16 GEMM."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))
os.chdir(TASK_DIR)

from kernel import gemm  # noqa: E402


CASES = [(%r, %d, %d, %d)]


def make_case(index):
    _, m, n, k = CASES[index]
    torch.manual_seed(100 + index)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    return a, b


def time_ms(fn, warmup=10, repeats=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def compile_kernel():
    a, b = make_case(0)
    gemm(a, b)
    torch.cuda.synchronize()
    print("Compilation: PASS")


def check_correctness():
    for index, (name, _, _, _) in enumerate(CASES):
        a, b = make_case(index)
        actual = gemm(a, b)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print("Correctness: PASS (%%s)" %% name)


def benchmark():
    results = []
    for index, (name, m, n, k) in enumerate(CASES):
        a, b = make_case(index)
        latency = time_ms(lambda: gemm(a, b))
        results.append({
            "test_case_id": name,
            "execution_time_ms": latency,
            "params": {"M": m, "N": n, "K": k, "dtype": "fp16"},
        })
        print("Perf: %%.6f ms (%%s)" %% (latency, name))
    build = TASK_DIR / "build"
    build.mkdir(exist_ok=True)
    (build / "performance_report.json").write_text(
        json.dumps({"test_cases": results}, indent=2) + "\\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    if mode == "compile":
        compile_kernel()
    elif mode == "correctness":
        check_correctness()
    else:
        benchmark()


if __name__ == "__main__":
    main()
''' % (case_name, m, n, k)
