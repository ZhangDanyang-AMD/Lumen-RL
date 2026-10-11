"""Deterministic, source-backed gfx942 candidate contract expansion.

The extractor intentionally supports a small, auditable subset of Python AST.
It never imports AITER and never forms a Cartesian product of parameters.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


LOCKED_AITER_SHA = "926eb3d059efd3c866c8f53ecb8b1fb8fb7135e8"
DEFAULT_AITER_ROOT = Path("/home/danyzhan/aiter")


@dataclass(frozen=True)
class PythonSource:
    path: str
    test_id: str
    value_names: tuple[str, ...]
    dimensions: tuple[str, ...]
    operator: str
    extraction: str
    function_name: str | None = None
    parameter_name: str | None = None
    dtype: str = "bf16"
    layout: str = "contiguous"
    priority: str = "P0"
    source_language: str = "triton"
    backend: str = "triton"
    oracle_method: str = ""
    architecture_evidence: str | None = None
    extra_contract: tuple[tuple[str, Any], ...] = ()


PYTHON_SOURCES = (
    PythonSource(
        "op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py",
        "test_gemm_a16_w16",
        ("M", "N", "K"),
        ("M", "N", "K"),
        "gemm",
        "function",
        function_name="get_x_vals",
        layout="TN",
        oracle_method="torch.nn.functional.linear",
        architecture_evidence=(
            "aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/DEFAULT.json"
        ),
        extra_contract=(("weight_dtype", "bf16"), ("accum_dtype", "fp32")),
    ),
    PythonSource(
        "op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py",
        "test_gemm_a16_w16_activation",
        ("M", "N", "K"),
        ("M", "N", "K"),
        "gemm_activation",
        "function",
        function_name="get_fewer_x_vals",
        layout="TN",
        priority="P1",
        oracle_method="F.linear followed by the selected torch activation",
        architecture_evidence=(
            "aiter/ops/triton/configs/gfx942/triton/gemm/gemm_a16w16/DEFAULT.json"
        ),
        extra_contract=(
            ("weight_dtype", "bf16"),
            ("accum_dtype", "fp32"),
            ("activation_options", ["gelu", "gelu_tanh", "silu"]),
        ),
    ),
    PythonSource(
        "op_tests/triton_tests/normalization/test_rmsnorm.py",
        "test_rmsnorm",
        ("M", "N"),
        ("M", "N"),
        "rms_norm",
        "function",
        function_name="get_vals",
        oracle_method="local pure-torch torch_rmsnorm",
        extra_contract=(("epsilon", 1e-5),),
    ),
    PythonSource(
        "op_tests/triton_tests/test_softmax.py",
        "test_softmax",
        ("M", "N"),
        ("M", "N"),
        "softmax",
        "parametrize",
        parameter_name="M, N",
        dtype="fp32",
        layout="row_major",
        oracle_method="torch.softmax",
        extra_contract=(("axis", 1),),
    ),
    PythonSource(
        "op_tests/triton_tests/quant/test_quant.py",
        "test_static_per_tensor_quant",
        ("M", "N"),
        ("M", "N"),
        "static_per_tensor_quant",
        "parametrize",
        parameter_name="M, N",
        oracle_method="pure-torch x divided by scalar scale then dtype cast",
        extra_contract=(("output_dtype", "fp8_e4m3fnuz"), ("scale", "fp32_scalar")),
    ),
    PythonSource(
        "op_tests/triton_tests/quant/test_quant.py",
        "test_dynamic_per_tensor_quant",
        ("M", "N"),
        ("M", "N"),
        "dynamic_per_tensor_quant",
        "parametrize",
        parameter_name="M, N",
        oracle_method="pure-torch global absmax scaling then dtype cast",
        extra_contract=(("output_dtype", "fp8_e4m3fnuz"), ("scale", "fp32_scalar")),
    ),
    PythonSource(
        "op_tests/triton_tests/quant/test_quant.py",
        "test_dynamic_per_token_quant",
        ("M", "N"),
        ("M", "N"),
        "dynamic_per_token_quant",
        "parametrize",
        parameter_name="M, N",
        oracle_method="pure-torch row absmax scaling then dtype cast",
        extra_contract=(("output_dtype", "fp8_e4m3fnuz"), ("scale", "fp32_per_row")),
    ),
    PythonSource(
        "op_tests/triton_tests/fusions/test_fused_mul_add.py",
        "test_mul_add",
        ("shape",),
        ("shape",),
        "fused_mul_add",
        "parametrize",
        parameter_name="shape",
        priority="P1",
        oracle_method="pure-torch a * x.float() + b then input dtype cast",
        extra_contract=(
            (
                "operand_kinds",
                ["python_float_scalar", "python_int_scalar", "tensor_scalar", "tensor"],
            ),
        ),
    ),
    PythonSource(
        "op_tests/triton_tests/fusions/test_fused_silu_mul.py",
        "test_fused_silu_mul",
        ("shape",),
        ("shape",),
        "fused_silu_mul",
        "parametrize",
        parameter_name="shape",
        layout="split_last_dimension",
        priority="P1",
        oracle_method="local pure-torch silu_exp2 reference multiplied by gate",
    ),
)


class SafeEvaluator:
    """Evaluate only literal/list-building syntax used by locked test data."""

    def __init__(self, functions: dict[str, ast.FunctionDef], constants: dict[str, Any]):
        self.functions = functions
        self.constants = constants

    def expression(self, node: ast.AST, env: dict[str, Any] | None = None) -> Any:
        env = {} if env is None else env
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [self.expression(item, env) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in self.constants:
                return self.constants[node.id]
            raise ValueError(f"unsupported name: {node.id}")
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Mult, ast.Pow)
        ):
            left = self.expression(node.left, env)
            right = self.expression(node.right, env)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left**right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [self.expression(arg, env) for arg in node.args]
            if node.func.id == "range":
                return range(*args)
            if node.func.id in self.functions and not args:
                return self.function(node.func.id)
        if isinstance(node, ast.ListComp) and len(node.generators) == 1:
            generator = node.generators[0]
            if generator.ifs or generator.is_async or not isinstance(generator.target, ast.Name):
                raise ValueError("conditional/async comprehensions are not supported")
            result = []
            for value in self.expression(generator.iter, env):
                child_env = {**env, generator.target.id: value}
                result.append(self.expression(node.elt, child_env))
            return result
        raise ValueError(f"unsupported source expression: {ast.dump(node, include_attributes=False)}")

    def function(self, name: str) -> list[Any]:
        env: dict[str, Any] = {}
        for statement in self.functions[name].body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if not isinstance(target, ast.Name):
                    raise ValueError("only simple assignments are supported")
                env[target.id] = self.expression(statement.value, env)
            elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                if not isinstance(statement.op, ast.Add):
                    raise ValueError("only list += is supported")
                env[statement.target.id] += self.expression(statement.value, env)
            elif isinstance(statement, ast.Return):
                value = self.expression(statement.value, env)
                if not isinstance(value, list):
                    raise ValueError(f"{name} did not return a list")
                return value
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue
            else:
                raise ValueError(f"unsupported statement in {name}: {type(statement).__name__}")
        raise ValueError(f"{name} has no return")


def _python_values(root: Path, spec: PythonSource) -> list[tuple[Any, int]]:
    source = (root / spec.path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=spec.path)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    constants: dict[str, Any] = {}
    bootstrap = SafeEvaluator(functions, constants)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                constants[node.targets[0].id] = bootstrap.expression(node.value)
            except ValueError:
                pass
    evaluator = SafeEvaluator(functions, constants)

    if spec.extraction == "function":
        assert spec.function_name
        function = functions[spec.function_name]
        values = evaluator.function(spec.function_name)
        return [(value, function.lineno) for value in values]

    test = functions[spec.test_id]
    for decorator in test.decorator_list:
        if not isinstance(decorator, ast.Call) or len(decorator.args) < 2:
            continue
        try:
            parameter_name = evaluator.expression(decorator.args[0])
        except ValueError:
            continue
        if parameter_name == spec.parameter_name:
            values_node = decorator.args[1]
            values = evaluator.expression(values_node)
            if not isinstance(values, list):
                raise ValueError(f"{spec.path}:{spec.test_id} parameter data is not a list")
            if isinstance(values_node, ast.List):
                return [
                    (value, item.lineno)
                    for value, item in zip(values, values_node.elts, strict=True)
                ]
            return [(value, values_node.lineno) for value in values]
    raise ValueError(
        f"missing parameter {spec.parameter_name!r} on {spec.path}:{spec.test_id}"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _shape_cluster(shape: dict[str, Any]) -> str:
    numeric: list[int] = []
    for value in shape.values():
        if isinstance(value, int):
            numeric.append(value)
        elif isinstance(value, (list, tuple)):
            numeric.extend(item for item in value if isinstance(item, int))
    volume = 1
    for value in numeric:
        volume *= value
    if volume <= 4096:
        size = "small"
    elif volume <= 16_777_216:
        size = "medium"
    else:
        size = "large"
    irregular = any(value & (value - 1) for value in numeric if value > 0)
    return f"{len(numeric)}d-{size}-{'irregular' if irregular else 'power2'}"


def _candidate_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "gfx942-auto-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _contract_family_id(
    lineage: str, operator: str, dtype: str, layout: str
) -> str:
    """Group shape records with the same source-backed operator contract."""
    payload = {
        "lineage": lineage,
        "operator": operator,
        "dtype": dtype,
        "layout": layout,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "CF-GFX942-AUTO-" + hashlib.sha256(canonical.encode()).hexdigest()[:12].upper()


def _normalise_shape(value: Any, dimensions: tuple[str, ...]) -> dict[str, Any]:
    if dimensions == ("shape",):
        raw = list(value) if isinstance(value, (list, tuple)) else [value]
        return {"dims": raw}
    values = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    if len(values) != len(dimensions):
        raise ValueError(f"expected {len(dimensions)} dimensions, got {value!r}")
    return dict(zip(dimensions, values, strict=True))


def _python_candidates(root: Path, spec: PythonSource) -> Iterable[dict[str, Any]]:
    source_path = root / spec.path
    source_sha = _file_sha256(source_path)
    evidence_sha = (
        _file_sha256(root / spec.architecture_evidence)
        if spec.architecture_evidence
        else None
    )
    lineage = f"aiter:{spec.path}:{spec.test_id}"
    for ordinal, (value, source_line) in enumerate(_python_values(root, spec)):
        shape = _normalise_shape(value, spec.dimensions)
        contract = {
            "operator": spec.operator,
            "shape": shape,
            "input_dtype": spec.dtype,
            "output_dtype": spec.dtype,
            "layout": spec.layout,
            **dict(spec.extra_contract),
        }
        identity = {
            "source_lineage_id": lineage,
            "source_record": value,
            "contract": contract,
        }
        architecture_guard = {
            "target": "gfx942",
            "evidence": (
                {
                    "kind": "gfx942_tuned_config",
                    "path": spec.architecture_evidence,
                    "sha256": evidence_sha,
                }
                if spec.architecture_evidence
                else {"kind": "target_constraint_only"}
            ),
        }
        yield {
            "id": _candidate_id(identity),
            "priority": spec.priority,
            "contract_family_id": _contract_family_id(
                lineage, spec.operator, spec.dtype, spec.layout
            ),
            "status": "extracted_candidate",
            "gpu_validated": False,
            "source_lineage_id": lineage,
            "source": {
                "revision": "aiter",
                "test_path": spec.path,
                "test_id": spec.test_id,
                "source_language": spec.source_language,
                "source_backend": spec.backend,
                "git_sha": LOCKED_AITER_SHA,
                "sha256": source_sha,
                "line": source_line,
                "record_ordinal": ordinal,
                "extraction": spec.extraction,
            },
            "lineage": {
                "kind": "pytest_parameter_record",
                "value_names": list(spec.value_names),
                "record": value,
                "no_cross_product": True,
            },
            "shape_cluster": _shape_cluster(shape),
            "contract": contract,
            "oracle": {"tier": "pure_torch", "method": spec.oracle_method},
            "architecture_guard": architecture_guard,
            "target_lanes": ["hip_gfx942", "triton_gfx942"],
        }


def _csv_candidates(root: Path) -> Iterable[dict[str, Any]]:
    relative = "aiter/configs/a8w8_tuned_gemm.csv"
    path = root / relative
    source_sha = _file_sha256(path)
    lineage = f"aiter:{relative}:gfx942-tuned-row"
    with path.open(newline="", encoding="utf-8") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            if row["gfx"] != "gfx942":
                continue
            shape = {name: int(row[name]) for name in ("M", "N", "K")}
            q_dtype = row["q_dtype_w"].removeprefix("torch.")
            contract = {
                "operator": "scaled_quant_gemm",
                "shape": shape,
                "input_dtype": q_dtype,
                "weight_dtype": q_dtype,
                "accum_dtype": "int32" if q_dtype == "int8" else "fp32",
                "output_dtype": "bf16",
                "layout": "TN",
                "scale": {"activation": "per_row", "weight": "per_column"},
            }
            identity = {
                "source_lineage_id": lineage,
                "source_record": {
                    "gfx": row["gfx"],
                    **shape,
                    "q_dtype_w": row["q_dtype_w"],
                    "kernelId": row["kernelId"],
                    "splitK": row["splitK"],
                },
                "contract": contract,
            }
            yield {
                "id": _candidate_id(identity),
                "priority": "P0",
                "contract_family_id": _contract_family_id(
                    lineage, "scaled_quant_gemm", q_dtype, "TN"
                ),
                "status": "extracted_candidate",
                "gpu_validated": False,
                "source_lineage_id": lineage,
                "source": {
                    "revision": "aiter",
                    "test_path": relative,
                    "test_id": "gfx942-tuned-row",
                    "source_language": "hip",
                    "source_backend": "aiter_asm_tuned",
                    "git_sha": LOCKED_AITER_SHA,
                    "sha256": source_sha,
                    "line": line,
                    "record_ordinal": line - 2,
                    "extraction": "csv_row",
                },
                "lineage": {
                    "kind": "gfx942_tuned_csv_row",
                    "kernel_id": int(row["kernelId"]),
                    "kernel_name": row["kernelName"],
                    "split_k": int(row["splitK"]),
                    "no_cross_product": True,
                },
                "shape_cluster": _shape_cluster(shape),
                "contract": contract,
                "oracle": {
                    "tier": "pure_torch",
                    "method": "dequantize scales then torch.nn.functional.linear",
                },
                "architecture_guard": {
                    "target": "gfx942",
                    "evidence": {
                        "kind": "csv_gfx_column",
                        "path": relative,
                        "line": line,
                        "value": "gfx942",
                    },
                },
                "target_lanes": ["hip_gfx942", "triton_gfx942"],
            }


def generate_document(aiter_root: Path = DEFAULT_AITER_ROOT) -> dict[str, Any]:
    actual_sha = _git_sha(aiter_root)
    if actual_sha != LOCKED_AITER_SHA:
        raise ValueError(
            f"AITER revision mismatch: expected {LOCKED_AITER_SHA}, got {actual_sha}"
        )
    candidates = list(_csv_candidates(aiter_root))
    for spec in PYTHON_SOURCES:
        candidates.extend(_python_candidates(aiter_root, spec))
    candidates.sort(key=lambda candidate: candidate["id"])
    ids = [candidate["id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("generated candidate IDs are not unique")
    return {
        "schema_version": "geak_phase1_gfx942_candidate_expansion_v1",
        "source_revisions": {
            "aiter": {
                "repository": "https://github.com/ROCm/aiter.git",
                "branch": "main",
                "git_sha": LOCKED_AITER_SHA,
                "license": "MIT",
                "local_root": str(aiter_root),
            }
        },
        "policy": {
            "generated_candidates_are_gpu_validated": False,
            "cross_product_allowed": False,
            "merge_preserves_existing_candidates": True,
            "target_architecture": "gfx942",
            "priority_policy": {
                "P0": "direct gfx942 tuned rows and unfused baseline primitives",
                "P1": "fused or activation-bearing variants",
            },
        },
        "candidates": candidates,
    }


def merge_documents(existing: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    """Append new IDs while preserving every existing candidate byte-for-byte in memory."""
    result = dict(existing)
    existing_candidates = list(existing.get("candidates", []))
    by_id = {candidate["id"]: candidate for candidate in existing_candidates}
    if len(by_id) != len(existing_candidates):
        raise ValueError("existing document contains duplicate candidate IDs")
    additions = [
        candidate
        for candidate in generated["candidates"]
        if candidate["id"] not in by_id
    ]
    result["candidates"] = existing_candidates + additions
    result.setdefault("candidate_expansion_sources", {}).update(
        generated["source_revisions"]
    )
    return result


def write_document(document: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(document, sort_keys=False, width=1000),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter-root", type=Path, default=DEFAULT_AITER_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge", type=Path)
    args = parser.parse_args(argv)
    generated = generate_document(args.aiter_root)
    if args.merge:
        existing = yaml.safe_load(args.merge.read_text(encoding="utf-8"))
        generated = merge_documents(existing, generated)
    write_document(generated, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
