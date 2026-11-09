"""Conservative, operator-specific reuse of independently trusted templates.

Reuse produces an untrusted draft.  A draft is promoted only through the same
static and GPU compile/correctness/performance gate used by template bootstrap.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from geak_utils.template_validation import validate_generated_template

from .template_bootstrap import (
    KernelContract,
    TemplateDraft,
    promote_validated_template,
    run_template_gpu_gate,
)


TRAIN_ADAPTABLE_OPERATORS = frozenset(
    {
        "rms_norm",
        "dynamic_per_token_quant",
        "dynamic_per_tensor_quant",
        "fused_silu_mul",
        "gemm",
    }
)
ADAPTABLE_OPERATORS = TRAIN_ADAPTABLE_OPERATORS | {
    "static_per_tensor_quant",
    "softmax",
}
CANONICAL_FILES = (
    "kernel.py",
    "config.yaml",
    "scripts/task_runner.py",
    "metadata.json",
)
TRUST_FLAGS = (
    "trusted",
    "static_valid",
    "compiled",
    "correct",
    "performance_valid",
)
TRUST_COMMANDS = ("compile", "correctness", "performance")


class ReuseError(ValueError):
    """A request or template cannot be adapted without changing semantics."""


@dataclass(frozen=True)
class TrustedTemplate:
    task: Mapping[str, Any]
    path: Path
    contract: Mapping[str, Any]
    shape: tuple[int, ...]
    semantics: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class ReusePlanItem:
    request: Mapping[str, Any]
    category: str
    template: TrustedTemplate | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ReusePlan:
    items: tuple[ReusePlanItem, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {
            name: sum(item.category == name for item in self.items)
            for name in ("exact", "parameterizable", "unsupported")
        }


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ReuseError(f"{label} root must be a mapping")
    return value


def _frozen_contract(text: str) -> dict[str, Any]:
    marker = "Frozen contract JSON:"
    start = text.find(marker)
    if start < 0:
        raise ReuseError("request has no Frozen contract JSON")
    start = text.find("{", start + len(marker))
    if start < 0:
        raise ReuseError("request has malformed Frozen contract JSON")
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ReuseError("request has invalid Frozen contract JSON") from exc
    if not isinstance(value, dict):
        raise ReuseError("frozen contract must be an object")
    return value


def _shape(contract: Mapping[str, Any]) -> tuple[int, ...]:
    raw = contract.get("shape")
    if not isinstance(raw, Mapping):
        raise ReuseError("contract shape must be a mapping")
    if isinstance(raw.get("dims"), list):
        values = raw["dims"]
    elif "M" in raw and "N" in raw:
        values = [raw["M"], raw["N"]]
        if "K" in raw:
            values.append(raw["K"])
        if "B" in raw:
            values.insert(0, raw["B"])
    elif "rows" in raw and "last_dim" in raw:
        values = [raw["rows"], raw["last_dim"]]
    else:
        values = list(raw.values())
    if not values or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in values
    ):
        raise ReuseError("contract shape must contain positive integers")
    return tuple(values)


def _dtype(contract: Mapping[str, Any], name: str) -> Any:
    nested = contract.get("dtype")
    if isinstance(nested, Mapping) and name in nested:
        return nested[name]
    return contract.get(f"{name}_dtype")


def _semantic_mapping(
    contract: Mapping[str, Any], *, language: str, architecture: str
) -> dict[str, Any]:
    operator = str(contract.get("operator") or "")
    result = {
        "operator": operator,
        "language": language,
        "architecture": architecture,
        "layout": contract.get("layout"),
        "input_dtype": _dtype(contract, "input"),
        "weight_dtype": _dtype(contract, "weight"),
        "accum_dtype": _dtype(contract, "accum"),
        "output_dtype": _dtype(contract, "output"),
        "scale": (
            contract.get("dtype", {}).get("scale")
            if isinstance(contract.get("dtype"), Mapping)
            else contract.get("scale")
        ),
        "epsilon": contract.get("epsilon"),
        "axis": contract.get("axis"),
    }
    if operator == "rms_norm":
        result["epsilon"] = (
            float(result["epsilon"]) if result["epsilon"] is not None else 1.0e-5
        )
        result["weight_dtype"] = result["weight_dtype"] or result["input_dtype"]
        result["accum_dtype"] = result["accum_dtype"] or "fp32"
    return result


def _freeze_semantics(values: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(values.items()))


def _request_contract(request: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    recognized = request.get("recognized_contract")
    if not isinstance(recognized, Mapping):
        raise ReuseError("request is missing recognized_contract")
    contract = recognized.get("contract")
    if not isinstance(contract, Mapping):
        contract = _frozen_contract(str(request.get("request") or ""))
    language = str(recognized.get("language") or "").strip().lower()
    architecture = str(recognized.get("target_gpu") or "").strip().lower()
    if not language or not architecture:
        raise ReuseError("recognized contract requires language and target_gpu")
    return dict(contract), language, architecture


def _trusted_path_and_metadata(
    task: Mapping[str, Any],
    catalog: Path,
    *,
    allowed_source_splits: frozenset[str] = frozenset({"train"}),
) -> tuple[Path, Mapping[str, Any]]:
    case_id = str(task.get("id") or "")
    raw_path = Path(str(task.get("kernel_path") or "")).expanduser()
    path = (catalog.parent / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
    if not case_id or not path.is_dir() or path.is_symlink():
        raise ReuseError(f"trusted task {case_id or '<unknown>'} has invalid path")
    for relative in CANONICAL_FILES:
        candidate = path / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ReuseError(f"trusted task {case_id} is missing {relative}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    trust = metadata.get("trust") if isinstance(metadata, Mapping) else None
    if not isinstance(trust, Mapping) or any(trust.get(flag) is not True for flag in TRUST_FLAGS):
        raise ReuseError(f"trusted task {case_id} lacks complete gate evidence")
    commands = trust.get("commands")
    if not isinstance(commands, Mapping):
        raise ReuseError(f"trusted task {case_id} lacks gate command evidence")
    for mode in TRUST_COMMANDS:
        result = commands.get(mode)
        if (
            not isinstance(result, Mapping)
            or result.get("ok") is not True
            or result.get("returncode") != 0
            or result.get("timed_out") is not False
        ):
            raise ReuseError(f"trusted task {case_id} has invalid {mode} evidence")
    contract_hash = str(task.get("contract_hash") or "")
    if not contract_hash or metadata.get("contract_hash") != contract_hash:
        raise ReuseError(f"trusted task {case_id} has a contract hash mismatch")
    report = validate_generated_template(path)
    if not report.valid:
        raise ReuseError(f"trusted task {case_id} no longer passes static validation")
    provenance = task.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("contract_hash") != contract_hash
        or not isinstance(provenance.get("case_seed"), Mapping)
        or provenance["case_seed"].get("split_group") not in allowed_source_splits
    ):
        raise ReuseError(
            f"trusted task {case_id} has forbidden source split provenance"
        )
    return path, metadata


def _validate_trusted_source(
    task: Mapping[str, Any],
    catalog: Path,
    *,
    allowed_source_splits: frozenset[str] = frozenset({"train"}),
) -> TrustedTemplate:
    path, _ = _trusted_path_and_metadata(
        task, catalog, allowed_source_splits=allowed_source_splits
    )
    source_contract = _frozen_contract(str(task.get("direction") or ""))
    language = str(task.get("backend") or "").strip().lower()
    architecture = str(task.get("architecture") or "").strip().lower()
    semantics = _freeze_semantics(
        _semantic_mapping(source_contract, language=language, architecture=architecture)
    )
    return TrustedTemplate(task, path, source_contract, _shape(source_contract), semantics)


def load_trusted_templates(
    catalog: Path,
    *,
    allowed_source_splits: frozenset[str] = frozenset({"train"}),
) -> tuple[TrustedTemplate, ...]:
    payload = _load_yaml_mapping(catalog, "production catalog")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ReuseError("production catalog requires a tasks list")
    return tuple(
        _validate_trusted_source(
            task, catalog, allowed_source_splits=allowed_source_splits
        )
        for task in tasks
        if isinstance(task, Mapping)
        and str(task.get("operator") or "") in ADAPTABLE_OPERATORS
    )


def load_generation_requests(
    path: Path, *, split_group: str = "train"
) -> tuple[dict[str, Any], ...]:
    if split_group not in {"train", "dev"}:
        raise ReuseError("reuse requests must select train or dev")
    payload = _load_yaml_mapping(path, "generation requests")
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise ReuseError("generation requests requires a requests list")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in requests:
        if not isinstance(raw, Mapping):
            raise ReuseError("every request must be a mapping")
        item = dict(raw)
        request_id = str(item.get("id") or "")
        provenance = item.get("seed_provenance")
        if not request_id or request_id in seen:
            raise ReuseError(f"invalid or duplicate request ID: {request_id!r}")
        seen.add(request_id)
        if not isinstance(provenance, Mapping):
            raise ReuseError(f"request {request_id} lacks seed provenance")
        split = provenance.get("split_group")
        if split == "held_out":
            raise ReuseError(f"held_out request is forbidden: {request_id}")
        if split == split_group:
            selected.append(item)
    return tuple(sorted(selected, key=lambda item: str(item["id"])))


def load_train_requests(path: Path) -> tuple[dict[str, Any], ...]:
    """Compatibility wrapper for the original Train-only reuse path."""
    return load_generation_requests(path, split_group="train")


def _distance(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right):
        return 1000.0 + abs(len(left) - len(right))
    return sum(abs(math.log2(a / b)) for a, b in zip(left, right))


def build_reuse_plan(
    catalog: Path,
    requests_path: Path,
    *,
    request_split: str = "train",
    template_catalogs: Sequence[Path] = (),
) -> ReusePlan:
    allowed_source_splits = (
        frozenset({"train", "dev"})
        if request_split == "dev"
        else frozenset({"train"})
    )
    adaptable_operators = (
        ADAPTABLE_OPERATORS
        if request_split == "dev"
        else TRAIN_ADAPTABLE_OPERATORS
    )
    catalogs = (catalog, *template_catalogs)
    templates = tuple(
        template
        for source_catalog in catalogs
        for template in load_trusted_templates(
            source_catalog, allowed_source_splits=allowed_source_splits
        )
    )
    requests = load_generation_requests(requests_path, split_group=request_split)
    catalog_payload = _load_yaml_mapping(catalog, "production catalog")
    exact_ids: set[str] = set()
    for task in catalog_payload.get("tasks", []):
        if isinstance(task, Mapping) and task.get("id"):
            _trusted_path_and_metadata(
                task, catalog, allowed_source_splits=allowed_source_splits
            )
            exact_ids.add(str(task["id"]))
    items: list[ReusePlanItem] = []
    for request in requests:
        request_id = str(request["id"])
        if request_id in exact_ids:
            items.append(ReusePlanItem(request, "exact"))
            continue
        try:
            contract, language, architecture = _request_contract(request)
            operator = str(contract.get("operator") or "")
            semantics = _freeze_semantics(
                _semantic_mapping(
                    contract, language=language, architecture=architecture
                )
            )
            if operator not in adaptable_operators:
                raise ReuseError("operator has no explicit reuse adapter")
            compatible = [
                template for template in templates if template.semantics == semantics
            ]
            if not compatible:
                raise ReuseError("semantic/dtype/layout/epsilon mismatch")
            target_shape = _shape(contract)
            nearest = min(
                compatible,
                key=lambda template: (
                    _distance(template.shape, target_shape),
                    str(template.task["id"]),
                ),
            )
            items.append(ReusePlanItem(request, "parameterizable", nearest))
        except ReuseError as exc:
            items.append(ReusePlanItem(request, "unsupported", reason=str(exc)))
    return ReusePlan(tuple(items))


def _kernel_contract(request: Mapping[str, Any]) -> KernelContract:
    recognized = request["recognized_contract"]
    contract, language, architecture = _request_contract(request)
    shape = _shape(contract)
    operator = str(contract["operator"])
    input_dtype = _dtype(contract, "input") or recognized.get("input_dtype")
    weight_dtype = _dtype(contract, "weight") or recognized.get("weight_dtype")
    if operator == "gemm" and not weight_dtype:
        weight_dtype = input_dtype
    output_dtype = _dtype(contract, "output") or recognized.get("output_dtype")
    format_name = recognized.get("format") or input_dtype
    return KernelContract(
        operator=operator,
        request=str(request["request"]),
        target_gpu=str(recognized["target_gpu"]),
        architecture=architecture,
        language=language,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        output_dtype=output_dtype,
        input_format=format_name,
        weight_format=format_name if weight_dtype else None,
        input_scale_granularity=recognized.get("input_scale_granularity"),
        weight_scale_granularity=recognized.get("weight_scale_granularity"),
        block_size=recognized.get("block_size"),
        shapes=[shape],
    )


def _sequence_node(values: Sequence[int], template: ast.List | ast.Tuple) -> ast.expr:
    elements = [ast.Constant(value=value) for value in values]
    if isinstance(template, ast.List):
        return ast.copy_location(ast.List(elts=elements, ctx=ast.Load()), template)
    return ast.copy_location(ast.Tuple(elts=elements, ctx=ast.Load()), template)


class _ExplicitShapeAdapter(ast.NodeTransformer):
    def __init__(
        self,
        operator: str,
        replacements: Mapping[tuple[int, ...], tuple[int, ...]],
        dimensions: Mapping[str, int],
        source_dimensions: Mapping[str, int],
        epsilon: float | None,
    ) -> None:
        self.operator = operator
        self.replacements = dict(replacements)
        self.dimensions = dict(dimensions)
        self.source_dimensions = dict(source_dimensions)
        self.epsilon = epsilon
        self.replacement_count = 0

    def visit_List(self, node: ast.List) -> ast.AST:
        self.generic_visit(node)
        return self._replace_sequence(node)

    def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
        self.generic_visit(node)
        return self._replace_sequence(node)

    def _replace_sequence(self, node: ast.List | ast.Tuple) -> ast.AST:
        if all(isinstance(item, ast.Constant) and type(item.value) is int for item in node.elts):
            values = tuple(int(item.value) for item in node.elts)
            replacement = self.replacements.get(values)
            if replacement is not None and replacement != values:
                self.replacement_count += 1
                return _sequence_node(replacement, node)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        self.generic_visit(node)
        key = (node.arg or "").upper()
        if key in self.dimensions and isinstance(node.value, ast.Constant):
            if type(node.value.value) is int and node.value.value != self.dimensions[key]:
                node.value = ast.copy_location(ast.Constant(self.dimensions[key]), node.value)
                self.replacement_count += 1
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        function_name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else ""
        )
        if (
            self.operator == "rms_norm"
            and function_name in {"empty", "ones", "randn", "zeros"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and type(node.args[0].value) is int
            and node.args[0].value == self.source_dimensions.get("N")
        ):
            node.args[0] = ast.copy_location(
                ast.Constant(self.dimensions["N"]), node.args[0]
            )
            self.replacement_count += 1
        if len(node.args) >= 2:
            leading = node.args[:2]
            if all(
                isinstance(value, ast.Constant) and type(value.value) is int
                for value in leading
            ):
                pair = tuple(int(value.value) for value in leading)
                replacement = self.replacements.get(pair)
                if replacement is not None and len(replacement) == 2:
                    node.args[:2] = [
                        ast.copy_location(ast.Constant(value), old)
                        for value, old in zip(replacement, leading)
                    ]
                    self.replacement_count += 1
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if (
                self.epsilon is not None
                and name.lower() in {"eps", "epsilon"}
                and isinstance(node.value, ast.Constant)
            ):
                node.value = ast.copy_location(ast.Constant(self.epsilon), node.value)
                self.replacement_count += 1
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        if self.epsilon is not None and node.args.defaults:
            positional = node.args.args[-len(node.args.defaults) :]
            for index, argument in enumerate(positional):
                if argument.arg.lower() in {"eps", "epsilon"}:
                    node.args.defaults[index] = ast.copy_location(
                        ast.Constant(self.epsilon), node.args.defaults[index]
                    )
                    self.replacement_count += 1
        return node


def _adapter_replacements(
    operator: str, source: tuple[int, ...], target: tuple[int, ...]
) -> tuple[dict[tuple[int, ...], tuple[int, ...]], dict[str, int]]:
    if operator in {
        "rms_norm",
        "dynamic_per_token_quant",
        "dynamic_per_tensor_quant",
        "static_per_tensor_quant",
        "softmax",
    }:
        if len(source) != 2 or len(target) != 2:
            raise ReuseError(f"{operator} adapter requires a two-dimensional shape")
        sm, sn = source
        tm, tn = target
        return {
            (sm, sn): (tm, tn),
            (1, sn): (1, tn),
            (sn,): (tn,),
        }, {"M": tm, "N": tn}
    if operator == "gemm":
        if len(source) != 3 or len(target) != 3:
            raise ReuseError("gemm adapter requires M/N/K without batching")
        sm, sn, sk = source
        tm, tn, tk = target
        return {
            (sm, sn, sk): (tm, tn, tk),
            (sm, sk): (tm, tk),
            (sn, sk): (tn, tk),
            (sk, sn): (tk, tn),
            (sm, sn): (tm, tn),
        }, {"M": tm, "N": tn, "K": tk}
    if operator == "fused_silu_mul":
        if len(target) < 2 or target[-1] % 2:
            raise ReuseError("fused_silu_mul requires an even split-last dimension")
        output = (*target[:-1], target[-1] // 2)
        source_output = (*source[:-1], source[-1] // 2)
        return {source: target, source_output: output}, {}
    raise ReuseError(f"no explicit adapter for {operator}")


class _FusedSiluReferenceAdapter(ast.NodeTransformer):
    """Change the trusted 2-D reference slices to split the final dimension."""

    def __init__(self) -> None:
        self.replacement_count = 0

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "input_tensor"
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
            and isinstance(node.slice.elts[0], ast.Slice)
            and node.slice.elts[0].lower is None
            and node.slice.elts[0].upper is None
            and node.slice.elts[0].step is None
            and isinstance(node.slice.elts[1], ast.Slice)
        ):
            node.slice.elts[0] = ast.Constant(Ellipsis)
            self.replacement_count += 1
        return node


def _adapt_runner(
    source: str,
    *,
    operator: str,
    language: str | None = None,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    epsilon: float | None,
) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ReuseError("trusted runner is not valid Python") from exc
    replacements, dimensions = _adapter_replacements(
        operator, source_shape, target_shape
    )
    source_dimensions = {}
    if len(source_shape) >= 2:
        source_dimensions = {"M": source_shape[0], "N": source_shape[1]}
    if len(source_shape) == 3:
        source_dimensions["K"] = source_shape[2]
    adapter = _ExplicitShapeAdapter(
        operator, replacements, dimensions, source_dimensions, epsilon
    )
    tree = adapter.visit(tree)
    if (
        operator == "fused_silu_mul"
        and language == "hip"
        and len(target_shape) > 2
    ):
        reference_adapter = _FusedSiluReferenceAdapter()
        tree = reference_adapter.visit(tree)
        if reference_adapter.replacement_count != 2:
            raise ReuseError(
                "multidimensional fused_silu_mul runner lacks verified reference slices"
            )
    ast.fix_missing_locations(tree)
    if source_shape != target_shape and adapter.replacement_count == 0:
        raise ReuseError("runner exposes no recognized operator-specific shape sites")
    return ast.unparse(tree) + "\n"


def _replace_anchor(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ReuseError(f"{label} expected exactly one verified anchor, found {count}")
    return source.replace(old, new, 1)


def _adapt_hip_rms_norm_kernel(
    source: str, source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> str:
    sm, sn = source_shape
    tm, tn = target_shape
    source = _replace_anchor(
        source,
        f"input.size(0) == {sm} && input.size(1) == {sn}",
        f"input.size(0) == {tm} && input.size(1) == {tn}",
        "HIP rms_norm input shape",
    )
    source = _replace_anchor(
        source,
        f'"input shape must be {sm}x{sn}"',
        f'"input shape must be {tm}x{tn}"',
        "HIP rms_norm input assertion",
    )
    source = _replace_anchor(
        source,
        f"weight.size(0) == {sn}",
        f"weight.size(0) == {tn}",
        "HIP rms_norm weight shape",
    )
    source = _replace_anchor(
        source,
        f'"weight shape must be {sn}"',
        f'"weight shape must be {tn}"',
        "HIP rms_norm weight assertion",
    )
    return source


def _adapt_hip_fused_silu_kernel(
    source: str, source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> str:
    if len(source_shape) != 2 or len(target_shape) < 2:
        raise ReuseError("HIP fused_silu_mul requires split-last tensor shapes")
    sm, sn = source_shape
    rows = math.prod(target_shape[:-1])
    last_dim = target_shape[-1]
    source = _replace_anchor(
        source,
        f"input.size(0) == {sm} && input.size(1) == {sn}",
        f"input.size(0) == {rows} && input.size(1) == {last_dim}",
        "HIP fused_silu_mul flattened shape",
    )
    source = _replace_anchor(
        source,
        f'"input shape must be {sm}x{sn}"',
        f'"input shape must be {rows}x{last_dim}"',
        "HIP fused_silu_mul shape assertion",
    )
    wrapper = (
        "def fused_silu_mul(input_tensor: torch.Tensor) -> torch.Tensor:\n"
        "    return _module().fused_silu_mul(input_tensor)\n"
    )
    adapted_wrapper = (
        "def fused_silu_mul(input_tensor: torch.Tensor) -> torch.Tensor:\n"
        "    original_shape = input_tensor.shape\n"
        f"    matrix = input_tensor.reshape(-1, {last_dim})\n"
        "    output = _module().fused_silu_mul(matrix)\n"
        "    return output.reshape(*original_shape[:-1], original_shape[-1] // 2)\n"
    )
    return _replace_anchor(
        source, wrapper, adapted_wrapper, "HIP fused_silu_mul Python wrapper"
    )


def _adapt_hip_gemm_kernel(
    source: str, source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> str:
    sm, sn, sk = source_shape
    tm, tn, tk = target_shape
    for name, old, new in (("M", sm, tm), ("N", sn, tn), ("K", sk, tk)):
        source = _replace_anchor(
            source,
            f"constexpr int {name} = {old};",
            f"constexpr int {name} = {new};",
            f"HIP gemm {name} constant",
        )
    anchors = (
        (f"{{{sm}, {sk}}}", f"{{{tm}, {tk}}}", "HIP gemm A shape"),
        (f"{{{sn}, {sk}}}", f"{{{tn}, {tk}}}", "HIP gemm B shape"),
        (f"{{{sm}, {sn}}}", f"{{{tm}, {tn}}}", "HIP gemm output shape"),
        (f'"A must be {sm}x{sk}"', f'"A must be {tm}x{tk}"', "HIP gemm A assertion"),
        (f'"B must be {sn}x{sk}"', f'"B must be {tn}x{tk}"', "HIP gemm B assertion"),
    )
    for old, new, label in anchors:
        source = _replace_anchor(source, old, new, label)
    blocks = (tm * tn + 255) // 256
    return _replace_anchor(
        source,
        "gemm_kernel<<<16, 256, 0, stream>>>",
        f"gemm_kernel<<<{blocks}, 256, 0, stream>>>",
        "HIP gemm launch grid",
    )


def _adapt_hip_static_quant_kernel(
    source: str, source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> str:
    sm, sn = source_shape
    tm, tn = target_shape
    shape_anchor = f"input.sizes() == at::IntArrayRef({{{sm}, {sn}}})"
    if shape_anchor not in source:
        generic_anchors = (
            'input.dim() == 2, "input must be rank two"',
            "input.numel()",
            "input.sizes()",
        )
        if any(source.count(anchor) < 1 for anchor in generic_anchors):
            raise ReuseError(
                "HIP static_per_tensor_quant lacks verified dynamic-shape anchors"
            )
        return source
    source = _replace_anchor(
        source,
        shape_anchor,
        f"input.sizes() == at::IntArrayRef({{{tm}, {tn}}})",
        "HIP static_per_tensor_quant input shape",
    )
    return _replace_anchor(
        source,
        f'"input shape must be {sm}x{sn}"',
        f'"input shape must be {tm}x{tn}"',
        "HIP static_per_tensor_quant input assertion",
    )


def _adapt_kernel(
    source: str,
    *,
    operator: str,
    language: str,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> str:
    if language == "hip":
        if operator == "rms_norm":
            return _adapt_hip_rms_norm_kernel(source, source_shape, target_shape)
        if operator == "fused_silu_mul":
            return _adapt_hip_fused_silu_kernel(source, source_shape, target_shape)
        if operator == "gemm":
            return _adapt_hip_gemm_kernel(source, source_shape, target_shape)
        if operator == "static_per_tensor_quant":
            return _adapt_hip_static_quant_kernel(
                source, source_shape, target_shape
            )
        if operator == "dynamic_per_tensor_quant":
            anchor = "*scale = nextafterf(quotient, INFINITY);"
            return (
                _replace_anchor(
                    source,
                    anchor,
                    "*scale = quotient;",
                    "HIP dynamic_per_tensor_quant scale finalization",
                )
                if anchor in source
                else source
            )
        return source
    if language == "triton" and operator == "static_per_tensor_quant":
        rendered_shape = repr(tuple(source_shape))
        if rendered_shape in source or repr(list(source_shape)) in source:
            return _adapt_runner(
                source,
                operator=operator,
                language=language,
                source_shape=source_shape,
                target_shape=target_shape,
                epsilon=None,
            )
        generic_anchors = (
            "x.numel()",
            "torch.empty_like(x",
            "tl.program_id(0)",
        )
        if any(source.count(anchor) < 1 for anchor in generic_anchors):
            raise ReuseError(
                "Triton static_per_tensor_quant lacks verified dynamic-shape anchors"
            )
        return source
    if language == "triton" and operator == "dynamic_per_tensor_quant":
        adapted = _adapt_runner(
            source,
            operator=operator,
            language=language,
            source_shape=source_shape,
            target_shape=target_shape,
            epsilon=None,
        )
        adapted = _replace_anchor(
            adapted,
            "    rounded = tl.floor(unrounded).to(tl.int32)\n"
            "    rounded += (unrounded - tl.floor(unrounded) > 0.5).to(tl.int32)\n",
            "    rounded_floor = tl.floor(unrounded).to(tl.int32)\n"
            "    rounded_fraction = unrounded - tl.floor(unrounded)\n"
            "    rounded_increment = (rounded_fraction > 0.5) | "
            "((rounded_fraction == 0.5) & ((rounded_floor & 1) == 1))\n"
            "    rounded = rounded_floor + rounded_increment.to(tl.int32)\n",
            "Triton FP8 normal round-to-nearest-even",
        )
        return _replace_anchor(
            adapted,
            "    subnormal_bits += (subnormal_unrounded - subnormal_floor > 0.5).to(tl.int32)\n",
            "    subnormal_fraction = subnormal_unrounded - subnormal_floor\n"
            "    subnormal_increment = (subnormal_fraction > 0.5) | "
            "((subnormal_fraction == 0.5) & ((subnormal_bits & 1) == 1))\n"
            "    subnormal_bits += subnormal_increment.to(tl.int32)\n",
            "Triton FP8 subnormal round-to-nearest-even",
        )
    return source


def _atomic_install(directory: Path, files: Mapping[str, str]) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".tmp-{directory.name}-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(mode=0o700)
        for relative, text in files.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        if directory.exists():
            current = {
                relative: (directory / relative).read_text(encoding="utf-8")
                for relative in CANONICAL_FILES
            }
            if current == dict(files):
                return
            raise ReuseError(f"candidate already exists with different content: {directory}")
        os.replace(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def materialize_candidate(item: ReusePlanItem, candidate_root: Path) -> TemplateDraft:
    if item.category != "parameterizable" or item.template is None:
        raise ReuseError("only parameterizable plan items can be materialized")
    request = item.request
    template = item.template
    contract = _kernel_contract(request)
    frozen, _, _ = _request_contract(request)
    target_shape = _shape(frozen)
    epsilon = float(frozen["epsilon"]) if frozen.get("epsilon") is not None else None
    runner = _adapt_runner(
        (template.path / "scripts/task_runner.py").read_text(encoding="utf-8"),
        operator=contract.operator,
        language=contract.language,
        source_shape=template.shape,
        target_shape=target_shape,
        epsilon=epsilon,
    )
    kernel = _adapt_kernel(
        (template.path / "kernel.py").read_text(encoding="utf-8"),
        operator=contract.operator,
        language=contract.language,
        source_shape=template.shape,
        target_shape=target_shape,
    )
    config = yaml.safe_load((template.path / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ReuseError("trusted config must be a mapping")
    config["shapes"] = [list(target_shape)]
    metadata = dict(contract.metadata)
    metadata["provenance"] = {
        "generator": "multi_tune_agent.trusted_template_reuse",
        "generation_method": "trusted_template_parameterization",
        "source_request": str(request["request"]),
        "request_id": str(request["id"]),
        "contract_hash": contract.contract_hash,
        "case_seed": dict(request["seed_provenance"]),
        "template_source": {
            "case_id": str(template.task["id"]),
            "contract_hash": str(template.task["contract_hash"]),
            "kernel_sha256": hashlib.sha256(
                (template.path / "kernel.py").read_bytes()
            ).hexdigest(),
        },
        "adapter": contract.operator,
    }
    metadata["reuse_status"] = "untrusted_pending_gate"
    candidate_id = hashlib.sha256(
        f"{contract.contract_hash}:{request['id']}".encode()
    ).hexdigest()
    destination = candidate_root.expanduser().resolve() / candidate_id
    files = {
        "kernel.py": kernel,
        "config.yaml": yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        "scripts/task_runner.py": runner,
        "metadata.json": json.dumps(metadata, sort_keys=True, indent=2) + "\n",
    }
    _atomic_install(destination, files)
    report = validate_generated_template(destination, contract.expected_contract)
    if not report.valid:
        raise ReuseError(
            "adapted candidate failed static validation: "
            + "; ".join(str(issue) for issue in report.errors)
        )
    return TemplateDraft(
        destination,
        contract,
        report,
        "trusted_template_parameterization",
    )


def _task_record(
    item: ReusePlanItem, draft: TemplateDraft, promoted: Path
) -> dict[str, Any]:
    metadata = json.loads((promoted / "metadata.json").read_text(encoding="utf-8"))
    recognized = item.request["recognized_contract"]
    return {
        "id": str(item.request["id"]),
        "type": "aiter_generated",
        "kernel_path": str(promoted),
        "direction": str(item.request["request"]),
        "operator": draft.contract.operator,
        "backend": draft.contract.language,
        "architecture": draft.contract.architecture,
        "contract_hash": draft.contract_hash,
        "provenance": {
            **dict(metadata["provenance"]),
            "contract_hash": draft.contract_hash,
            "case_seed": dict(item.request["seed_provenance"]),
        },
        "recognized_contract": dict(recognized),
    }


def _atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            yaml.safe_dump(dict(payload), temporary, sort_keys=False, allow_unicode=True)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def merge_promoted_tasks(
    source_catalog: Path, output_catalog: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    lock_path = output_catalog.with_suffix(output_catalog.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        source = _load_yaml_mapping(source_catalog, "production catalog")
        current = (
            _load_yaml_mapping(output_catalog, "output catalog")
            if output_catalog.is_file()
            else {key: value for key, value in source.items() if key != "tasks"}
        )
        tasks: dict[str, dict[str, Any]] = {}
        for raw in source.get("tasks", []):
            if isinstance(raw, Mapping):
                tasks[str(raw["id"])] = dict(raw)
        for raw in current.get("tasks", []):
            if isinstance(raw, Mapping):
                task_id = str(raw["id"])
                if task_id in tasks and tasks[task_id] != dict(raw):
                    raise ReuseError(f"output conflicts with exact task {task_id}")
                tasks[task_id] = dict(raw)
        for raw in records:
            record = dict(raw)
            task_id = str(record["id"])
            if task_id in tasks and tasks[task_id] != record:
                raise ReuseError(f"promotion would replace existing task {task_id}")
            metadata = json.loads(
                (Path(record["kernel_path"]) / "metadata.json").read_text(encoding="utf-8")
            )
            trust = metadata.get("trust")
            if not isinstance(trust, Mapping) or trust.get("trusted") is not True:
                raise ReuseError(f"refusing to catalog untrusted task {task_id}")
            tasks[task_id] = record
        current["tasks"] = [tasks[key] for key in sorted(tasks)]
        _atomic_write_yaml(output_catalog, current)


def _selected_items(
    plan: ReusePlan, *, shard_index: int, shard_count: int
) -> list[ReusePlanItem]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ReuseError("shard index must be in [0, shard count)")
    candidates = [
        item for item in plan.items if item.category == "parameterizable"
    ]
    return [
        item for index, item in enumerate(candidates) if index % shard_count == shard_index
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "materialize", "gate"))
    parser.add_argument("--production-catalog", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument(
        "--request-split", choices=("train", "dev"), default="train"
    )
    parser.add_argument(
        "--template-catalog",
        action="append",
        default=[],
        type=Path,
        help="additional trusted implementation scaffold catalog; repeatable",
    )
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--verified-root", type=Path)
    parser.add_argument("--output-catalog", type=Path)
    parser.add_argument(
        "--merge-catalog",
        type=Path,
        help="atomically merge trusted promotions into this catalog after writing output",
    )
    parser.add_argument("--geak-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--command-timeout", type=int, default=300)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="limit materialize/gate to an exact parameterizable request ID; repeatable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = args.production_catalog.expanduser().resolve()
    requests = args.requests.expanduser().resolve()
    template_catalogs = tuple(
        path.expanduser().resolve() for path in args.template_catalog
    )
    plan = build_reuse_plan(
        catalog,
        requests,
        request_split=args.request_split,
        template_catalogs=template_catalogs,
    )
    summary: dict[str, Any] = {
        **plan.counts,
        "requests": len(plan.items),
        "request_split": args.request_split,
    }
    if args.mode == "plan":
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.candidate_root is None:
        raise ReuseError("materialize and gate require --candidate-root")
    selected = _selected_items(
        plan, shard_index=args.shard_index, shard_count=args.shard_count
    )
    if args.case_id:
        requested_ids = set(args.case_id)
        known_ids = {
            str(item.request["id"])
            for item in plan.items
            if item.category == "parameterizable"
        }
        unknown = sorted(requested_ids - known_ids)
        if unknown:
            raise ReuseError(
                "case selector is not parameterizable: " + ", ".join(unknown)
            )
        selected = [
            item for item in selected if str(item.request["id"]) in requested_ids
        ]
    existing_ids: set[str] = set()
    if args.mode == "gate" and args.output_catalog is not None:
        output_path = args.output_catalog.expanduser().resolve()
        if output_path.is_file():
            output_payload = _load_yaml_mapping(output_path, "output catalog")
            existing_ids = {
                str(task.get("id") or "")
                for task in output_payload.get("tasks", [])
                if isinstance(task, Mapping)
            }
        selected = [
            item for item in selected if str(item.request["id"]) not in existing_ids
        ]
    drafts = [
        (item, materialize_candidate(item, args.candidate_root))
        for item in selected
    ]
    summary["selected"] = len(drafts)
    summary["materialized"] = len(drafts)
    summary["shard_index"] = args.shard_index
    summary["shard_count"] = args.shard_count
    if args.mode == "materialize":
        print(json.dumps(summary, sort_keys=True))
        return 0
    required = {
        "--verified-root": args.verified_root,
        "--output-catalog": args.output_catalog,
        "--geak-root": args.geak_root,
        "--run-root": args.run_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ReuseError("gate requires " + ", ".join(missing))
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    diagnostics_path = args.run_root.expanduser().resolve() / "gate-results.jsonl"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    for item, draft in drafts:
        result = run_template_gpu_gate(
            draft,
            geak_root=args.geak_root,
            run_root=args.run_root,
            gpu_ids=args.gpu_id,
            command_timeout=args.command_timeout,
        )
        diagnostic = {
            "case_id": str(item.request["id"]),
            "contract_hash": draft.contract_hash,
            "trusted": result.trusted,
            "compiled": result.compiled,
            "correct": result.correct,
            "performance_valid": result.performance_valid,
            "errors": list(result.errors),
            "commands": {
                mode: dict(summary)
                for mode, summary in result.command_summaries.items()
            },
            "validation_workspace": (
                str(result.validation_workspace)
                if result.validation_workspace is not None
                else None
            ),
        }
        with diagnostics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(diagnostic, sort_keys=True) + "\n")
        if not result.trusted:
            failures.append(str(item.request["id"]))
            continue
        promoted = promote_validated_template(draft, result, args.verified_root)
        records.append(_task_record(item, draft, promoted))
    merge_promoted_tasks(catalog, args.output_catalog, records)
    if args.merge_catalog is not None:
        merge_promoted_tasks(
            catalog, args.merge_catalog.expanduser().resolve(), records
        )
    summary["promoted"] = len(records)
    summary["failed_gate"] = failures
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
