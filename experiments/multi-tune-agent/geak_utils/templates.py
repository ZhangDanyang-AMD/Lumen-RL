"""Immutable registry for bundled dense-GEMM task templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class GemmTemplate:
    """Descriptor for one canonical GEMM format and its GEAK contract."""

    format: str
    template_dir: str
    case_type: str
    supported_architectures: tuple[str, ...]
    input_contract: str
    scale_contract: str
    output_contract: str
    backend: str
    input_scale_granularity: str | None = None
    weight_scale_granularity: str | None = None
    block_size: int | None = None

    def supports(self, architecture: str) -> bool:
        return architecture in self.supported_architectures


_TEMPLATES = {
    "fp16": GemmTemplate(
        format="fp16",
        template_dir="gemm",
        case_type="gemm",
        supported_architectures=("gfx942", "gfx950"),
        input_contract="A[M,K] FP16, B[K,N] FP16",
        scale_contract="none",
        output_contract="C[M,N] FP16",
        backend="triton",
        input_scale_granularity=None,
        weight_scale_granularity=None,
        block_size=None,
    ),
    "fp8": GemmTemplate(
        format="fp8",
        template_dir="gemm_fp8",
        case_type="scaled_quant_gemm",
        supported_architectures=("gfx942",),
        input_contract="A[M,K] and B[K,N] float8_e4m3fnuz",
        scale_contract="activation FP32 [M] per-token; weight FP32 [N] per-output-channel",
        output_contract="C[M,N] FP16",
        backend="triton",
        input_scale_granularity="per_token",
        weight_scale_granularity="per_channel",
        block_size=None,
    ),
    "mxfp4": GemmTemplate(
        format="mxfp4",
        template_dir="gemm_mxfp4",
        case_type="quant_fp4_mxfp",
        supported_architectures=("gfx950",),
        input_contract="packed E2M1 A[M,K/2] and W[N,K/2], logical A @ W.T",
        scale_contract="E8M0 block scales A[M,K/32] and W[N,K/32]",
        output_contract="C[M,N] BF16",
        backend="aiter",
        input_scale_granularity="per_block",
        weight_scale_granularity="per_block",
        block_size=32,
    ),
}

GEMM_TEMPLATES: Mapping[str, GemmTemplate] = MappingProxyType(_TEMPLATES)

FORMAT_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "fp8": "fp8",
        "float8": "fp8",
        "e4m3": "fp8",
        "e4m3fnuz": "fp8",
        "a8w8": "fp8",
        "mxfp4": "mxfp4",
        "mx-fp4": "mxfp4",
        "fp4": "mxfp4",
        "e2m1": "mxfp4",
    }
)


def normalize_gemm_format(value: object, *, default: str = "fp16") -> str:
    """Normalize a user/model format alias to a registered format."""

    raw = str(value or default).strip().lower()
    try:
        return FORMAT_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            "unsupported GEMM format %r; supported formats: fp16, fp8, mxfp4" % raw
        ) from exc


def architecture_for_target(target: object) -> str:
    """Map supported AMD product names and architecture strings to a GFX ISA."""

    raw = re.sub(r"[\s_-]+", "", str(target or "")).upper()
    if raw in {"GFX942", "MI300", "MI300X", "MI308", "MI308X"}:
        return "gfx942"
    if raw in {"GFX950", "MI350", "MI350X", "MI355", "MI355X"}:
        return "gfx950"
    if raw.startswith("GFX"):
        raise ValueError(
            "unsupported known architecture %r; supported architectures: gfx942, gfx950"
            % str(target)
        )
    if raw.startswith("MI") and re.fullmatch(r"MI\d{3,4}X?", raw):
        raise ValueError(
            "unsupported known AMD Instinct target %r; supported families: "
            "MI300/MI308 and MI350/MI355" % str(target)
        )
    raise ValueError(
        "unknown GPU target %r; specify MI300, MI308, MI350, MI355, gfx942, or gfx950"
        % str(target)
    )


def get_gemm_template(format_name: object) -> GemmTemplate:
    """Return the immutable descriptor for a format alias."""

    return GEMM_TEMPLATES[normalize_gemm_format(format_name)]


def validate_template_target(format_name: object, target: object) -> tuple[GemmTemplate, str]:
    """Resolve and validate one format/target combination."""

    descriptor = get_gemm_template(format_name)
    architecture = architecture_for_target(target)
    if not descriptor.supports(architecture):
        raise ValueError(
            "%s GEMM is unsupported on %s (%s); supported architecture(s): %s"
            % (
                descriptor.format.upper(),
                target,
                architecture,
                ", ".join(descriptor.supported_architectures),
            )
        )
    return descriptor, architecture


def gemm_template_matches_contract(
    template: GemmTemplate | object, contract: Mapping[str, Any]
) -> bool:
    """Return whether recognized fields exactly describe a canonical GEMM template."""

    if not isinstance(contract, Mapping):
        return False
    descriptor = (
        template if isinstance(template, GemmTemplate) else get_gemm_template(template)
    )
    operator = str(contract.get("operator") or "").strip().lower().replace("-", "_")
    if operator not in {"gemm", "dense_gemm"}:
        return False
    raw_format = contract.get("format") or contract.get("input_dtype")
    if raw_format is None:
        return False
    try:
        if normalize_gemm_format(raw_format) != descriptor.format:
            return False
    except ValueError:
        return False
    expected_output = "bf16" if descriptor.format == "mxfp4" else "fp16"
    expected_input = descriptor.format
    for field in ("input_dtype", "weight_dtype"):
        value = contract.get(field)
        if value and _normalized_dtype(value) != expected_input:
            return False
    output_dtype = contract.get("output_dtype")
    if output_dtype and _normalized_dtype(output_dtype) != expected_output:
        return False
    explicit_fields = set(contract.get("explicit_fields") or ())
    language = _normalized_optional(contract.get("language"))
    if "language" in explicit_fields and language != descriptor.backend:
        return False
    return (
        _normalized_optional(contract.get("input_scale_granularity"))
        == descriptor.input_scale_granularity
        and _normalized_optional(contract.get("weight_scale_granularity"))
        == descriptor.weight_scale_granularity
        and _normalized_block_size(contract.get("block_size")) == descriptor.block_size
    )


def canonical_gemm_template_for_contract(
    contract: Mapping[str, Any],
) -> GemmTemplate | None:
    """Find the one canonical GEMM template exactly matching a recognized contract."""

    return next(
        (
            descriptor
            for descriptor in GEMM_TEMPLATES.values()
            if gemm_template_matches_contract(descriptor, contract)
        ),
        None,
    )


def _normalized_optional(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = re.sub(r"[\s-]+", "_", str(value).strip().lower())
    if normalized in {"token", "channel", "block", "tensor", "group"}:
        normalized = "per_" + normalized
    return normalized


def _normalized_block_size(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _normalized_dtype(value: object) -> str:
    normalized = _normalized_optional(value) or ""
    aliases = {
        "float16": "fp16",
        "half": "fp16",
        "float8": "fp8",
        "e4m3": "fp8",
        "e4m3fnuz": "fp8",
        "fp4": "mxfp4",
        "mx_fp4": "mxfp4",
        "bfloat16": "bf16",
    }
    return aliases.get(normalized, normalized)
