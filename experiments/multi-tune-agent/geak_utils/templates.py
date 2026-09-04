"""Immutable registry for bundled dense-GEMM task templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


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
