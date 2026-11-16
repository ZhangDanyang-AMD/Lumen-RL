"""Input element types for the MICake worker beyond C144's FP16 harness.

Runs inside MICake's Python (no torch). For ``bf16`` and ``fp8`` (OCP e4m3,
Triton ``fp8e4nv``) the C144 kernel is emitted unchanged except for the load
cast, compiled with the matching pointer signature, and fed the C144 affine
input pattern rounded to the element type. The oracle is exact for the rounded
inputs: A depends only on (row % 17, k % 23) and B on (k % 29, column % 19), so
every output equals one of 17 x 19 table entries.
"""

from __future__ import annotations

import bisect
import contextlib
import math
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterator

A_ROW_PERIOD, A_K_PERIOD, B_K_PERIOD, B_COL_PERIOD = 17, 23, 29, 19


@dataclass(frozen=True)
class ElementType:
    name: str
    triton: str
    pointer: str
    element_bytes: int
    drop_ieee: bool


DTYPES = {
    "fp16": ElementType("fp16", "tl.float16", "*fp16", 2, False),
    "bf16": ElementType("bf16", "tl.bfloat16", "*bf16", 2, False),
    "fp8": ElementType("fp8", "tl.float8e4nv", "*fp8e4nv", 1, True),
}


def _e4m3_value(code: int) -> float:
    exponent, mantissa = (code >> 3) & 0xF, code & 0x7
    if exponent == 0:
        return mantissa * 2.0 ** -9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


_E4M3_CODES = list(range(0x7F))
_E4M3_VALUES = [_e4m3_value(code) for code in _E4M3_CODES]


def encode_e4m3(value: float) -> int:
    """Round-to-nearest-even OCP e4m3 code, saturating at 448."""
    sign = 0x80 if value < 0 or (value == 0 and math.copysign(1.0, value) < 0) else 0
    magnitude = min(abs(value), _E4M3_VALUES[-1])
    index = bisect.bisect_left(_E4M3_VALUES, magnitude)
    if index == len(_E4M3_VALUES):
        index -= 1
    elif index > 0:
        low, high = _E4M3_VALUES[index - 1], _E4M3_VALUES[index]
        if magnitude - low < high - magnitude or (magnitude - low == high - magnitude and (index - 1) % 2 == 0):
            index -= 1
    return sign | _E4M3_CODES[index]


def decode_e4m3(code: int) -> float:
    value = _E4M3_VALUES[code & 0x7F]
    return -value if code & 0x80 else value


def encode_bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16) & 0xFFFF


def decode_bf16(code: int) -> float:
    return struct.unpack("<f", struct.pack("<I", code << 16))[0]


def _codec(dtype: ElementType) -> tuple[Callable[[float], int], Callable[[int], float], str]:
    if dtype.name == "fp8":
        return encode_e4m3, decode_e4m3, "B"
    return encode_bf16, decode_bf16, "H"


def emit_source(fp16_source: str, dtype: ElementType) -> str:
    source = fp16_source.replace(".to(tl.float16)", f".to({dtype.triton})")
    if dtype.drop_ieee:
        source = source.replace('            input_precision="ieee",\n', "")
    return source


def compiler_for(dtype: ElementType, signature: dict[str, str]) -> Callable[[Any, Any], Any]:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    typed = {**signature, "a": dtype.pointer, "b": dtype.pointer}

    def compile_typed(function: Any, options: Any) -> Any:
        source = ASTSource(function, signature=typed, constexprs={})
        return triton.compile(source, target=GPUTarget("hip", "gfx950", 64), options=dict(options))

    return compile_typed


class TypedHarness:
    """``deterministic_fp16_inputs`` / ``correctness_check`` replacements for one element type."""

    def __init__(self, gpu: Any, dtype: ElementType) -> None:
        self.gpu = gpu
        self.dtype = dtype
        self.encode, self.decode, self.code_format = _codec(dtype)
        self.qa = [[self.decode(self.encode(gpu._alpha(r) + gpu._u(i))) for i in range(A_K_PERIOD)] for r in range(A_ROW_PERIOD)]
        self.qb = [[self.decode(self.encode(gpu._v(i) + gpu._beta(c))) for c in range(B_COL_PERIOD)] for i in range(B_K_PERIOD)]
        self.tables: dict[int, list[list[float]]] = {}

    def inputs(self, workload: Any) -> dict[str, Any]:
        shape = workload if type(workload) is self.gpu.RuntimeWorkload else self.gpu._normalize_workload(workload, 0)
        pack = struct.Struct("<" + self.code_format * shape.k).pack
        a_rows = [pack(*(self.encode(self.gpu._alpha(r) + self.gpu._u(i)) for i in range(shape.k))) for r in range(A_ROW_PERIOD)]
        pack_n = struct.Struct("<" + self.code_format * shape.n).pack
        b_rows = [pack_n(*(self.encode(self.gpu._v(i) + self.gpu._beta(c)) for c in range(shape.n))) for i in range(B_K_PERIOD)]
        a_raw = b"".join(a_rows[r % A_ROW_PERIOD] for r in range(shape.m))
        b_raw = b"".join(b_rows[i % B_K_PERIOD] for i in range(shape.k))
        return {"profile": f"LUMENRL_MICAKE_{self.dtype.name.upper()}_AFFINE_ROUNDED", "workload": shape.to_dict(),
                "a": a_raw, "b": b_raw, "a_sha256": self.gpu._sha(a_raw), "b_sha256": self.gpu._sha(b_raw)}

    def _table(self, k_extent: int) -> list[list[float]]:
        if k_extent not in self.tables:
            pairs = Counter((i % A_K_PERIOD, i % B_K_PERIOD) for i in range(k_extent))
            self.tables[k_extent] = [
                [math.fsum(count * self.qa[r][p] * self.qb[q][c] for (p, q), count in pairs.items()) for c in range(B_COL_PERIOD)]
                for r in range(A_ROW_PERIOD)
            ]
        return self.tables[k_extent]

    def check(self, workload: Any, output_fp32_le: bytes, *, abs_tolerance: float, rel_tolerance: float) -> dict[str, Any]:
        shape = workload if type(workload) is self.gpu.RuntimeWorkload else self.gpu._normalize_workload(workload, 0)
        if type(output_fp32_le) is not bytes or len(output_fp32_le) != shape.m * shape.n * 4:
            raise self.gpu.GpuRuntimeRefused("C144_GPU_OUTPUT_SIZE")
        table = self._table(shape.k)
        values = struct.iter_unpack("<f", output_fp32_le)
        mismatches, first, max_abs, max_rel = 0, None, 0.0, 0.0
        for row in range(shape.m):
            expected_row = table[row % A_ROW_PERIOD]
            for column in range(shape.n):
                actual = next(values)[0]
                expected = expected_row[column % B_COL_PERIOD]
                abs_error = abs(actual - expected) if math.isfinite(actual) else float("inf")
                rel_error = abs_error / abs(expected) if expected != 0.0 else abs_error
                max_abs, max_rel = max(max_abs, abs_error), max(max_rel, rel_error)
                allowed = abs_tolerance + rel_tolerance * abs(expected)
                if not abs_error <= allowed:
                    mismatches += 1
                    if first is None:
                        first = {"row": row, "column": column, "actual_hex": float(actual).hex(),
                                 "expected_hex": expected.hex(), "abs_error_hex": float(abs_error).hex(),
                                 "allowed_error_hex": allowed.hex()}
        return {"status": "PASS" if mismatches == 0 else "FAIL", "workload": shape.to_dict(),
                "checked_elements": shape.m * shape.n, "mismatch_count": mismatches, "first_mismatch": first,
                "abs_tolerance_hex": abs_tolerance.hex(), "rel_tolerance_hex": rel_tolerance.hex(),
                "max_abs_error_hex": float(max_abs).hex(), "max_rel_error_hex": float(max_rel).hex(),
                "output_sha256": self.gpu._sha(output_fp32_le), "oracle_complexity": "O(MN+K)",
                "full_element_coverage": True}

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        """Route the C144 session's input and oracle functions through this harness."""
        saved = self.gpu.deterministic_fp16_inputs, self.gpu.correctness_check
        self.gpu.deterministic_fp16_inputs, self.gpu.correctness_check = self.inputs, self.check
        try:
            yield
        finally:
            self.gpu.deterministic_fp16_inputs, self.gpu.correctness_check = saved
