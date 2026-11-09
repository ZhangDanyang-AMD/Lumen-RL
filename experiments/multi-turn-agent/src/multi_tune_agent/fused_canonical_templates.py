"""Source-backed canonical factories for selected frozen gfx942 fusion contracts.

The generated bundles are untrusted drafts.  They can only enter a catalog
after the normal fresh-workspace GPU compile/correctness/performance gate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
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


LOCKED_AITER_SHA = "926eb3d059efd3c866c8f53ecb8b1fb8fb7135e8"
SUPPORTED_ARCH = "gfx942"
SUPPORTED_FAMILIES = frozenset({"gemm_activation", "fused_mul_add"})
SUPPORTED_LANGUAGES = frozenset({"hip", "triton"})
ACTIVATIONS = ("gelu", "gelu_tanh", "silu")
OPERAND_KINDS = (
    "python_float_scalar",
    "python_int_scalar",
    "tensor_scalar",
    "tensor",
)
PROVENANCE = {
    "gemm_activation": (
        "op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py",
        "test_gemm_a16_w16_activation",
        "07bb9e3f7ef794ea8e4a962b8083c7469fb132b3fe7bf9a627d59ad73a896172",
    ),
    "fused_mul_add": (
        "op_tests/triton_tests/fusions/test_fused_mul_add.py",
        "test_mul_add",
        "99b883efc0e14bb91a9acedc6f6a34e3abf673d4844ce5572cce54f4a1bfb9ab",
    ),
}
CANONICAL_FILES = (
    "kernel.py",
    "config.yaml",
    "scripts/task_runner.py",
    "metadata.json",
)
COMMAND = (
    "docker exec -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} "
    '-w "$PWD" ${GEAK_CONTAINER_NAME:-geak-phase1-vllm} '
    "python3 scripts/task_runner.py %s"
)


class CanonicalFactoryError(ValueError):
    """A request is outside the exact source-backed canonical contract."""


@dataclass(frozen=True)
class CanonicalRequest:
    request: Mapping[str, Any]
    contract: Mapping[str, Any]
    kernel_contract: KernelContract

    @property
    def request_id(self) -> str:
        return str(self.request["id"])


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise CanonicalFactoryError(f"{label} root must be a mapping")
    return value


def _frozen_contract(text: str) -> dict[str, Any]:
    marker = "Frozen contract JSON:"
    start = text.find(marker)
    if start < 0:
        raise CanonicalFactoryError("request has no Frozen contract JSON")
    start = text.find("{", start + len(marker))
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, TypeError) as exc:
        raise CanonicalFactoryError("request has invalid Frozen contract JSON") from exc
    if not isinstance(value, dict):
        raise CanonicalFactoryError("frozen contract must be an object")
    return value


def _shape(contract: Mapping[str, Any]) -> tuple[int, ...]:
    raw = contract.get("shape")
    if not isinstance(raw, Mapping):
        raise CanonicalFactoryError("contract shape must be a mapping")
    if isinstance(raw.get("dims"), list):
        values = raw["dims"]
    elif all(name in raw for name in ("M", "N", "K")):
        values = [raw["M"], raw["N"], raw["K"]]
    else:
        raise CanonicalFactoryError("unsupported canonical shape")
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise CanonicalFactoryError("shape dimensions must be positive integers")
    return tuple(values)


def _validate_exact_contract(contract: Mapping[str, Any], language: str) -> None:
    operator = contract.get("operator")
    if operator not in SUPPORTED_FAMILIES:
        raise CanonicalFactoryError("unsupported canonical family")
    if language not in SUPPORTED_LANGUAGES:
        raise CanonicalFactoryError("canonical language must be HIP or Triton")
    if contract.get("input_dtype") != "bf16" or contract.get("output_dtype") != "bf16":
        raise CanonicalFactoryError("canonical contracts require bf16 input/output")
    _shape(contract)
    if operator == "gemm_activation":
        expected = {
            "operator": "gemm_activation",
            "input_dtype": "bf16",
            "weight_dtype": "bf16",
            "accum_dtype": "fp32",
            "output_dtype": "bf16",
            "layout": "TN",
            "activation_options": list(ACTIVATIONS),
        }
    else:
        expected = {
            "operator": "fused_mul_add",
            "input_dtype": "bf16",
            "output_dtype": "bf16",
            "layout": "contiguous",
            "operand_kinds": list(OPERAND_KINDS),
        }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise CanonicalFactoryError(f"frozen contract mismatch for {key}")
    if set(contract) != {*expected, "shape"}:
        raise CanonicalFactoryError("frozen contract has unexpected fields")


def _verify_provenance(
    request: Mapping[str, Any], contract: Mapping[str, Any], aiter_root: Path
) -> None:
    provenance = request.get("seed_provenance")
    if not isinstance(provenance, Mapping):
        raise CanonicalFactoryError("request lacks seed provenance")
    expected_path, expected_test, expected_hash = PROVENANCE[str(contract["operator"])]
    expected = {
        "source_repo": "https://github.com/ROCm/aiter.git",
        "source_sha": LOCKED_AITER_SHA,
        "source_license": "MIT",
        "source_test_path": expected_path,
        "source_test_id": expected_test,
        "source_test_sha256": expected_hash,
        "split_group": "train",
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise CanonicalFactoryError(f"source provenance mismatch for {key}")
    actual_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=aiter_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_revision != LOCKED_AITER_SHA:
        raise CanonicalFactoryError(
            "locked AITER revision mismatch: "
            f"expected {LOCKED_AITER_SHA}, got {actual_revision}"
        )
    source = (aiter_root / expected_path).resolve(strict=True)
    source.relative_to(aiter_root.resolve(strict=True))
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise CanonicalFactoryError(
            f"locked AITER source hash mismatch: expected {expected_hash}, got {actual_hash}"
        )


def _kernel_contract(
    request: Mapping[str, Any], contract: Mapping[str, Any], language: str
) -> KernelContract:
    shape = _shape(contract)
    return KernelContract(
        operator=str(contract["operator"]),
        request=str(request["request"]),
        target_gpu=SUPPORTED_ARCH,
        architecture=SUPPORTED_ARCH,
        language=language,
        input_dtype="bf16",
        weight_dtype="bf16" if contract["operator"] == "gemm_activation" else None,
        output_dtype="bf16",
        input_format="bf16",
        weight_format="bf16" if contract["operator"] == "gemm_activation" else None,
        shapes=[shape],
    )


def load_canonical_requests(
    requests_path: Path, aiter_root: Path
) -> tuple[CanonicalRequest, ...]:
    payload = _load_yaml(requests_path, "generation requests")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        raise CanonicalFactoryError("generation requests requires a requests list")
    selected: list[CanonicalRequest] = []
    seen: set[str] = set()
    for raw in raw_requests:
        if not isinstance(raw, Mapping):
            continue
        recognized = raw.get("recognized_contract")
        if not isinstance(recognized, Mapping):
            continue
        if recognized.get("operator") not in SUPPORTED_FAMILIES:
            continue
        language = str(recognized.get("language") or "").lower()
        architecture = str(recognized.get("target_gpu") or "").lower()
        if architecture != SUPPORTED_ARCH:
            continue
        contract = _frozen_contract(str(raw.get("request") or ""))
        if recognized.get("contract") != contract:
            raise CanonicalFactoryError(f"{raw.get('id')}: recognized/frozen mismatch")
        _validate_exact_contract(contract, language)
        _verify_provenance(raw, contract, aiter_root)
        request_id = str(raw.get("id") or "")
        if not request_id or request_id in seen:
            raise CanonicalFactoryError(f"invalid or duplicate request ID: {request_id!r}")
        seen.add(request_id)
        selected.append(
            CanonicalRequest(
                dict(raw),
                contract,
                _kernel_contract(raw, contract, language),
            )
        )
    selected.sort(key=lambda item: item.request_id)
    return tuple(selected)


def coverage_counts(items: Sequence[CanonicalRequest]) -> dict[str, int]:
    counts = {
        f"{operator}_{language}": sum(
            item.contract["operator"] == operator
            and item.kernel_contract.language == language
            for item in items
        )
        for operator in sorted(SUPPORTED_FAMILIES)
        for language in sorted(SUPPORTED_LANGUAGES)
    }
    counts["total"] = len(items)
    return counts


def _hip_gemm_kernel(extension_name: str) -> str:
    return f'''from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>

constexpr int TILE = 16;

__device__ __forceinline__ float activate(float x, int kind) {{
  if (kind == 0) return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
  if (kind == 1) {{
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
  }}
  return x / (1.0f + expf(-x));
}}

__global__ void gemm_activation_kernel(
    const __hip_bfloat16* a, const __hip_bfloat16* w, __hip_bfloat16* out,
    int64_t M, int64_t N, int64_t K, int activation) {{
  __shared__ float tile_a[TILE][TILE];
  __shared__ float tile_w[TILE][TILE];
  const int64_t row = blockIdx.y * TILE + threadIdx.y;
  const int64_t col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int64_t base = 0; base < K; base += TILE) {{
    const int64_t ak = base + threadIdx.x;
    const int64_t wk = base + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] =
        row < M && ak < K ? __bfloat162float(a[row * K + ak]) : 0.0f;
    tile_w[threadIdx.y][threadIdx.x] =
        col < N && wk < K ? __bfloat162float(w[col * K + wk]) : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k)
      acc += tile_a[threadIdx.y][k] * tile_w[k][threadIdx.x];
    __syncthreads();
  }}
  if (row < M && col < N)
    out[row * N + col] = __float2bfloat16(activate(acc, activation));
}}

torch::Tensor gemm_activation(
    torch::Tensor a, torch::Tensor w, int64_t activation) {{
  TORCH_CHECK(a.is_cuda() && w.is_cuda(), "inputs must be ROCm tensors");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 &&
              w.scalar_type() == at::kBFloat16, "inputs must be bf16");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.size(1) == w.size(1),
              "expected A[M,K] and W[N,K]");
  TORCH_CHECK(activation >= 0 && activation <= 2, "invalid activation");
  const int64_t M = a.size(0), N = w.size(0), K = a.size(1);
  auto out = torch::empty({{M, N}}, a.options());
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  hipStream_t stream = at::hip::getDefaultHIPStream();
  gemm_activation_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(a.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(w.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr()), M, N, K, activation);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("gemm_activation", &gemm_activation);
}}
"""

_MODULE = None
_ACTIVATIONS = {{"gelu": 0, "gelu_tanh": 1, "silu": 2}}


def _module():
    global _MODULE
    if _MODULE is None:
        build = Path(__file__).resolve().parent / "build" / "hip_extension"
        build.mkdir(parents=True, exist_ok=True)
        _MODULE = load_inline(
            name="{extension_name}",
            cpp_sources="",
            cuda_sources=_SOURCE,
            functions=None,
            extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
            with_cuda=True,
            build_directory=str(build),
            verbose=False,
        )
    return _MODULE


def gemm_activation(a: torch.Tensor, w: torch.Tensor, activation: str) -> torch.Tensor:
    if activation not in _ACTIVATIONS:
        raise ValueError("activation must be gelu, gelu_tanh, or silu")
    return _module().gemm_activation(a, w, _ACTIVATIONS[activation])
'''


def _triton_gemm_kernel() -> str:
    return '''import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_activation_kernel(
    a_ptr, w_ptr, out_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for base in range(0, tl.cdiv(K, BLOCK_K)):
        k = base * BLOCK_K + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * K + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a, w)
    if ACTIVATION == "gelu":
        acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))
    elif ACTIVATION == "gelu_tanh":
        inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
        acc = 0.5 * acc * (1.0 + (2.0 * tl.sigmoid(2.0 * inner) - 1.0))
    else:
        acc = acc * tl.sigmoid(acc)
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def gemm_activation(a: torch.Tensor, w: torch.Tensor, activation: str) -> torch.Tensor:
    if a.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise TypeError("inputs must be bf16")
    if not a.is_cuda or not w.is_cuda or not a.is_contiguous() or not w.is_contiguous():
        raise ValueError("inputs must be contiguous ROCm tensors")
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[1]:
        raise ValueError("expected A[M,K] and W[N,K]")
    if activation not in ("gelu", "gelu_tanh", "silu"):
        raise ValueError("unsupported activation")
    M, K = a.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 32) * triton.cdiv(N, 32),)
    _gemm_activation_kernel[grid](
        a, w, out, M, N, K,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
        ACTIVATION=activation, num_warps=4, num_stages=2,
    )
    return out
'''


def _hip_fma_kernel(extension_name: str) -> str:
    return f'''from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

__global__ void fused_mul_add_kernel(
    const __hip_bfloat16* x, const __hip_bfloat16* a,
    const __hip_bfloat16* b, __hip_bfloat16* out, int64_t n,
    int a_kind, int b_kind, float a_scalar, float b_scalar) {{
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  const float av = a_kind == 0 ? a_scalar :
      __bfloat162float(a[a_kind == 1 ? 0 : index]);
  const float bv = b_kind == 0 ? b_scalar :
      __bfloat162float(b[b_kind == 1 ? 0 : index]);
  out[index] = __float2bfloat16(av * __bfloat162float(x[index]) + bv);
}}

torch::Tensor fused_mul_add(
    torch::Tensor x, torch::Tensor a, torch::Tensor b,
    int64_t a_kind, int64_t b_kind, double a_scalar, double b_scalar) {{
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 &&
              x.is_contiguous(), "x must be contiguous ROCm bf16");
  TORCH_CHECK(a_kind >= 0 && a_kind <= 2 && b_kind >= 0 && b_kind <= 2,
              "invalid operand kind");
  if (a_kind != 0) {{
    TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous(),
                "tensor a must be contiguous ROCm bf16");
    TORCH_CHECK(a.numel() == (a_kind == 1 ? 1 : x.numel()), "invalid a size");
  }}
  if (b_kind != 0) {{
    TORCH_CHECK(b.is_cuda() && b.scalar_type() == at::kBFloat16 && b.is_contiguous(),
                "tensor b must be contiguous ROCm bf16");
    TORCH_CHECK(b.numel() == (b_kind == 1 ? 1 : x.numel()), "invalid b size");
  }}
  auto out = torch::empty_like(x);
  const int threads = 256;
  hipStream_t stream = at::hip::getDefaultHIPStream();
  fused_mul_add_kernel<<<(x.numel() + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(a.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(b.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr()), x.numel(),
      a_kind, b_kind, static_cast<float>(a_scalar), static_cast<float>(b_scalar));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("fused_mul_add", &fused_mul_add);
}}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        build = Path(__file__).resolve().parent / "build" / "hip_extension"
        build.mkdir(parents=True, exist_ok=True)
        _MODULE = load_inline(
            name="{extension_name}",
            cpp_sources="",
            cuda_sources=_SOURCE,
            functions=None,
            extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
            with_cuda=True,
            build_directory=str(build),
            verbose=False,
        )
    return _MODULE


def _operand(value, x):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value, 1, 0.0
        if value.shape != x.shape:
            raise ValueError("tensor operand must be scalar or match x")
        return value, 2, 0.0
    if type(value) not in (float, int):
        raise TypeError("operand must be float, int, or tensor")
    return x, 0, float(value)


def fused_mul_add(x: torch.Tensor, a, b) -> torch.Tensor:
    a_tensor, a_kind, a_scalar = _operand(a, x)
    b_tensor, b_kind, b_scalar = _operand(b, x)
    return _module().fused_mul_add(
        x, a_tensor, b_tensor, a_kind, b_kind, a_scalar, b_scalar
    )
'''


def _triton_fma_kernel() -> str:
    return '''import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mul_add_kernel(
    x_ptr, a_ptr, b_ptr, out_ptr, n,
    A_KIND: tl.constexpr, B_KIND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    if A_KIND == 0:
        a = a_ptr
    elif A_KIND == 1:
        a = tl.load(a_ptr)
    else:
        a = tl.load(a_ptr + offsets, mask=mask)
    if B_KIND == 0:
        b = b_ptr
    elif B_KIND == 1:
        b = tl.load(b_ptr)
    else:
        b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a.to(tl.float32) * x + b.to(tl.float32), mask=mask)


def _operand(value, x):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value, 1
        if value.shape != x.shape or not value.is_contiguous():
            raise ValueError("tensor operand must be contiguous scalar or match x")
        return value, 2
    if type(value) not in (float, int):
        raise TypeError("operand must be float, int, or tensor")
    return float(value), 0


def fused_mul_add(x: torch.Tensor, a, b) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be contiguous ROCm bf16")
    a_value, a_kind = _operand(a, x)
    b_value, b_kind = _operand(b, x)
    out = torch.empty_like(x)
    n = x.numel()
    _fused_mul_add_kernel[(triton.cdiv(n, 256),)](
        x, a_value, b_value, out, n,
        A_KIND=a_kind, B_KIND=b_kind, BLOCK=256, num_warps=4,
    )
    return out
'''


def _gemm_runner(shape: tuple[int, ...]) -> str:
    m, n, k = shape
    return f'''import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHAPE = ({m}, {n}, {k})
ACTIVATIONS = ("gelu", "gelu_tanh", "silu")
RTOL = 1.0e-2
ATOL = 1.0e-1


def enforce_architecture(required):
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable")
    actual = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if actual != required:
        raise RuntimeError(f"architecture mismatch: required {{required}}, got {{actual}}")


def make_inputs():
    torch.manual_seed(1701)
    m, n, k = SHAPE
    a = (torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    w = (torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    return a, w


def reference(a, w, activation):
    value = torch.nn.functional.linear(a, w)
    if activation == "gelu":
        return torch.nn.functional.gelu(value)
    if activation == "gelu_tanh":
        return torch.nn.functional.gelu(value, approximate="tanh")
    return torch.nn.functional.silu(value)


def compile_mode():
    import kernel
    a, w = make_inputs()
    actual = kernel.gemm_activation(a, w, ACTIVATIONS[0])
    torch.cuda.synchronize()
    if actual.shape != (SHAPE[0], SHAPE[1]) or actual.dtype != torch.bfloat16:
        raise AssertionError("compiled kernel returned the wrong contract")
    print("Compile: OK")


def correctness_mode():
    import kernel
    a, w = make_inputs()
    for activation in ACTIVATIONS:
        expected = reference(a, w, activation)
        actual = kernel.gemm_activation(a, w, activation)
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    print("Correctness: OK")


def performance_mode():
    import kernel
    a, w = make_inputs()
    results = []
    for activation in ACTIVATIONS:
        for _ in range(2):
            kernel.gemm_activation(a, w, activation)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(3):
            kernel.gemm_activation(a, w, activation)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 3.0
        case_id = f"gemm_activation_bf16_{{SHAPE[0]}}x{{SHAPE[1]}}x{{SHAPE[2]}}_{{activation}}"
        print(f"Perf: {{elapsed:.6f}} ms ({{case_id}})")
        results.append({{"test_case_id": case_id, "execution_time_ms": elapsed}})
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    (build / "performance_report.json").write_text(
        json.dumps({{"test_cases": results}}), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    enforce_architecture("gfx942")
    if mode == "compile":
        compile_mode()
    elif mode == "correctness":
        correctness_mode()
    else:
        performance_mode()


if __name__ == "__main__":
    main()
'''


def _fma_runner(shape: tuple[int, ...]) -> str:
    return f'''import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHAPE = {shape!r}
KINDS = ("python_float_scalar", "python_int_scalar", "tensor_scalar", "tensor")
RTOL = 1.0e-2
ATOL = 1.0e-2


def enforce_architecture(required):
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable")
    actual = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if actual != required:
        raise RuntimeError(f"architecture mismatch: required {{required}}, got {{actual}}")


def make_x():
    torch.manual_seed(1701)
    return torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16).contiguous()


def make_operand(kind, seed):
    torch.manual_seed(seed)
    if kind == "python_float_scalar":
        return 1.25
    if kind == "python_int_scalar":
        return -2
    if kind == "tensor_scalar":
        return torch.randn((1,), device="cuda", dtype=torch.bfloat16)
    return torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16).contiguous()


def reference(x, a, b):
    return (a * x.float() + b).to(torch.bfloat16)


def compile_mode():
    import kernel
    x = make_x()
    actual = kernel.fused_mul_add(x, 1.25, -2)
    torch.cuda.synchronize()
    if actual.shape != SHAPE or actual.dtype != torch.bfloat16:
        raise AssertionError("compiled kernel returned the wrong contract")
    print("Compile: OK")


def correctness_mode():
    import kernel
    x = make_x()
    for a_index, a_kind in enumerate(KINDS):
        a = make_operand(a_kind, 1801 + a_index)
        for b_index, b_kind in enumerate(KINDS):
            b = make_operand(b_kind, 1901 + b_index)
            expected = reference(x, a, b)
            actual = kernel.fused_mul_add(x, a, b)
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    print("Correctness: OK")


def performance_mode():
    import kernel
    x = make_x()
    results = []
    for a_index, a_kind in enumerate(KINDS):
        a = make_operand(a_kind, 1801 + a_index)
        for b_index, b_kind in enumerate(KINDS):
            b = make_operand(b_kind, 1901 + b_index)
            for _ in range(3):
                kernel.fused_mul_add(x, a, b)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                kernel.fused_mul_add(x, a, b)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) / 20.0
            case_id = f"fused_mul_add_bf16_{{'x'.join(map(str, SHAPE))}}_{{a_kind}}_{{b_kind}}"
            print(f"Perf: {{elapsed:.6f}} ms ({{case_id}})")
            results.append({{"test_case_id": case_id, "execution_time_ms": elapsed}})
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    (build / "performance_report.json").write_text(
        json.dumps({{"test_cases": results}}), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    enforce_architecture("gfx942")
    if mode == "compile":
        compile_mode()
    elif mode == "correctness":
        correctness_mode()
    else:
        performance_mode()


if __name__ == "__main__":
    main()
'''


def _bundle(item: CanonicalRequest) -> dict[str, str]:
    contract = item.kernel_contract
    operator = contract.operator
    language = contract.language
    shape = tuple(contract.shapes[0])
    extension_name = f"canonical_{operator}_{contract.contract_hash[:12]}"
    if operator == "gemm_activation":
        kernel = (
            _hip_gemm_kernel(extension_name)
            if language == "hip"
            else _triton_gemm_kernel()
        )
        runner = _gemm_runner(shape)
        target = "gemm_activation"
    else:
        kernel = (
            _hip_fma_kernel(extension_name)
            if language == "hip"
            else _triton_fma_kernel()
        )
        runner = _fma_runner(shape)
        target = "fused_mul_add"
    config = {
        "operator": operator,
        "language": language,
        "architecture": SUPPORTED_ARCH,
        "input_dtype": "bf16",
        "output_dtype": "bf16",
        "layout": item.contract["layout"],
        "shapes": [list(shape)],
        "source_file_path": ["kernel.py"],
        "target_kernel_functions": [target],
        "compile_command": [COMMAND % "compile"],
        "correctness_command": [COMMAND % "correctness"],
        "performance_command": [COMMAND % "performance"],
    }
    metadata = dict(contract.metadata)
    source_path, source_test, source_hash = PROVENANCE[operator]
    metadata["provenance"] = {
        "generator": "multi_tune_agent.fused_canonical_templates",
        "generation_method": "source_backed_canonical_factory",
        "source_request": str(item.request["request"]),
        "request_id": item.request_id,
        "contract_hash": contract.contract_hash,
        "case_seed": dict(item.request["seed_provenance"]),
        "source_artifacts": [
            {
                "repository": "https://github.com/ROCm/aiter.git",
                "revision": LOCKED_AITER_SHA,
                "path": source_path,
                "test_id": source_test,
                "sha256": source_hash,
            }
        ],
    }
    metadata["canonical_status"] = "untrusted_pending_gpu_gate"
    return {
        "kernel.py": kernel,
        "config.yaml": yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        "scripts/task_runner.py": runner,
        "metadata.json": json.dumps(metadata, sort_keys=True, indent=2) + "\n",
    }


def _atomic_install(destination: Path, files: Mapping[str, str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".tmp-{destination.name}-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(mode=0o700)
        for relative, text in files.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        if destination.exists():
            if not destination.is_dir() or destination.is_symlink():
                raise CanonicalFactoryError(f"invalid existing draft: {destination}")
            existing = {
                relative: (destination / relative).read_text(encoding="utf-8")
                for relative in CANONICAL_FILES
            }
            if existing != dict(files):
                raise CanonicalFactoryError(
                    f"draft exists with different content: {destination}"
                )
            return
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def materialize_candidate(
    item: CanonicalRequest, candidate_root: Path
) -> TemplateDraft:
    destination = candidate_root.expanduser().resolve() / item.kernel_contract.contract_hash
    _atomic_install(destination, _bundle(item))
    report = validate_generated_template(
        destination, item.kernel_contract.expected_contract
    )
    if not report.valid:
        raise CanonicalFactoryError(
            "canonical draft failed static validation: "
            + "; ".join(str(issue) for issue in report.errors)
        )
    return TemplateDraft(
        destination,
        item.kernel_contract,
        report,
        "source_backed_canonical_factory",
    )


def _task_record(
    item: CanonicalRequest, draft: TemplateDraft, promoted: Path
) -> dict[str, Any]:
    metadata = json.loads((promoted / "metadata.json").read_text(encoding="utf-8"))
    return {
        "id": item.request_id,
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
        "recognized_contract": dict(item.request["recognized_contract"]),
    }


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
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
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def merge_promoted_tasks(
    source_catalog: Path | None,
    output_catalog: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    lock_path = output_catalog.with_suffix(output_catalog.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output_catalog.is_file():
            payload = _load_yaml(output_catalog, "output catalog")
        elif source_catalog is not None:
            payload = _load_yaml(source_catalog, "source catalog")
        else:
            payload = {"schema_version": "canonical_gpu_promotions_v1", "tasks": []}
        tasks = {
            str(task["id"]): dict(task)
            for task in payload.get("tasks", [])
            if isinstance(task, Mapping)
        }
        for raw in records:
            record = dict(raw)
            task_id = str(record["id"])
            if task_id in tasks and tasks[task_id] != record:
                raise CanonicalFactoryError(
                    f"promotion would replace existing task {task_id}"
                )
            metadata = json.loads(
                (Path(record["kernel_path"]) / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            trust = metadata.get("trust")
            if not isinstance(trust, Mapping) or trust.get("trusted") is not True:
                raise CanonicalFactoryError(f"refusing to catalog untrusted task {task_id}")
            tasks[task_id] = record
        payload["tasks"] = [tasks[key] for key in sorted(tasks)]
        _atomic_yaml(output_catalog, payload)


def _append_diagnostic(path: Path, diagnostic: Mapping[str, Any]) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(diagnostic), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _selected(
    items: Sequence[CanonicalRequest], shard_index: int, shard_count: int
) -> list[CanonicalRequest]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise CanonicalFactoryError("shard index must be in [0, shard count)")
    return [
        item for index, item in enumerate(items) if index % shard_count == shard_index
    ]


def _trusted_completed_ids(output_catalog: Path) -> set[str]:
    if not output_catalog.is_file():
        return set()
    payload = _load_yaml(output_catalog, "output catalog")
    completed: set[str] = set()
    for raw in payload.get("tasks", []):
        if not isinstance(raw, Mapping):
            continue
        task_id = str(raw.get("id") or "")
        contract_hash = str(raw.get("contract_hash") or "")
        kernel_path = Path(str(raw.get("kernel_path") or "")).expanduser()
        if not task_id or not contract_hash or not kernel_path.is_dir():
            continue
        try:
            metadata = json.loads(
                (kernel_path / "metadata.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        trust = metadata.get("trust") if isinstance(metadata, Mapping) else None
        if (
            isinstance(trust, Mapping)
            and trust.get("trusted") is True
            and trust.get("contract_hash") == contract_hash
            and metadata.get("contract_hash") == contract_hash
        ):
            completed.add(task_id)
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "materialize", "gate"))
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--aiter-root", required=True, type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--verified-root", type=Path)
    parser.add_argument("--output-catalog", type=Path)
    parser.add_argument("--source-catalog", type=Path)
    parser.add_argument("--geak-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpu-id", choices=tuple(str(i) for i in range(1, 8)), default="1")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=7)
    parser.add_argument("--command-timeout", type=int, default=900)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = load_canonical_requests(
        args.requests.expanduser().resolve(),
        args.aiter_root.expanduser().resolve(),
    )
    summary: dict[str, Any] = coverage_counts(items)
    if args.mode == "plan":
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.candidate_root is None:
        raise CanonicalFactoryError("materialize and gate require --candidate-root")
    selected = _selected(items, args.shard_index, args.shard_count)
    if args.mode == "gate" and args.output_catalog and args.output_catalog.is_file():
        completed = _trusted_completed_ids(args.output_catalog)
        selected = [item for item in selected if item.request_id not in completed]
    drafts = [
        (item, materialize_candidate(item, args.candidate_root)) for item in selected
    ]
    summary.update(
        selected=len(drafts),
        materialized=len(drafts),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
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
        raise CanonicalFactoryError("gate requires " + ", ".join(missing))
    if args.shard_count == 7 and int(args.gpu_id) != args.shard_index + 1:
        raise CanonicalFactoryError(
            "seven-way gate requires GPU N to run shard N-1"
        )
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    diagnostics = args.run_root.expanduser().resolve() / "canonical-gate-results.jsonl"
    for item, draft in drafts:
        result = run_template_gpu_gate(
            draft,
            geak_root=args.geak_root,
            run_root=args.run_root,
            gpu_ids=args.gpu_id,
            command_timeout=args.command_timeout,
        )
        _append_diagnostic(
            diagnostics,
            {
                "case_id": item.request_id,
                "contract_hash": draft.contract_hash,
                "trusted": result.trusted,
                "compiled": result.compiled,
                "correct": result.correct,
                "performance_valid": result.performance_valid,
                "errors": list(result.errors),
                "commands": {
                    mode: dict(command)
                    for mode, command in result.command_summaries.items()
                },
                "validation_workspace": (
                    str(result.validation_workspace)
                    if result.validation_workspace is not None
                    else None
                ),
            },
        )
        if not result.trusted:
            failures.append(item.request_id)
            continue
        promoted = promote_validated_template(draft, result, args.verified_root)
        records.append(_task_record(item, draft, promoted))
    merge_promoted_tasks(args.source_catalog, args.output_catalog, records)
    summary["promoted"] = len(records)
    summary["failed_gate"] = failures
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVATIONS",
    "CanonicalFactoryError",
    "CanonicalRequest",
    "LOCKED_AITER_SHA",
    "OPERAND_KINDS",
    "coverage_counts",
    "load_canonical_requests",
    "materialize_candidate",
    "merge_promoted_tasks",
]
