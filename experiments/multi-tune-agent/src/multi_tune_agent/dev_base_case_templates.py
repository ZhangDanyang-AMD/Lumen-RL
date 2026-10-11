"""Deterministic split-v4 templates for the Kernel Dev base-case deficits.

The factory is intentionally narrow: it accepts exactly the reviewed request
families needed to add twelve HIP and three Triton base cases.  Materialized
bundles are untrusted until the normal fresh-workspace GEAK GPU gate succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from .top10_dense_templates import render_dense_kernel, render_dense_runner
from .trusted_template_reuse import (
    ReuseError,
    _atomic_install,
    _frozen_contract,
    _shape,
    _task_record,
    merge_promoted_tasks,
)


LOCKED_AITER_SHA = "926eb3d059efd3c866c8f53ecb8b1fb8fb7135e8"
SUPPORTED_ARCH = "gfx942"
HIP_SOFTMAX_IDS = frozenset(
    {
        "phase1-hip-gfx942-gfx942-auto-0a90225f7d5a9a8b",
        "phase1-hip-gfx942-gfx942-auto-2f97edd8dbc068b4",
        "phase1-hip-gfx942-gfx942-auto-51ce9f72f21c9458",
        "phase1-hip-gfx942-gfx942-auto-a02093d46f93be46",
        "phase1-hip-gfx942-gfx942-auto-a13af183a3c97841",
        "phase1-hip-gfx942-gfx942-auto-ae865441144d1050",
        "phase1-hip-gfx942-gfx942-auto-b16a1896cd6eb7a3",
        "phase1-hip-gfx942-gfx942-auto-ca4bd98c29a00fe5",
        "phase1-hip-gfx942-gfx942-auto-d6ec8de71714dab2",
        "phase1-hip-gfx942-gfx942-auto-e32e1091c8b995b2",
    }
)
TRITON_SOFTMAX_IDS = frozenset(
    {
        "phase1-triton-gfx942-gfx942-auto-2f97edd8dbc068b4",
        "phase1-triton-gfx942-gfx942-auto-a02093d46f93be46",
    }
)
HIP_GEMM_ID = "phase1-hip-gfx942-a16w16-nt-layout-bf16"
HIP_BLOCKSCALE_ID = "phase1-hip-gfx942-fp8-blockscale-wide-n"
TRITON_SCALED_SILU_ID = "phase1-triton-gfx942-hip-scaled-silu-and-mul"
SELECTED_IDS = (
    HIP_SOFTMAX_IDS
    | TRITON_SOFTMAX_IDS
    | {HIP_GEMM_ID, HIP_BLOCKSCALE_ID, TRITON_SCALED_SILU_ID}
)
COMMAND = (
    "docker exec -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} "
    '-w "$PWD" ${GEAK_CONTAINER_NAME:-geak-phase1-vllm} '
    "python3 scripts/task_runner.py %s"
)


@dataclass(frozen=True)
class BaseCase:
    request: Mapping[str, Any]
    frozen: Mapping[str, Any]
    contract: KernelContract

    @property
    def request_id(self) -> str:
        return str(self.request["id"])


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ReuseError(f"{label} root must be a mapping")
    return value


def _dtype(contract: Mapping[str, Any], role: str) -> Any:
    nested = contract.get("dtype")
    if isinstance(nested, Mapping) and role in nested:
        return nested[role]
    return contract.get(f"{role}_dtype")


def _kernel_contract(request: Mapping[str, Any], frozen: Mapping[str, Any]) -> KernelContract:
    recognized = request["recognized_contract"]
    language = str(recognized["language"]).lower()
    input_dtype = _dtype(frozen, "input")
    weight_dtype = _dtype(frozen, "weight")
    output_dtype = _dtype(frozen, "output")
    return KernelContract(
        operator=str(frozen["operator"]),
        request=str(request["request"]),
        target_gpu=SUPPORTED_ARCH,
        architecture=SUPPORTED_ARCH,
        language=language,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        output_dtype=output_dtype,
        input_format=recognized.get("format") or input_dtype,
        weight_format=recognized.get("format") if weight_dtype else None,
        shapes=[_shape(frozen)],
    )


def load_base_cases(path: Path) -> tuple[BaseCase, ...]:
    payload = _load_yaml(path, "generation requests")
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise ReuseError("generation requests requires a requests list")
    selected: list[BaseCase] = []
    seen: set[str] = set()
    for raw in requests:
        if not isinstance(raw, Mapping) or str(raw.get("id") or "") not in SELECTED_IDS:
            continue
        request = dict(raw)
        request_id = str(request["id"])
        if request_id in seen:
            raise ReuseError(f"duplicate selected request: {request_id}")
        seen.add(request_id)
        provenance = request.get("seed_provenance")
        recognized = request.get("recognized_contract")
        if not isinstance(provenance, Mapping) or not isinstance(recognized, Mapping):
            raise ReuseError(f"{request_id}: missing contract or provenance")
        required = {
            "source_sha": LOCKED_AITER_SHA,
            "split_group": "dev",
            "split_version": "v4",
        }
        if any(provenance.get(key) != value for key, value in required.items()):
            raise ReuseError(f"{request_id}: non-frozen or non-Dev provenance")
        if (
            str(recognized.get("target_gpu") or "").lower() != SUPPORTED_ARCH
            or provenance.get("target_lane")
            != f"{str(recognized.get('language') or '').lower()}_gfx942"
        ):
            raise ReuseError(f"{request_id}: lane/architecture mismatch")
        frozen = _frozen_contract(str(request["request"]))
        if recognized.get("contract") != frozen:
            raise ReuseError(f"{request_id}: recognized/frozen contract mismatch")
        selected.append(BaseCase(request, frozen, _kernel_contract(request, frozen)))
    missing = sorted(SELECTED_IDS - seen)
    if missing:
        raise ReuseError("missing selected Dev requests: " + ", ".join(missing))
    selected.sort(key=lambda item: item.request_id)
    return tuple(selected)


def coverage_counts(items: Sequence[BaseCase]) -> dict[str, int]:
    hip = sum(item.contract.language == "hip" for item in items)
    triton = sum(item.contract.language == "triton" for item in items)
    return {"hip": hip, "triton": triton, "total": len(items)}


def _hip_softmax_kernel(extension_name: str) -> str:
    return f'''from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <cfloat>
#include <cmath>

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {{
  const int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float reduction[256];
  float local_max = -FLT_MAX;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_max = fmaxf(local_max, input[(int64_t)row * cols + col]);
  reduction[threadIdx.x] = local_max;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset; offset >>= 1) {{
    if (threadIdx.x < offset)
      reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
    __syncthreads();
  }}
  const float maximum = reduction[0];
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_sum += expf(input[(int64_t)row * cols + col] - maximum);
  reduction[threadIdx.x] = local_sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset; offset >>= 1) {{
    if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    __syncthreads();
  }}
  const float inverse = 1.0f / reduction[0];
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    output[(int64_t)row * cols + col] =
        expf(input[(int64_t)row * cols + col] - maximum) * inverse;
}}

torch::Tensor softmax(torch::Tensor input) {{
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat,
              "input must be ROCm fp32");
  TORCH_CHECK(input.is_contiguous() && input.dim() == 2,
              "input must be a contiguous matrix");
  auto output = torch::empty_like(input);
  softmax_kernel<<<input.size(0), 256, 0, at::hip::getCurrentHIPStream()>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return output;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def("softmax", &softmax); }}
"""

_MODULE = None
def _module():
    global _MODULE
    if _MODULE is None:
        build = Path(__file__).resolve().parent / "build" / "hip_extension"
        build.mkdir(parents=True, exist_ok=True)
        _MODULE = load_inline(
            name={extension_name!r}, cpp_sources="", cuda_sources=_SOURCE,
            functions=None, extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
            with_cuda=True, build_directory=str(build), verbose=False,
        )
    return _MODULE

def softmax(input_tensor: torch.Tensor, axis: int = 1) -> torch.Tensor:
    if axis != 1:
        raise ValueError("only axis=1 is supported")
    return _module().softmax(input_tensor)
'''


def _triton_softmax_kernel() -> str:
    return '''import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(x, out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    value = tl.load(x + row * N + col, mask=mask, other=-float("inf")).to(tl.float32)
    value -= tl.max(value, axis=0)
    numerator = tl.exp(value)
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(out + row * N + col, result, mask=mask)

def softmax(input_tensor: torch.Tensor, axis: int = 1) -> torch.Tensor:
    if axis != 1 or input_tensor.ndim != 2:
        raise ValueError("softmax requires a matrix and axis=1")
    if input_tensor.dtype != torch.float32 or not input_tensor.is_cuda:
        raise TypeError("input must be ROCm fp32")
    if not input_tensor.is_contiguous():
        raise ValueError("input must be contiguous")
    rows, columns = input_tensor.shape
    output = torch.empty_like(input_tensor)
    _softmax_kernel[(rows,)](
        input_tensor, output, N=columns, BLOCK=triton.next_power_of_2(columns),
        num_warps=8,
    )
    return output
'''


def _softmax_runner(shape: tuple[int, ...]) -> str:
    rows, columns = shape
    return f'''import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHAPE = {(rows, columns)!r}
CASE_ID = "softmax_fp32_{{}}x{{}}".format(*SHAPE)

def require_gfx942():
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx942":
        raise RuntimeError("expected gfx942, found " + arch)

def make_input():
    torch.manual_seed(942)
    return torch.randn(SHAPE, device="cuda", dtype=torch.float32).contiguous()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    require_gfx942()
    import kernel
    value = make_input()
    if mode == "compile":
        actual = kernel.softmax(value)
        torch.cuda.synchronize()
        if actual.shape != value.shape or actual.dtype != torch.float32:
            raise AssertionError("compiled kernel returned the wrong contract")
        print("Compile: OK")
    elif mode == "correctness":
        expected = torch.softmax(value, dim=1)
        actual = kernel.softmax(value)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        print("Correctness: OK")
    else:
        for _ in range(2):
            kernel.softmax(value)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            kernel.softmax(value)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 5.0
        print(f"Perf: {{elapsed:.6f}} ms ({{CASE_ID}})")
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        (build / "performance_report.json").write_text(
            json.dumps({{"test_cases": [{{"test_case_id": CASE_ID, "execution_time_ms": elapsed}}]}}),
            encoding="utf-8",
        )

if __name__ == "__main__":
    main()
'''


def _hip_gemm_kernel(extension_name: str) -> str:
    return f'''from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
constexpr int TILE = 16;
__global__ void gemm_kernel(
    const __hip_bfloat16* a, const __hip_bfloat16* weight,
    __hip_bfloat16* out, int M, int N, int K) {{
  __shared__ float as[TILE][TILE], ws[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int base = 0; base < K; base += TILE) {{
    int ak = base + threadIdx.x, wk = base + threadIdx.y;
    as[threadIdx.y][threadIdx.x] =
        row < M && ak < K ? __bfloat162float(a[(int64_t)row * K + ak]) : 0.0f;
    ws[threadIdx.y][threadIdx.x] =
        col < N && wk < K ? __bfloat162float(weight[(int64_t)col * K + wk]) : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) acc += as[threadIdx.y][k] * ws[k][threadIdx.x];
    __syncthreads();
  }}
  if (row < M && col < N) out[(int64_t)row * N + col] = __float2bfloat16(acc);
}}
torch::Tensor gemm(torch::Tensor a, torch::Tensor weight) {{
  TORCH_CHECK(a.is_cuda() && weight.is_cuda(), "inputs must be ROCm tensors");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
              "inputs must be bf16");
  TORCH_CHECK(a.is_contiguous() && weight.is_contiguous() && a.dim() == 2 &&
              weight.dim() == 2 && a.size(1) == weight.size(1),
              "expected contiguous A[M,K] and weight[N,K]");
  int M = a.size(0), N = weight.size(0), K = a.size(1);
  auto out = torch::empty({{M, N}}, a.options());
  dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_kernel<<<grid, block, 0, at::hip::getCurrentHIPStream()>>>(
      reinterpret_cast<const __hip_bfloat16*>(a.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr()), M, N, K);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def("gemm", &gemm); }}
"""
_MODULE = None
def _module():
    global _MODULE
    if _MODULE is None:
        build = Path(__file__).resolve().parent / "build" / "hip_extension"
        build.mkdir(parents=True, exist_ok=True)
        _MODULE = load_inline(
            name={extension_name!r}, cpp_sources="", cuda_sources=_SOURCE,
            functions=None, extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
            with_cuda=True, build_directory=str(build), verbose=False,
        )
    return _MODULE
def gemm(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _module().gemm(a, weight)
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
SHAPE = {(m, n, k)!r}
CASE_ID = "gemm_nt_bf16_{{}}x{{}}x{{}}".format(*SHAPE)

def require_gfx942():
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx942":
        raise RuntimeError("expected gfx942, found " + arch)

def make_inputs():
    torch.manual_seed(942)
    m, n, k = SHAPE
    a = (torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    weight = (torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    return a, weight

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    require_gfx942()
    import kernel
    args = make_inputs()
    if mode == "compile":
        actual = kernel.gemm(*args)
        torch.cuda.synchronize()
        if actual.shape != (SHAPE[0], SHAPE[1]) or actual.dtype != torch.bfloat16:
            raise AssertionError("compiled kernel returned the wrong contract")
        print("Compile: OK")
    elif mode == "correctness":
        expected = torch.nn.functional.linear(*args)
        actual = kernel.gemm(*args)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        print("Correctness: OK")
    else:
        for _ in range(2):
            kernel.gemm(*args)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(3):
            kernel.gemm(*args)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 3.0
        print(f"Perf: {{elapsed:.6f}} ms ({{CASE_ID}})")
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        (build / "performance_report.json").write_text(
            json.dumps({{"test_cases": [{{"test_case_id": CASE_ID, "execution_time_ms": elapsed}}]}}),
            encoding="utf-8",
        )

if __name__ == "__main__":
    main()
'''


def _triton_scaled_silu_kernel() -> str:
    return '''import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_silu_kernel(
    x, scale, out, elements: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < elements
    row = offset // HALF
    column = offset % HALF
    input_offset = row * (2 * HALF) + column
    first = tl.load(x + input_offset, mask=valid).to(tl.float32)
    second = tl.load(x + input_offset + HALF, mask=valid).to(tl.float32)
    value = first * tl.sigmoid(first) * second / scale
    tl.store(out + offset, value, mask=valid)

def scaled_silu_and_mul(input_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    if input_tensor.dtype != torch.bfloat16 or not input_tensor.is_cuda:
        raise TypeError("input must be ROCm bf16")
    if input_tensor.ndim != 2 or input_tensor.shape[1] % 2 or not input_tensor.is_contiguous():
        raise ValueError("input must be contiguous [M,2N]")
    elements = input_tensor.numel() // 2
    output = torch.empty(
        (input_tensor.shape[0], input_tensor.shape[1] // 2),
        device=input_tensor.device, dtype=torch.float8_e4m3fnuz,
    )
    _scaled_silu_kernel[(triton.cdiv(elements, 256),)](
        input_tensor, float(scale), output, elements=elements,
        HALF=input_tensor.shape[1] // 2, BLOCK=256, num_warps=4
    )
    return output
'''


def _scaled_silu_runner(shape: tuple[int, ...]) -> str:
    m, n = shape
    return f'''import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHAPE = {(m, n)!r}
SCALE = 0.75
CASE_ID = "scaled_silu_and_mul_bf16_{{}}x{{}}".format(*SHAPE)

def require_gfx942():
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx942":
        raise RuntimeError("expected gfx942, found " + arch)

def make_input():
    torch.manual_seed(942)
    return (torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()

def reference(value):
    first, second = value.chunk(2, dim=-1)
    return (torch.nn.functional.silu(first.float()) * second.float() / SCALE).to(
        torch.float8_e4m3fnuz
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    require_gfx942()
    import kernel
    value = make_input()
    if mode == "compile":
        actual = kernel.scaled_silu_and_mul(value, SCALE)
        torch.cuda.synchronize()
        if actual.shape != (SHAPE[0], SHAPE[1] // 2) or actual.dtype != torch.float8_e4m3fnuz:
            raise AssertionError("compiled kernel returned the wrong contract")
        print("Compile: OK")
    elif mode == "correctness":
        expected = reference(value)
        actual = kernel.scaled_silu_and_mul(value, SCALE)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.25)
        print("Correctness: OK")
    else:
        for _ in range(3):
            kernel.scaled_silu_and_mul(value, SCALE)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            kernel.scaled_silu_and_mul(value, SCALE)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 20.0
        print(f"Perf: {{elapsed:.6f}} ms ({{CASE_ID}})")
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        (build / "performance_report.json").write_text(
            json.dumps({{"test_cases": [{{"test_case_id": CASE_ID, "execution_time_ms": elapsed}}]}}),
            encoding="utf-8",
        )

if __name__ == "__main__":
    main()
'''


def _bundle(item: BaseCase) -> dict[str, str]:
    request_id = item.request_id
    operator = item.contract.operator
    language = item.contract.language
    shape = tuple(item.contract.shapes[0])
    extension = f"dev_base_{operator}_{item.contract.contract_hash[:12]}"
    if operator == "softmax":
        kernel = (
            _hip_softmax_kernel(extension)
            if language == "hip"
            else _triton_softmax_kernel()
        )
        runner = _softmax_runner(shape)
        target = "softmax"
    elif request_id == HIP_GEMM_ID:
        kernel = _hip_gemm_kernel(extension)
        runner = _gemm_runner(shape)
        target = "gemm"
    elif request_id == HIP_BLOCKSCALE_ID:
        kernel = render_dense_kernel("blockscale_gemm", "hip", item.frozen)
        runner = render_dense_runner("blockscale_gemm", item.frozen)
        target = "blockscale_gemm"
    elif request_id == TRITON_SCALED_SILU_ID:
        kernel = _triton_scaled_silu_kernel()
        runner = _scaled_silu_runner(shape)
        target = "scaled_silu_and_mul"
    else:
        raise ReuseError(f"no reviewed template for {request_id}")
    config = {
        "operator": operator,
        "language": language,
        "architecture": SUPPORTED_ARCH,
        "input_dtype": item.contract.input_dtype,
        "output_dtype": item.contract.output_dtype,
        "layout": item.frozen.get("layout"),
        "shapes": [list(shape)],
        "source_file_path": ["kernel.py"],
        "target_kernel_functions": [target],
        "compile_command": [COMMAND % "compile"],
        "correctness_command": [COMMAND % "correctness"],
        "performance_command": [COMMAND % "performance"],
    }
    metadata = dict(item.contract.metadata)
    metadata["provenance"] = {
        "generator": "multi_tune_agent.dev_base_case_templates",
        "generation_method": "deterministic_dev_base_case_factory",
        "source_request": str(item.request["request"]),
        "request_id": request_id,
        "contract_hash": item.contract.contract_hash,
        "case_seed": dict(item.request["seed_provenance"]),
    }
    metadata["canonical_status"] = "untrusted_pending_gpu_gate"
    return {
        "kernel.py": kernel,
        "config.yaml": yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        "scripts/task_runner.py": runner,
        "metadata.json": json.dumps(metadata, sort_keys=True, indent=2) + "\n",
    }


def materialize_candidate(item: BaseCase, candidate_root: Path) -> TemplateDraft:
    candidate_id = hashlib.sha256(
        f"{item.contract.contract_hash}:{item.request_id}".encode()
    ).hexdigest()
    destination = candidate_root.expanduser().resolve() / candidate_id
    _atomic_install(destination, _bundle(item))
    report = validate_generated_template(destination, item.contract.expected_contract)
    if not report.valid:
        raise ReuseError(
            "Dev base candidate failed static validation: "
            + "; ".join(str(issue) for issue in report.errors)
        )
    return TemplateDraft(
        destination,
        item.contract,
        report,
        "deterministic_dev_base_case_factory",
    )


def _selected(
    items: Sequence[BaseCase], shard_index: int, shard_count: int
) -> list[BaseCase]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ReuseError("shard index must be in [0, shard count)")
    return [
        item for index, item in enumerate(items) if index % shard_count == shard_index
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "materialize", "gate"))
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--production-catalog", required=True, type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--verified-root", type=Path)
    parser.add_argument("--output-catalog", type=Path)
    parser.add_argument("--geak-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpu-id", choices=tuple(str(i) for i in range(1, 8)), default="1")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--command-timeout", type=int, default=1800)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = load_base_cases(args.requests.expanduser().resolve())
    summary: dict[str, Any] = coverage_counts(items)
    if args.mode == "plan":
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.candidate_root is None:
        raise ReuseError("materialize and gate require --candidate-root")
    selected = _selected(items, args.shard_index, args.shard_count)
    catalog = args.production_catalog.expanduser().resolve()
    exact = {
        str(task.get("id") or "")
        for task in _load_yaml(catalog, "production catalog").get("tasks", [])
        if isinstance(task, Mapping)
    }
    selected = [item for item in selected if item.request_id not in exact]
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
        raise ReuseError("gate requires " + ", ".join(missing))
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    diagnostics = args.run_root.expanduser().resolve() / "dev-base-gate-results.jsonl"
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    for item, draft in drafts:
        result = run_template_gpu_gate(
            draft,
            geak_root=args.geak_root,
            run_root=args.run_root,
            gpu_ids=args.gpu_id,
            command_timeout=args.command_timeout,
        )
        diagnostic = {
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
        }
        with diagnostics.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(diagnostic, sort_keys=True) + "\n")
        if not result.trusted:
            failures.append(item.request_id)
            continue
        promoted = promote_validated_template(draft, result, args.verified_root)
        records.append(_task_record(
            # _task_record only relies on the shared request/contract attributes.
            item, draft, promoted  # type: ignore[arg-type]
        ))
    merge_promoted_tasks(catalog, args.output_catalog, records)
    summary["promoted"] = len(records)
    summary["failed_gate"] = failures
    summary["report"] = str(diagnostics)
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
