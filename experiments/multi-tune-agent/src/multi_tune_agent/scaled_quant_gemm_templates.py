"""Canonical standalone templates for the gfx942 int8 scaled GEMM family.

The factory is deterministic and source-backed, but generated kernels do not
import or call AITER.  Materialized candidates are deliberately untrusted until
the standard static and GPU compile/correctness/performance gates promote them.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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


FAMILY_ID = "CF-GFX942-AUTO-0289A9B7C215"
LOCKED_AITER_SHA = "926eb3d059efd3c866c8f53ecb8b1fb8fb7135e8"
CANONICAL_FILES = (
    "kernel.py",
    "config.yaml",
    "scripts/task_runner.py",
    "metadata.json",
)
EXPECTED_SHAPES = frozenset(
    (m, n, k)
    for m in (1, 32, 64, 128, 192, 256, 320, 512, 1024, 2048, 4096, 8192, 16384)
    for n, k in ((1280, 8192), (8192, 1024))
)
SOURCE_ARTIFACTS = (
    {
        "path": "op_tests/test_gemm_a8w8.py",
        "sha256": "ae1226beb297b8d4de63fbcbf0baa72bf924e145f13425ddc3714dfd0cd2d653",
        "role": "hip_contract_oracle_test",
    },
    {
        "path": "op_tests/triton_tests/gemm/basic/test_gemm_a8w8_per_token_scale.py",
        "sha256": "d5b85856ac85a23d280c547ac65d1918a1c869ac89e33b7dc570d21dda81d615",
        "role": "triton_contract_oracle_test",
    },
    {
        "path": "aiter/ops/triton/gemm/basic/gemm_a8w8_per_token_scale.py",
        "sha256": "238b7438f62d99523e2a4351782359e5863dd82d7618da8f4fb242b63083e659",
        "role": "triton_wrapper_contract",
    },
    {
        "path": "aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_per_token_scale.py",
        "sha256": "9800939f7d8ad75fee897cef2e18c7073edb01df99c306acee5d86436b96f973",
        "role": "triton_source_provenance",
    },
    {
        "path": "aiter/configs/a8w8_tuned_gemm.csv",
        "sha256": "c0b9a313244b38a2b60ca55650ad270ff49307dc05050bbf49e641c717fad3f0",
        "role": "shape_provenance",
    },
)


class CanonicalTemplateError(ValueError):
    """The request or generated template violates the frozen family contract."""


@dataclass(frozen=True)
class FamilyRequest:
    request_id: str
    request_text: str
    language: str
    shape: tuple[int, int, int]
    seed_provenance: Mapping[str, Any]
    recognized_contract: Mapping[str, Any]

    @property
    def contract(self) -> KernelContract:
        return KernelContract(
            operator="scaled_quant_gemm",
            request=self.request_text,
            target_gpu="gfx942",
            architecture="gfx942",
            language=self.language,
            input_dtype="int8",
            weight_dtype="int8",
            output_dtype="bf16",
            input_scale_granularity="per_row",
            weight_scale_granularity="per_column",
            shapes=(self.shape,),
        )


def _mapping_file(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise CanonicalTemplateError(f"{label} root must be a mapping")
    return value


def _dtype(contract: Mapping[str, Any], name: str) -> Any:
    nested = contract.get("dtype")
    if isinstance(nested, Mapping):
        return nested.get(name)
    return contract.get(f"{name}_dtype")


def _parse_family_request(raw: Mapping[str, Any]) -> FamilyRequest:
    request_id = str(raw.get("id") or "").strip()
    request_text = str(raw.get("request") or "").strip()
    seed = raw.get("seed_provenance")
    recognized = raw.get("recognized_contract")
    if not request_id or not request_text:
        raise CanonicalTemplateError("family request requires id and request text")
    if not isinstance(seed, Mapping) or not isinstance(recognized, Mapping):
        raise CanonicalTemplateError(f"{request_id}: missing request provenance/contract")
    if seed.get("source_sha") != LOCKED_AITER_SHA:
        raise CanonicalTemplateError(f"{request_id}: unexpected AITER source revision")
    if seed.get("split_group") != "train":
        raise CanonicalTemplateError(f"{request_id}: only train requests may materialize")
    language = str(recognized.get("language") or "").lower()
    if language not in {"hip", "triton"}:
        raise CanonicalTemplateError(f"{request_id}: unsupported language {language!r}")
    contract = recognized.get("contract")
    if not isinstance(contract, Mapping):
        raise CanonicalTemplateError(f"{request_id}: missing frozen contract")
    shape = contract.get("shape")
    if not isinstance(shape, Mapping):
        raise CanonicalTemplateError(f"{request_id}: missing M/N/K")
    try:
        mnk = tuple(int(shape[name]) for name in ("M", "N", "K"))
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalTemplateError(f"{request_id}: invalid M/N/K") from exc
    if mnk not in EXPECTED_SHAPES:
        raise CanonicalTemplateError(f"{request_id}: shape is outside canonical family")
    expected = {
        "operator": "scaled_quant_gemm",
        "input": "int8",
        "weight": "int8",
        "accum": "int32",
        "output": "bf16",
        "layout": "TN",
    }
    actual = {
        "operator": contract.get("operator"),
        "input": _dtype(contract, "input"),
        "weight": _dtype(contract, "weight"),
        "accum": _dtype(contract, "accum"),
        "output": _dtype(contract, "output"),
        "layout": contract.get("layout"),
    }
    if actual != expected:
        raise CanonicalTemplateError(
            f"{request_id}: frozen semantics mismatch: {actual!r}"
        )
    if recognized.get("input_scale_granularity") != "per_row":
        raise CanonicalTemplateError(f"{request_id}: activation scale must be per-row")
    if recognized.get("weight_scale_granularity") != "per_column":
        raise CanonicalTemplateError(f"{request_id}: weight scale must be per-column")
    if recognized.get("target_gpu") != "gfx942":
        raise CanonicalTemplateError(f"{request_id}: target must be gfx942")
    return FamilyRequest(
        request_id=request_id,
        request_text=request_text,
        language=language,
        shape=mnk,  # type: ignore[arg-type]
        seed_provenance=dict(seed),
        recognized_contract=dict(recognized),
    )


def load_family_requests(path: Path | str) -> tuple[FamilyRequest, ...]:
    payload = _mapping_file(Path(path).expanduser().resolve(), "generation requests")
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise CanonicalTemplateError("generation requests requires a requests list")
    selected = [
        _parse_family_request(raw)
        for raw in requests
        if isinstance(raw, Mapping)
        and isinstance(raw.get("seed_provenance"), Mapping)
        and raw["seed_provenance"].get("contract_family_id") == FAMILY_ID
    ]
    selected.sort(key=lambda item: item.request_id)
    if len(selected) != 52:
        raise CanonicalTemplateError(
            f"expected 52 requests for {FAMILY_ID}, found {len(selected)}"
        )
    for language in ("hip", "triton"):
        shapes = {item.shape for item in selected if item.language == language}
        if shapes != EXPECTED_SHAPES:
            raise CanonicalTemplateError(
                f"{language} lane does not contain all 26 canonical shapes"
            )
    return tuple(selected)


def verify_locked_source(aiter_root: Path | str) -> None:
    root = Path(aiter_root).expanduser().resolve(strict=True)
    for artifact in SOURCE_ARTIFACTS:
        candidate = root / str(artifact["path"])
        if candidate.is_symlink() or not candidate.is_file():
            raise CanonicalTemplateError(
                f"locked AITER artifact is missing: {artifact['path']}"
            )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise CanonicalTemplateError(
                f"unsafe AITER artifact path: {artifact['path']}"
            ) from exc
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != artifact["sha256"]:
            raise CanonicalTemplateError(
                f"locked AITER artifact hash mismatch: {artifact['path']}"
            )


def _hip_kernel_source() -> str:
    return r'''from __future__ import annotations

import hashlib
import os

import torch
from torch.utils.cpp_extension import load_inline


_HIP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 32;

__global__ void scaled_quant_gemm_i8_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ w,
    const float* __restrict__ a_scale,
    const float* __restrict__ w_scale,
    __hip_bfloat16* __restrict__ out,
    int64_t M,
    int64_t N,
    int64_t K) {
  __shared__ int8_t as[TILE_M][TILE_K];
  __shared__ int8_t ws[TILE_N][TILE_K];
  const int lm = threadIdx.y;
  const int ln = threadIdx.x;
  const int64_t m = static_cast<int64_t>(blockIdx.y) * TILE_M + lm;
  const int64_t n = static_cast<int64_t>(blockIdx.x) * TILE_N + ln;
  int32_t acc = 0;
  for (int64_t kb = 0; kb < K; kb += TILE_K) {
    for (int kk = ln; kk < TILE_K; kk += TILE_N) {
      as[lm][kk] = (m < M && kb + kk < K) ? a[m * K + kb + kk] : 0;
    }
    for (int kk = lm; kk < TILE_K; kk += TILE_M) {
      ws[ln][kk] = (n < N && kb + kk < K) ? w[n * K + kb + kk] : 0;
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      acc += static_cast<int32_t>(as[lm][kk]) *
             static_cast<int32_t>(ws[ln][kk]);
    }
    __syncthreads();
  }
  if (m < M && n < N) {
    const float scaled =
        static_cast<float>(acc) * a_scale[m] * w_scale[n];
    out[m * N + n] = __float2bfloat16(scaled);
  }
}

torch::Tensor scaled_quant_gemm_i8(
    torch::Tensor a,
    torch::Tensor w,
    torch::Tensor a_scale,
    torch::Tensor w_scale) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && a_scale.is_cuda() && w_scale.is_cuda(),
              "all inputs must be HIP tensors");
  TORCH_CHECK(a.scalar_type() == torch::kInt8 && w.scalar_type() == torch::kInt8,
              "activation and weight must be int8");
  TORCH_CHECK(a_scale.scalar_type() == torch::kFloat32 &&
              w_scale.scalar_type() == torch::kFloat32,
              "scales must be float32");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2, "activation and weight must be rank two");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() &&
              a_scale.is_contiguous() && w_scale.is_contiguous(),
              "all inputs must be contiguous");
  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "weight K must match activation K");
  TORCH_CHECK(a_scale.numel() == M, "activation scale must have M elements");
  TORCH_CHECK(w_scale.numel() == N, "weight scale must have N elements");
  TORCH_CHECK(M > 0 && N > 0 && K > 0, "M/N/K must be positive");
  auto out = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));
  const dim3 block(TILE_N, TILE_M);
  const dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  hipLaunchKernelGGL(
      scaled_quant_gemm_i8_kernel, grid, block, 0, at::hip::getCurrentHIPStream(),
      a.data_ptr<int8_t>(), w.data_ptr<int8_t>(),
      a_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()), M, N, K);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "scaled_quant_gemm_i8 launch failed");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scaled_quant_gemm_i8", &scaled_quant_gemm_i8,
        "standalone gfx942 int8 TN scaled GEMM");
}
"""

_EXTENSION = None


def _extension():
    global _EXTENSION
    if _EXTENSION is None:
        digest = hashlib.sha256(_HIP_SOURCE.encode("utf-8")).hexdigest()[:16]
        _EXTENSION = load_inline(
            name="geak_scaled_quant_gemm_" + digest,
            cpp_sources="",
            cuda_sources=_HIP_SOURCE,
            functions=None,
            extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
            with_cuda=True,
            verbose=os.environ.get("GEAK_VERBOSE_BUILD") == "1",
        )
    return _EXTENSION


def scaled_quant_gemm(a, weight, activation_scale, weight_scale):
    return _extension().scaled_quant_gemm_i8(
        a, weight, activation_scale, weight_scale
    )
'''


def _triton_kernel_source() -> str:
    return r'''from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scaled_quant_gemm_i8_kernel(
    a_ptr,
    w_ptr,
    a_scale_ptr,
    w_scale_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = k_start * BLOCK_K + offs_k < K
        a = tl.load(
            a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0
        )
        w = tl.load(
            w_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0
        )
        accumulator += tl.dot(a, w, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    row_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    column_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    scaled = accumulator.to(tl.float32) * row_scale[:, None] * column_scale[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        scaled.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def scaled_quant_gemm(a, weight, activation_scale, weight_scale):
    if not (
        a.is_cuda and weight.is_cuda and activation_scale.is_cuda and weight_scale.is_cuda
    ):
        raise ValueError("all inputs must be HIP tensors")
    if a.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("activation and weight must be int8")
    if activation_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        raise TypeError("scales must be float32")
    if a.ndim != 2 or weight.ndim != 2 or a.shape[1] != weight.shape[1]:
        raise ValueError("expected activation [M,K] and weight [N,K]")
    if not (
        a.is_contiguous()
        and weight.is_contiguous()
        and activation_scale.is_contiguous()
        and weight_scale.is_contiguous()
    ):
        raise ValueError("all inputs must be contiguous")
    M, K = a.shape
    N = weight.shape[0]
    if activation_scale.numel() != M or weight_scale.numel() != N:
        raise ValueError("scale lengths must be M and N")
    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    _scaled_quant_gemm_i8_kernel[grid](
        a,
        weight,
        activation_scale,
        weight_scale,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return out
'''


def _runner_source(shape: tuple[int, int, int]) -> str:
    return f'''from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch


TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))
SHAPE = {shape!r}
SUPPORTED_ARCH = "gfx942"
WARMUP = 1
REPEATS = 3
MAX_REFERENCE_ELEMENTS = 16 * 1024 * 1024
RTOL = 0.0
ATOL = 0.0


def require_gpu_arch():
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    raw = str(torch.cuda.get_device_properties(0).gcnArchName)
    arch = raw.split(":", 1)[0]
    if arch != SUPPORTED_ARCH:
        raise RuntimeError("expected gfx942, found " + arch)


def load_candidate():
    return importlib.import_module("kernel").scaled_quant_gemm


def make_case(seed):
    torch.manual_seed(seed)
    M, N, K = SHAPE
    a = torch.randint(-8, 9, (M, K), device="cuda", dtype=torch.int8)
    weight = torch.randint(-8, 9, (N, K), device="cuda", dtype=torch.int8)
    activation_scale = torch.rand((M,), device="cuda", dtype=torch.float32) * 0.125
    weight_scale = torch.rand((N,), device="cuda", dtype=torch.float32) * 0.125
    return a, weight, activation_scale, weight_scale


def torch_reference(a, weight, activation_scale, weight_scale):
    # With values in [-8,8] and K <= 8192, fp32 GEMM is exact for the integer
    # dot product.  The explicit int32 round-trip pins accumulation semantics.
    accumulator = torch.matmul(a.float(), weight.float().transpose(0, 1)).to(torch.int32)
    scaled = (
        accumulator.float()
        * activation_scale.reshape(-1, 1)
        * weight_scale.reshape(1, -1)
    )
    return scaled.to(torch.bfloat16)


def compile_kernel(candidate):
    candidate(*make_case(7101))
    torch.cuda.synchronize()


def check_correctness(candidate):
    a, weight, activation_scale, weight_scale = make_case(7102)
    actual = candidate(a, weight, activation_scale, weight_scale)
    M, N, _ = SHAPE
    if actual.shape != (M, N) or actual.dtype != torch.bfloat16:
        raise AssertionError("candidate returned the wrong shape or dtype")
    rows_per_chunk = max(1, MAX_REFERENCE_ELEMENTS // N)
    for start in range(0, M, rows_per_chunk):
        stop = min(M, start + rows_per_chunk)
        expected = torch_reference(
            a[start:stop], weight, activation_scale[start:stop], weight_scale
        )
        torch.testing.assert_close(
            actual[start:stop], expected, rtol=RTOL, atol=ATOL
        )


def benchmark(candidate):
    inputs = make_case(7103)
    for _ in range(WARMUP):
        candidate(*inputs)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    for index in range(REPEATS):
        starts[index].record()
        candidate(*inputs)
        ends[index].record()
    torch.cuda.synchronize()
    samples = sorted(starts[i].elapsed_time(ends[i]) for i in range(REPEATS))
    latency = float(samples[len(samples) // 2])
    if not latency > 0.0:
        raise AssertionError("benchmark produced invalid latency")
    case_id = "m%d-n%d-k%d" % SHAPE
    print("Perf: %.6f ms (%s)" % (latency, case_id))
    build = TASK_DIR / "build"
    build.mkdir(exist_ok=True)
    report = {{
        "test_cases": [
            {{"test_case_id": case_id, "execution_time_ms": latency}}
        ]
    }}
    (build / "performance_report.json").write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    require_gpu_arch()
    candidate = load_candidate()
    if mode == "compile":
        compile_kernel(candidate)
    elif mode == "correctness":
        check_correctness(candidate)
    else:
        benchmark(candidate)


if __name__ == "__main__":
    main()
'''


def _config_source() -> str:
    command = (
        "docker exec -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} "
        '-w "$PWD" ${GEAK_CONTAINER_NAME:-geak-phase1-vllm} '
        "python3 scripts/task_runner.py "
    )
    return yaml.safe_dump(
        {
            "source_file_path": ["kernel.py"],
            "target_kernel_functions": ["scaled_quant_gemm"],
            "compile_command": [command + "compile"],
            "correctness_command": [command + "correctness"],
            "performance_command": [command + "performance"],
            "task_type": "scaled_quant_gemm",
        },
        sort_keys=False,
    )


def render_template(request: FamilyRequest) -> dict[str, str]:
    contract = request.contract
    canonical_metadata = contract.metadata
    canonical_metadata["layout"] = "TN"
    canonical_metadata["accum_dtype"] = "int32"
    canonical_metadata["contract"]["layout"] = "TN"
    canonical_metadata["contract"]["accum_dtype"] = "int32"
    metadata = {
        **canonical_metadata,
        "provenance": {
            "generator": "multi_tune_agent.scaled_quant_gemm_templates",
            "generation_method": "canonical_source_backed_factory",
            "source_request": request.request_text,
            "source_repo": "https://github.com/ROCm/aiter.git",
            "source_sha": LOCKED_AITER_SHA,
            "source_artifacts": [dict(item) for item in SOURCE_ARTIFACTS],
            "case_seed": dict(request.seed_provenance),
            "contract_hash": contract.contract_hash,
            "runtime_dependency": "none",
        },
        "materialization_status": "untrusted_pending_gpu_gate",
    }
    return {
        "kernel.py": (
            _hip_kernel_source() if request.language == "hip" else _triton_kernel_source()
        ),
        "config.yaml": _config_source(),
        "scripts/task_runner.py": _runner_source(request.shape),
        "metadata.json": json.dumps(metadata, sort_keys=True, indent=2) + "\n",
    }


def _atomic_install(directory: Path, files: Mapping[str, str]) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".tmp-{directory.name}-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(mode=0o700)
        for relative in CANONICAL_FILES:
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(files[relative], encoding="utf-8")
        if directory.exists():
            if not directory.is_dir() or directory.is_symlink():
                raise CanonicalTemplateError(f"unsafe candidate destination: {directory}")
            old_hashes = {
                relative: hashlib.sha256((directory / relative).read_bytes()).digest()
                for relative in CANONICAL_FILES
            }
            new_hashes = {
                relative: hashlib.sha256((temporary / relative).read_bytes()).digest()
                for relative in CANONICAL_FILES
            }
            if old_hashes != new_hashes:
                raise CanonicalTemplateError(
                    f"deterministic candidate conflicts with existing {directory}"
                )
            return
        os.rename(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def materialize_request(
    request: FamilyRequest, candidate_root: Path | str
) -> TemplateDraft:
    contract = request.contract
    destination = Path(candidate_root).expanduser().resolve() / contract.contract_hash
    _atomic_install(destination, render_template(request))
    report = validate_generated_template(destination, contract.expected_contract)
    if not report.valid:
        raise CanonicalTemplateError(
            "canonical candidate failed static validation: "
            + "; ".join(str(issue) for issue in report.errors)
        )
    return TemplateDraft(
        destination,
        contract,
        report,
        "canonical_source_backed_factory",
        tuple(SOURCE_ARTIFACTS),
    )


def _selected(
    requests: Sequence[FamilyRequest], shard_index: int, shard_count: int
) -> list[FamilyRequest]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise CanonicalTemplateError("shard index must be in [0, shard count)")
    return [
        request
        for index, request in enumerate(requests)
        if index % shard_count == shard_index
    ]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(dict(payload), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _task_record(
    request: FamilyRequest, draft: TemplateDraft, promoted: Path
) -> dict[str, Any]:
    metadata = json.loads((promoted / "metadata.json").read_text(encoding="utf-8"))
    return {
        "id": request.request_id,
        "type": "aiter_generated",
        "kernel_path": str(promoted),
        "direction": request.request_text,
        "operator": draft.contract.operator,
        "backend": request.language,
        "architecture": "gfx942",
        "contract_hash": draft.contract_hash,
        "provenance": dict(metadata["provenance"]),
        "recognized_contract": dict(request.recognized_contract),
    }


def merge_catalog(
    output_path: Path, records: Sequence[Mapping[str, Any]], base_catalog: Path | None
) -> None:
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output_path.is_file():
            payload = _mapping_file(output_path, "output catalog")
        elif base_catalog is not None:
            payload = _mapping_file(base_catalog, "base catalog")
        else:
            payload = {"version": 1, "tasks": []}
        tasks = {
            str(task["id"]): dict(task)
            for task in payload.get("tasks", [])
            if isinstance(task, Mapping) and task.get("id")
        }
        for raw in records:
            record = dict(raw)
            task_id = str(record["id"])
            existing = tasks.get(task_id)
            if existing is not None and existing != record:
                raise CanonicalTemplateError(
                    f"catalog already has conflicting task {task_id}"
                )
            metadata = json.loads(
                (Path(record["kernel_path"]) / "metadata.json").read_text(encoding="utf-8")
            )
            if metadata.get("trust", {}).get("trusted") is not True:
                raise CanonicalTemplateError(f"refusing to catalog untrusted {task_id}")
            tasks[task_id] = record
        payload["tasks"] = [tasks[key] for key in sorted(tasks)]
        _atomic_write_yaml(output_path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "materialize", "gate"))
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--aiter-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--verified-root", type=Path)
    parser.add_argument("--output-catalog", type=Path)
    parser.add_argument("--base-catalog", type=Path)
    parser.add_argument("--geak-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--command-timeout", type=int, default=1800)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requests = load_family_requests(args.requests)
    selected = _selected(requests, args.shard_index, args.shard_count)
    summary: dict[str, Any] = {
        "contract_family_id": FAMILY_ID,
        "requests": len(requests),
        "hip": sum(item.language == "hip" for item in requests),
        "triton": sum(item.language == "triton" for item in requests),
        "selected": len(selected),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    if args.mode == "plan":
        if args.aiter_root is not None:
            verify_locked_source(args.aiter_root)
            summary["source_verified"] = True
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.candidate_root is None:
        raise CanonicalTemplateError("materialize and gate require --candidate-root")
    if args.aiter_root is None:
        raise CanonicalTemplateError("materialize and gate require --aiter-root")
    verify_locked_source(args.aiter_root)
    summary["source_verified"] = True
    if args.mode == "materialize":
        for request in selected:
            materialize_request(request, args.candidate_root)
        summary["materialized"] = len(selected)
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
        raise CanonicalTemplateError("gate requires " + ", ".join(missing))
    output_catalog = args.output_catalog.expanduser().resolve()
    existing_ids: set[str] = set()
    if output_catalog.is_file():
        existing_ids = {
            str(task.get("id"))
            for task in _mapping_file(output_catalog, "output catalog").get("tasks", [])
            if isinstance(task, Mapping)
        }
    selected = [item for item in selected if item.request_id not in existing_ids]
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    diagnostics_root = args.run_root.expanduser().resolve() / "gate-results"
    for request in selected:
        draft = materialize_request(request, args.candidate_root)
        result = run_template_gpu_gate(
            draft,
            geak_root=args.geak_root,
            run_root=args.run_root,
            gpu_ids=args.gpu_id,
            command_timeout=args.command_timeout,
        )
        diagnostic = {
            "case_id": request.request_id,
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
        _atomic_write_json(diagnostics_root / f"{request.request_id}.json", diagnostic)
        if not result.trusted:
            failures.append(request.request_id)
            continue
        promoted = promote_validated_template(draft, result, args.verified_root)
        records.append(_task_record(request, draft, promoted))
        merge_catalog(output_catalog, [records[-1]], args.base_catalog)
    summary["resumed"] = len(existing_ids)
    summary["promoted"] = len(records)
    summary["failed_gate"] = failures
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CanonicalTemplateError",
    "EXPECTED_SHAPES",
    "FAMILY_ID",
    "FamilyRequest",
    "load_family_requests",
    "main",
    "materialize_request",
    "render_template",
    "verify_locked_source",
]
