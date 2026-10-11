"""Source-locked canonical GPU gates for the split-v4 Top10 families.

Materialization is deterministic and never confers trust.  A bundle is only
catalogued after a fresh GEAK compile/correctness/performance gate records
``trust.trusted=true``.  Multi-process all-reduce is deliberately fail-closed
until it can be run by a gang launcher holding an exclusive GPU reservation.
"""

from __future__ import annotations

import argparse
import contextlib
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
    TemplateGateResult,
    promote_validated_template,
    run_template_gpu_gate,
)
from .top10_attention_templates import (
    render_attention_kernel,
    render_attention_runner,
)
from .top10_dense_templates import render_dense_kernel, render_dense_runner
from .top10_sequence_templates import (
    render_sequence_kernel,
    render_sequence_runner,
)


LOCKED_AITER_SHA = "926eb3d059efd3c866c8f53ecb8b1fb8fb7135e8"
SUPPORTED_ARCH = "gfx942"
TOP10_FAMILIES = (
    "mha",
    "mla",
    "paged_attention",
    "fused_moe",
    "gemm",
    "rms_norm",
    "rope_kv_cache",
    "blockscale_gemm",
    "sampling",
    "all_reduce",
)
TRITON_FAMILIES = frozenset(TOP10_FAMILIES[:-1])
CANONICAL_FILES = (
    "kernel.py",
    "config.yaml",
    "scripts/task_runner.py",
    "metadata.json",
)
FORBIDDEN_RUNTIME_TOKENS = (
    "aiter",
    "composable_kernel",
    "hipblaslt",
    "torch_reference",
    "reference(",
)
FAIL_CLOSED_REASONS: dict[str, str] = {}


class Top10CanonicalError(ValueError):
    """A request or gate operation violates the frozen Top10 contract."""


@dataclass(frozen=True)
class Top10Request:
    request_id: str
    request_text: str
    family: str
    language: str
    shape: tuple[int, ...]
    seed_provenance: Mapping[str, Any]
    recognized_contract: Mapping[str, Any]

    @property
    def contract(self) -> KernelContract:
        frozen = self.recognized_contract.get("contract")
        assert isinstance(frozen, Mapping)
        dtype = frozen.get("dtype")
        dtype = dtype if isinstance(dtype, Mapping) else {}
        return KernelContract(
            operator=self.family,
            request=self.request_text,
            target_gpu=SUPPORTED_ARCH,
            architecture=SUPPORTED_ARCH,
            language=self.language,
            input_dtype=str(
                dtype.get("input") or frozen.get("input_dtype") or "bf16"
            ),
            weight_dtype=(
                str(dtype.get("weight") or frozen.get("weight_dtype"))
                if dtype.get("weight") or frozen.get("weight_dtype")
                else None
            ),
            output_dtype=str(
                dtype.get("output") or frozen.get("output_dtype") or "bf16"
            ),
            shapes=(self.shape,),
        )


def _mapping_file(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise Top10CanonicalError(f"{label} root must be a mapping")
    return value


def _normal_family(value: object) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "attention": "mha",
        "fmha": "mha",
        "multi_head_attention": "mha",
        "multi_latent_attention": "mla",
        "moe": "fused_moe",
        "rmsnorm": "rms_norm",
        "rope": "rope_kv_cache",
        "fused_bmm_rope_kv_cache": "rope_kv_cache",
        "block_scale_gemm": "blockscale_gemm",
        "topk_topp_sampling": "sampling",
        "hip_all_reduce": "all_reduce",
        "quick_all_reduce": "all_reduce",
    }
    return aliases.get(family, family)


def _shape(contract: Mapping[str, Any]) -> tuple[int, ...]:
    raw = contract.get("shape")
    if raw is None:
        shapes = contract.get("shapes")
        if isinstance(shapes, list) and shapes:
            raw = shapes[0]
    if isinstance(raw, Mapping):
        ordered = list(raw.values())
        values = raw.get("dims") if isinstance(raw.get("dims"), list) else ordered
    else:
        values = raw
    if not isinstance(values, (list, tuple)) or not values:
        raise Top10CanonicalError("frozen contract requires a non-empty shape")
    if any(type(value) is not int or value <= 0 for value in values):
        raise Top10CanonicalError("shape dimensions must be positive integers")
    return tuple(values)


def _source_artifacts(seed: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    raw = seed.get("source_artifacts")
    artifacts: list[dict[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and item.get("path") and item.get("sha256"):
                artifacts.append(
                    {"path": str(item["path"]), "sha256": str(item["sha256"])}
                )
    path = seed.get("source_test_path")
    digest = seed.get("source_test_sha256")
    if path and digest:
        artifacts.append({"path": str(path), "sha256": str(digest)})
    if not artifacts:
        raise Top10CanonicalError("source provenance has no hash-locked artifact")
    unique = {(item["path"], item["sha256"]): item for item in artifacts}
    return tuple(unique[key] for key in sorted(unique))


def verify_locked_source(
    aiter_root: Path | str, requests: Sequence[Top10Request] = ()
) -> None:
    root = Path(aiter_root).expanduser().resolve(strict=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != LOCKED_AITER_SHA:
        raise Top10CanonicalError(
            f"locked AITER revision mismatch: expected {LOCKED_AITER_SHA}, got {revision}"
        )
    artifacts = {
        (artifact["path"], artifact["sha256"])
        for request in requests
        for artifact in _source_artifacts(request.seed_provenance)
    }
    for relative, expected in sorted(artifacts):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise Top10CanonicalError(f"locked AITER artifact is missing: {relative}")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise Top10CanonicalError(f"unsafe AITER artifact path: {relative}") from exc
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise Top10CanonicalError(
                f"locked AITER artifact hash mismatch: {relative}"
            )


def _parse_request(raw: Mapping[str, Any]) -> Top10Request:
    request_id = str(raw.get("id") or "").strip()
    request_text = str(raw.get("request") or "").strip()
    recognized = raw.get("recognized_contract")
    seed = raw.get("seed_provenance")
    family = _normal_family(raw.get("top10_family"))
    if not family and isinstance(seed, Mapping):
        family = _normal_family(seed.get("top10_family"))
    if not family and isinstance(recognized, Mapping):
        family = _normal_family(recognized.get("top10_family"))
    if family not in TOP10_FAMILIES:
        raise Top10CanonicalError(f"{request_id}: unsupported top10_family {family!r}")
    if not request_id or not request_text:
        raise Top10CanonicalError("Top10 request requires id and request text")
    if not isinstance(recognized, Mapping) or not isinstance(seed, Mapping):
        raise Top10CanonicalError(f"{request_id}: missing contract or provenance")
    if seed.get("source_sha") != LOCKED_AITER_SHA:
        raise Top10CanonicalError(f"{request_id}: unexpected AITER source revision")
    if seed.get("split_version") != "v4":
        raise Top10CanonicalError(f"{request_id}: Top10 requests must come from split v4")
    if seed.get("split_group") not in {"train", "dev"}:
        raise Top10CanonicalError(f"{request_id}: held-out requests may not materialize")
    if str(recognized.get("target_gpu") or "").lower() != SUPPORTED_ARCH:
        raise Top10CanonicalError(f"{request_id}: target must be gfx942")
    language = str(recognized.get("language") or "").lower()
    supported_languages = {"hip"} if family == "all_reduce" else {"hip", "triton"}
    if language not in supported_languages:
        raise Top10CanonicalError(
            f"{request_id}: unsupported {family} canonical lane {language!r}"
        )
    contract = recognized.get("contract")
    if not isinstance(contract, Mapping):
        raise Top10CanonicalError(f"{request_id}: missing frozen contract")
    operator = _normal_family(contract.get("operator"))
    if operator != family:
        raise Top10CanonicalError(
            f"{request_id}: top10_family/operator mismatch ({family}/{operator})"
        )
    _source_artifacts(seed)
    return Top10Request(
        request_id=request_id,
        request_text=request_text,
        family=family,
        language=language,
        shape=_shape(contract),
        seed_provenance=dict(seed),
        recognized_contract=dict(recognized),
    )


def load_top10_requests(path: Path | str) -> tuple[Top10Request, ...]:
    payload = _mapping_file(Path(path).expanduser().resolve(), "generation requests")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        raise Top10CanonicalError("generation requests requires a requests list")
    selected = []
    seen: set[str] = set()
    for raw in raw_requests:
        if not isinstance(raw, Mapping):
            continue
        has_family = raw.get("top10_family")
        seed = raw.get("seed_provenance")
        recognized = raw.get("recognized_contract")
        has_family = has_family or (
            isinstance(seed, Mapping) and seed.get("top10_family")
        )
        has_family = has_family or (
            isinstance(recognized, Mapping) and recognized.get("top10_family")
        )
        if not has_family:
            continue
        item = _parse_request(raw)
        if item.request_id in seen:
            raise Top10CanonicalError(f"duplicate request ID: {item.request_id}")
        seen.add(item.request_id)
        selected.append(item)
    selected.sort(key=lambda item: item.request_id)
    return tuple(selected)


def family_counts(requests: Sequence[Top10Request]) -> dict[str, int]:
    counts = {
        family: sum(item.family == family for item in requests)
        for family in TOP10_FAMILIES
    }
    counts["total"] = len(requests)
    return counts


def _triton_kernel(family: str) -> str:
    function = family
    if family in {"gemm", "blockscale_gemm", "fused_moe"}:
        scale_args = ", row_scale, col_scale" if family == "blockscale_gemm" else ""
        postprocess = (
            "\n    acc *= tl.load(row_scale + rows)[:, None]"
            "\n    acc *= tl.load(col_scale + cols)[None, :]"
            if family == "blockscale_gemm"
            else (
                "\n    acc = tl.maximum(acc, 0.0)"
                if family == "fused_moe"
                else ""
            )
        )
        wrapper_args = ", row_scale, col_scale" if family == "blockscale_gemm" else ""
        call_args = ", row_scale, col_scale" if family == "blockscale_gemm" else ""
        return f'''import torch
import triton
import triton.language as tl

@triton.jit
def _{function}_kernel(a, w, out{scale_args}, M: tl.constexpr, N: tl.constexpr,
                       K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rows, cols, kk = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN), tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for base in range(0, tl.cdiv(K, BK)):
        kval = base * BK + kk
        av = tl.load(a + rows[:, None] * K + kval[None, :], mask=(rows[:, None] < M) & (kval[None, :] < K), other=0.0)
        wv = tl.load(w + cols[None, :] * K + kval[:, None], mask=(cols[None, :] < N) & (kval[:, None] < K), other=0.0)
        acc += tl.dot(av, wv){postprocess}
    tl.store(out + rows[:, None] * N + cols[None, :], acc, mask=(rows[:, None] < M) & (cols[None, :] < N))

def {function}(a, w{wrapper_args}):
    M, K = a.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    _{function}_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 32))](a, w, out{call_args}, M, N, K, BM=32, BN=32, BK=32)
    return out
'''
    if family == "rms_norm":
        return '''import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(x, weight, out, D: tl.constexpr, BLOCK: tl.constexpr):
    row, cols = tl.program_id(0), tl.arange(0, BLOCK)
    mask = cols < D
    value = tl.load(x + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, axis=0) / D + 1.0e-5)
    tl.store(out + row * D + cols, value * inv * tl.load(weight + cols, mask=mask), mask=mask)

def rms_norm(x, weight):
    out = torch.empty_like(x)
    block = triton.next_power_of_2(x.shape[-1])
    _rms_norm_kernel[(x.numel() // x.shape[-1],)](x, weight, out, D=x.shape[-1], BLOCK=block)
    return out
'''
    if family == "rope_kv_cache":
        return '''import torch
import triton
import triton.language as tl

@triton.jit
def _rope_kv_cache_kernel(q, k, cos, sin, qo, ko, n: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    row, d = tl.program_id(0), tl.arange(0, BLOCK)
    mask = d < D
    partner = tl.where(d < D // 2, d + D // 2, d - D // 2)
    sign = tl.where(d < D // 2, -1.0, 1.0)
    c, s = tl.load(cos + d, mask=mask), tl.load(sin + d, mask=mask)
    qv, kv = tl.load(q + row * D + d, mask=mask), tl.load(k + row * D + d, mask=mask)
    qr, kr = tl.load(q + row * D + partner, mask=mask), tl.load(k + row * D + partner, mask=mask)
    tl.store(qo + row * D + d, qv * c + sign * qr * s, mask=mask)
    tl.store(ko + row * D + d, kv * c + sign * kr * s, mask=mask)

def rope_kv_cache(q, k, cos, sin):
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    d = q.shape[-1]
    _rope_kv_cache_kernel[(q.numel() // d,)](q, k, cos, sin, qo, ko, q.numel(), D=d, BLOCK=triton.next_power_of_2(d))
    return qo, ko
'''
    if family == "sampling":
        return '''import torch
import triton
import triton.language as tl

@triton.jit
def _sampling_kernel(logits, out, V: tl.constexpr, BLOCK: tl.constexpr):
    row, col = tl.program_id(0), tl.arange(0, BLOCK)
    values = tl.load(logits + row * V + col, mask=col < V, other=-float("inf"))
    tl.store(out + row, tl.argmax(values, axis=0))

def sampling(logits):
    out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int32)
    _sampling_kernel[(logits.shape[0],)](logits, out, V=logits.shape[1], BLOCK=triton.next_power_of_2(logits.shape[1]))
    return out
'''
    # MHA, MLA, and paged attention share an independently written direct kernel.
    return f'''import torch
import triton
import triton.language as tl

@triton.jit
def _{function}_kernel(q, k, v, out, S: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    row, d = tl.program_id(0), tl.arange(0, BLOCK)
    mask = d < D
    qv = tl.load(q + row * D + d, mask=mask, other=0.0).to(tl.float32)
    denom = 0.0
    numer = tl.zeros((BLOCK,), tl.float32)
    for token in range(0, S):
        kval = tl.load(k + token * D + d, mask=mask, other=0.0).to(tl.float32)
        score = tl.sum(qv * kval, axis=0) * 0.125
        weight = tl.exp(score)
        numer += weight * tl.load(v + token * D + d, mask=mask, other=0.0)
        denom += weight
    tl.store(out + row * D + d, numer / denom, mask=mask)

def {function}(q, k, v):
    out = torch.empty_like(q)
    _{function}_kernel[(q.shape[0],)](q, k, v, out, S=k.shape[0], D=q.shape[1], BLOCK=triton.next_power_of_2(q.shape[1]))
    return out
'''


def _hip_kernel(family: str) -> str:
    """Render small, standalone HIP implementations for the requested lane."""
    if family in {"gemm", "fused_moe", "blockscale_gemm"}:
        scaled = family == "blockscale_gemm"
        activated = family == "fused_moe"
        scalar_type = "int8_t" if scaled else "__hip_bfloat16"
        convert = "static_cast<float>" if scaled else "__bfloat162float"
        extra_parameters = (
            ", const float* row_scale, const float* col_scale" if scaled else ""
        )
        post = (
            "value *= row_scale[row] * col_scale[col];"
            if scaled
            else ("value = value > 0.0f ? value : 0.0f;" if activated else "")
        )
        python_extra = ", row_scale, col_scale" if scaled else ""
        checks = (
            'TORCH_CHECK(a.scalar_type() == at::kChar && w.scalar_type() == at::kChar, "inputs must be int8");'
            if scaled
            else 'TORCH_CHECK(a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "inputs must be bf16");'
        )
        data_extra = (
            ", row_scale.data_ptr<float>(), col_scale.data_ptr<float>()"
            if scaled
            else ""
        )
        signature_extra = ", torch::Tensor row_scale, torch::Tensor col_scale" if scaled else ""
        return f'''import hashlib
import torch
from torch.utils.cpp_extension import load_inline
_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
__global__ void {family}_kernel(const {scalar_type}* a, const {scalar_type}* w,
    __hip_bfloat16* out, int M, int N, int K{extra_parameters}) {{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float value = 0.0f;
  for (int k = 0; k < K; ++k)
    value += {convert}(a[row * K + k]) * {convert}(w[col * K + k]);
  {post}
  out[row * N + col] = __float2bfloat16(value);
}}
torch::Tensor {family}(torch::Tensor a, torch::Tensor w{signature_extra}) {{
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && a.dim() == 2 && w.dim() == 2, "invalid HIP inputs");
  {checks}
  int M = a.size(0), K = a.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "K mismatch");
  auto out = torch::empty({{M, N}}, a.options().dtype(at::kBFloat16));
  dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
  hipLaunchKernelGGL({family}_kernel, grid, block, 0, at::hip::getCurrentHIPStream(),
    reinterpret_cast<const {scalar_type}*>(a.data_ptr()),
    reinterpret_cast<const {scalar_type}*>(w.data_ptr()),
    reinterpret_cast<__hip_bfloat16*>(out.data_ptr()), M, N, K{data_extra});
  TORCH_CHECK(hipGetLastError() == hipSuccess, "HIP launch failed");
  return out;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def("{family}", &{family}); }}
"""
_MOD = None
def _module():
    global _MOD
    if _MOD is None:
        name = "geak_top10_{family}_" + hashlib.sha256(_SOURCE.encode()).hexdigest()[:12]
        _MOD = load_inline(name=name, cpp_sources="", cuda_sources=_SOURCE, functions=None,
                           extra_cuda_cflags=["-O3", "--offload-arch=gfx942"], with_cuda=True)
    return _MOD
def {family}(a, w{python_extra}):
    return _module().{family}(a, w{python_extra})
'''
    if family == "rms_norm":
        body = """
  __shared__ float sums[256];
  float local = 0.0f;
  for (int d = col; d < D; d += blockDim.x) {
    float value = __bfloat162float(x[row * D + d]); local += value * value;
  }
  sums[threadIdx.x] = local; __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(sums[0] / D + 1.0e-5f);
  for (int d = col; d < D; d += blockDim.x)
    out[row * D + d] = __float2bfloat16(
      __bfloat162float(x[row * D + d]) * inv * __bfloat162float(weight[d]));"""
        wrapper_args = "x, weight"
        signature = "torch::Tensor x, torch::Tensor weight"
        launch = "reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()), reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()), reinterpret_cast<__hip_bfloat16*>(out.data_ptr()), x.size(0), x.size(1)"
        output = "auto out = torch::empty_like(x);"
        grid = "dim3(x.size(0)), dim3(256)"
    elif family == "sampling":
        body = """
  if (threadIdx.x || blockIdx.x >= B) return;
  int best = 0; float value = logits[blockIdx.x * V];
  for (int i = 1; i < V; ++i) if (logits[blockIdx.x * V + i] > value) {
    value = logits[blockIdx.x * V + i]; best = i;
  }
  out[blockIdx.x] = best;"""
        wrapper_args = "logits"
        signature = "torch::Tensor logits"
        launch = "logits.data_ptr<float>(), out.data_ptr<int32_t>(), logits.size(0), logits.size(1)"
        output = "auto out = torch::empty({logits.size(0)}, logits.options().dtype(at::kInt));"
        grid = "dim3(logits.size(0)), dim3(1)"
    elif family == "rope_kv_cache":
        body = """
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return; int d = i % D;
  int partner = d < D / 2 ? i + D / 2 : i - D / 2;
  float sign = d < D / 2 ? -1.0f : 1.0f;
  qo[i] = __float2bfloat16(__bfloat162float(q[i]) * cos[d] + sign * __bfloat162float(q[partner]) * sin[d]);
  ko[i] = __float2bfloat16(__bfloat162float(k[i]) * cos[d] + sign * __bfloat162float(k[partner]) * sin[d]);"""
        wrapper_args = "q, k, cos, sin"
        signature = "torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin"
        launch = "reinterpret_cast<const __hip_bfloat16*>(q.data_ptr()), reinterpret_cast<const __hip_bfloat16*>(k.data_ptr()), cos.data_ptr<float>(), sin.data_ptr<float>(), reinterpret_cast<__hip_bfloat16*>(first.data_ptr()), reinterpret_cast<__hip_bfloat16*>(second.data_ptr()), q.numel(), q.size(-1)"
        output = "auto first = torch::empty_like(q); auto second = torch::empty_like(k);"
        grid = "dim3((q.numel() + 255) / 256), dim3(256)"
    else:
        body = """
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Q * D) return; int row = i / D, d = i % D;
  float numerator = 0.0f, denominator = 0.0f;
  for (int token = 0; token < S; ++token) {
    float score = 0.0f;
    for (int j = 0; j < D; ++j)
      score += __bfloat162float(q[row * D + j]) * __bfloat162float(k[token * D + j]);
    float weight = expf(score * 0.125f);
    numerator += weight * __bfloat162float(v[token * D + d]); denominator += weight;
  }
  out[i] = __float2bfloat16(numerator / denominator);"""
        wrapper_args = "q, k, v"
        signature = "torch::Tensor q, torch::Tensor k, torch::Tensor v"
        launch = "reinterpret_cast<const __hip_bfloat16*>(q.data_ptr()), reinterpret_cast<const __hip_bfloat16*>(k.data_ptr()), reinterpret_cast<const __hip_bfloat16*>(v.data_ptr()), reinterpret_cast<__hip_bfloat16*>(out.data_ptr()), q.size(0), k.size(0), q.size(1)"
        output = "auto out = torch::empty_like(q);"
        grid = "dim3((q.numel() + 255) / 256), dim3(256)"
    if family == "rms_norm":
        c_signature = "const __hip_bfloat16* x, const __hip_bfloat16* weight, __hip_bfloat16* out, int B, int D"
        prefix = "int row = blockIdx.x, col = threadIdx.x;"
        result = "return py::cast(out);"
    elif family == "sampling":
        c_signature = "const float* logits, int32_t* out, int B, int V"
        prefix = ""
        result = "return py::cast(out);"
    elif family == "rope_kv_cache":
        c_signature = "const __hip_bfloat16* q, const __hip_bfloat16* k, const float* cos, const float* sin, __hip_bfloat16* qo, __hip_bfloat16* ko, int64_t n, int D"
        prefix = ""
        result = "return py::make_tuple(first, second);"
    else:
        c_signature = "const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v, __hip_bfloat16* out, int Q, int S, int D"
        prefix = ""
        result = "return py::cast(out);"
    return f'''import hashlib
import torch
from torch.utils.cpp_extension import load_inline
_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
__global__ void {family}_kernel({c_signature}) {{
  {prefix}{body}
}}
py::object {family}({signature}) {{
  {output}
  hipLaunchKernelGGL({family}_kernel, {grid}, 0, at::hip::getCurrentHIPStream(), {launch});
  TORCH_CHECK(hipGetLastError() == hipSuccess, "HIP launch failed");
  {result}
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def("{family}", &{family}); }}
"""
_MOD = None
def _module():
    global _MOD
    if _MOD is None:
        name = "geak_top10_{family}_" + hashlib.sha256(_SOURCE.encode()).hexdigest()[:12]
        _MOD = load_inline(name=name, cpp_sources="", cuda_sources=_SOURCE, functions=None,
                           extra_cuda_cflags=["-O3", "--offload-arch=gfx942"], with_cuda=True)
    return _MOD
def {family}({wrapper_args}):
    return _module().{family}({wrapper_args})
'''


def _hip_all_reduce_kernel() -> str:
    return r'''import hashlib
import torch
from torch.utils.cpp_extension import load_inline

_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <rccl/rccl.h>
#include <cstring>
#include <mutex>

#define RCCL_CHECK(expr) do { \
  ncclResult_t status = (expr); \
  TORCH_CHECK(status == ncclSuccess, "RCCL failure: ", ncclGetErrorString(status)); \
} while (0)

py::bytes unique_id() {
  ncclUniqueId id;
  RCCL_CHECK(ncclGetUniqueId(&id));
  return py::bytes(id.internal, NCCL_UNIQUE_ID_BYTES);
}

torch::Tensor all_reduce(torch::Tensor input, py::bytes serialized_id,
                         int rank, int world) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16,
              "input must be HIP bf16");
  TORCH_CHECK(world == 2 || world == 4,
              "world size must be 2 or 4");
  TORCH_CHECK(rank >= 0 && rank < world, "invalid rank");
  std::string blob = serialized_id;
  TORCH_CHECK(blob.size() == NCCL_UNIQUE_ID_BYTES, "invalid RCCL unique id");
  ncclUniqueId id;
  std::memcpy(id.internal, blob.data(), NCCL_UNIQUE_ID_BYTES);
  static std::mutex mutex;
  static ncclComm_t communicator = nullptr;
  static int initialized_rank = -1, initialized_world = -1;
  {
    std::lock_guard<std::mutex> guard(mutex);
    if (communicator == nullptr) {
      RCCL_CHECK(ncclCommInitRank(&communicator, world, id, rank));
      initialized_rank = rank;
      initialized_world = world;
    }
  }
  TORCH_CHECK(initialized_rank == rank && initialized_world == world,
              "communicator parameters changed in one process");
  auto output = torch::empty_like(input);
  RCCL_CHECK(ncclAllReduce(input.data_ptr(), output.data_ptr(), input.numel(),
                           ncclBfloat16, ncclSum, communicator,
                           at::hip::getCurrentHIPStream()));
  return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unique_id", &unique_id);
  m.def("all_reduce", &all_reduce);
}
"""
_MODULE = None
def _module():
    global _MODULE
    if _MODULE is None:
        name = "geak_top10_ar_" + hashlib.sha256(_SOURCE.encode()).hexdigest()[:12]
        _MODULE = load_inline(name=name, cpp_sources="", cuda_sources=_SOURCE,
                              functions=None,
                              extra_include_paths=["/opt/rocm/include"],
                              extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
                              extra_ldflags=["-L/opt/rocm/lib", "-lrccl"],
                              with_cuda=True, verbose=False)
    return _MODULE
def compile_extension():
    _module()
def unique_id():
    return _module().unique_id()
def all_reduce(input, communicator_id, rank, world):
    return _module().all_reduce(input, communicator_id, rank, world)
'''


def _runner_source(request: Top10Request) -> str:
    family = request.family
    frozen = request.recognized_contract["contract"]
    assert isinstance(frozen, Mapping)
    dimensions = frozen.get("shape")
    assert isinstance(dimensions, Mapping)
    if family == "all_reduce":
        world_size = int(frozen["world_size"])
        tokens = min(int(dimensions.get("TOKENS", 64)), 128)
        hidden = min(int(dimensions.get("HIDDEN", 4096)), 4096)
        return f'''import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORLD_SIZE = {world_size}
SHAPE = ({tokens}, {hidden})

def require_arch():
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"expected {{WORLD_SIZE}} visible GPUs, found {{torch.cuda.device_count()}}")
    for index in range(WORLD_SIZE):
        arch = torch.cuda.get_device_properties(index).gcnArchName.split(":", 1)[0]
        if arch != "gfx942":
            raise RuntimeError(f"GPU {{index}} is {{arch}}, expected gfx942")

def fp32_int32_oracle():
    return WORLD_SIZE * (WORLD_SIZE + 1) / 2

def worker(rank, communicator_id, mode):
    torch.cuda.set_device(rank)
    kernel = importlib.import_module("kernel")
    value = torch.full(SHAPE, rank + 1, device=f"cuda:{{rank}}", dtype=torch.bfloat16)
    expected_value = fp32_int32_oracle()
    for _ in range(2 if mode == "performance" else 1):
        output = kernel.all_reduce(value, communicator_id, rank, WORLD_SIZE)
    torch.cuda.synchronize(rank)
    if mode == "correctness":
        expected = torch.full(
            SHAPE,
            expected_value,
            device=f"cuda:{{rank}}",
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if mode == "performance":
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(5):
            output = kernel.all_reduce(value, communicator_id, rank, WORLD_SIZE)
        end.record()
        end.synchronize()
        latency = start.elapsed_time(end) / 5.0
        if not latency > 0:
            raise AssertionError("invalid all-reduce latency")
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        (build / f"rank-{{rank}}.json").write_text(json.dumps({{"latency": latency}}))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    torch.manual_seed(9410)
    require_arch()
    kernel = importlib.import_module("kernel")
    if mode == "compile":
        torch.cuda.set_device(0)
        kernel.compile_extension()
        print("Compile: OK")
        return
    communicator_id = kernel.unique_id()
    mp.spawn(worker, args=(communicator_id, mode), nprocs=WORLD_SIZE, join=True)
    if mode == "correctness":
        print("Correctness: OK")
        return
    latencies = [
        json.loads((ROOT / "build" / f"rank-{{rank}}.json").read_text())["latency"]
        for rank in range(WORLD_SIZE)
    ]
    latency = max(latencies)
    report = {{"test_cases": [{{"test_case_id": "all_reduce_{world_size}", "execution_time_ms": latency}}]}}
    (ROOT / "build" / "performance_report.json").write_text(json.dumps(report))
    print(f"Perf: {{latency:.6f}} ms (all_reduce_{world_size})")

if __name__ == "__main__":
    main()
'''
    cases = {
        "gemm": (
            "args = (torch.randn((32, 64), device='cuda', dtype=torch.bfloat16), torch.randn((48, 64), device='cuda', dtype=torch.bfloat16))",
            "expected = torch.matmul(args[0].float(), args[1].float().T).to(torch.bfloat16)",
        ),
        "blockscale_gemm": (
            "args = (torch.randint(-4, 5, (32, 64), device='cuda', dtype=torch.int8), torch.randint(-4, 5, (48, 64), device='cuda', dtype=torch.int8), torch.rand((32,), device='cuda'), torch.rand((48,), device='cuda'))",
            "acc = torch.matmul(args[0].float(), args[1].float().T).to(torch.int32); expected = (acc.float() * args[2][:, None] * args[3][None, :]).to(torch.bfloat16)",
        ),
        "fused_moe": (
            "args = (torch.randn((32, 64), device='cuda', dtype=torch.bfloat16), torch.randn((48, 64), device='cuda', dtype=torch.bfloat16))",
            "expected = torch.relu(torch.matmul(args[0].float(), args[1].float().T)).to(torch.bfloat16)",
        ),
        "rms_norm": (
            "args = (torch.randn((32, 64), device='cuda', dtype=torch.bfloat16), torch.randn((64,), device='cuda', dtype=torch.bfloat16))",
            "xf = args[0].float(); expected = (xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1.0e-5) * args[1].float()).to(torch.bfloat16)",
        ),
        "rope_kv_cache": (
            "q = torch.randn((32, 64), device='cuda', dtype=torch.bfloat16); k = torch.randn_like(q); theta = torch.arange(64, device='cuda').float() / 64; args = (q, k, theta.cos(), theta.sin())",
            "def rot(x): return torch.cat((-x[:, 32:], x[:, :32]), 1); expected = ((q.float() * args[2] + rot(q.float()) * args[3]).to(torch.bfloat16), (k.float() * args[2] + rot(k.float()) * args[3]).to(torch.bfloat16))",
        ),
        "sampling": (
            "args = (torch.randn((16, 128), device='cuda', dtype=torch.float32),)",
            "expected = torch.argmax(args[0].float(), dim=-1).to(torch.int32)",
        ),
    }
    if family == "gemm" and all(key in dimensions for key in ("M", "N", "K")):
        cases["gemm"] = (
            "args = (torch.randn((%d, %d), device='cuda', dtype=torch.bfloat16), torch.randn((%d, %d), device='cuda', dtype=torch.bfloat16))"
            % (
                int(dimensions["M"]),
                int(dimensions["K"]),
                int(dimensions["N"]),
                int(dimensions["K"]),
            ),
            cases["gemm"][1],
        )
    elif family == "rms_norm" and all(key in dimensions for key in ("M", "N")):
        cases["rms_norm"] = (
            "args = (torch.randn((%d, %d), device='cuda', dtype=torch.bfloat16), torch.randn((%d,), device='cuda', dtype=torch.bfloat16))"
            % (
                int(dimensions["M"]),
                int(dimensions["N"]),
                int(dimensions["N"]),
            ),
            cases["rms_norm"][1],
        )
    if family in {"mha", "mla", "paged_attention"}:
        setup = "q = torch.randn((16, 64), device='cuda', dtype=torch.bfloat16); k = torch.randn((16, 64), device='cuda', dtype=torch.bfloat16); v = torch.randn_like(k); args = (q, k, v)"
        oracle = "scores = torch.matmul(q.float(), k.float().T) * 0.125; expected = torch.matmul(torch.softmax(scores, -1), v.float()).to(torch.bfloat16)"
    elif family == "all_reduce":
        setup = "args = (torch.randn((2, 32, 64), device='cuda', dtype=torch.bfloat16),)"
        oracle = "expected = args[0].float().sum(0).to(torch.bfloat16)"
    else:
        setup, oracle = cases[family]
    return f'''import argparse
import importlib
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def require_arch():
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx942":
        raise RuntimeError("expected gfx942, found " + arch)

def make_case():
    torch.manual_seed(9410)
    {setup}
    return args

def fp32_int32_oracle(args):
    {oracle}
    return expected

def check(actual, expected):
    if isinstance(expected, tuple):
        for a, e in zip(actual, expected): torch.testing.assert_close(a, e, rtol=2e-2, atol=2e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    require_arch()
    candidate = getattr(importlib.import_module("kernel"), "{family}")
    args = make_case()
    if mode == "compile":
        candidate(*args); torch.cuda.synchronize(); print("Compile: OK"); return
    if mode == "correctness":
        check(candidate(*args), fp32_int32_oracle(args)); print("Correctness: OK"); return
    for _ in range(2): candidate(*args)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(5): candidate(*args)
    end.record(); end.synchronize()
    latency = start.elapsed_time(end) / 5.0
    if not latency > 0: raise AssertionError("invalid latency")
    build = ROOT / "build"; build.mkdir(exist_ok=True)
    report = {{"test_cases": [{{"test_case_id": "{family}", "execution_time_ms": latency}}]}}
    (build / "performance_report.json").write_text(json.dumps(report), encoding="utf-8")
    print(f"Perf: {{latency:.6f}} ms ({family})")

if __name__ == "__main__":
    main()
'''


def render_template(request: Top10Request) -> dict[str, str]:
    contract = request.contract
    frozen = request.recognized_contract["contract"]
    assert isinstance(frozen, Mapping)
    if request.family in {"mha", "mla", "paged_attention"}:
        kernel = render_attention_kernel(request.family, request.language, frozen)
        runner = render_attention_runner(request.family, frozen)
    elif request.family in {"fused_moe", "blockscale_gemm"}:
        kernel = render_dense_kernel(request.family, request.language, frozen)
        runner = render_dense_runner(request.family, frozen)
    elif request.family in {"rope_kv_cache", "sampling"}:
        kernel = render_sequence_kernel(request.family, request.language, frozen)
        runner = render_sequence_runner(request.family, frozen)
    else:
        kernel = (
            _hip_all_reduce_kernel()
            if request.family == "all_reduce"
            else (
                _hip_kernel(request.family)
                if request.language == "hip"
                else _triton_kernel(request.family)
            )
        )
        runner = _runner_source(request)
    lowered = kernel.lower()
    for token in FORBIDDEN_RUNTIME_TOKENS:
        if token in lowered:
            raise Top10CanonicalError(f"generated runtime contains forbidden token {token}")
    command = (
        "docker exec -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:?required} "
        '-w "$PWD" ${GEAK_CONTAINER_NAME:-geak-phase1-vllm} '
        "python3 scripts/task_runner.py "
    )
    config = {
        "source_file_path": ["kernel.py"],
        "target_kernel_functions": [request.family],
        "compile_command": [command + "compile"],
        "correctness_command": [command + "correctness"],
        "performance_command": [command + "performance"],
        "task_type": request.family,
    }
    metadata = contract.metadata
    metadata["top10_family"] = request.family
    metadata["provenance"] = {
        "generator": "multi_tune_agent.top10_canonical_templates",
        "generation_method": "source_locked_independent_canonical_factory",
        "source_repo": "https://github.com/ROCm/aiter.git",
        "source_sha": LOCKED_AITER_SHA,
        "source_artifacts": list(_source_artifacts(request.seed_provenance)),
        "case_seed": dict(request.seed_provenance),
        "request_id": request.request_id,
        "contract_hash": contract.contract_hash,
        "runtime_dependency": "none",
    }
    metadata["materialization_status"] = "untrusted_pending_gpu_gate"
    return {
        "kernel.py": kernel,
        "config.yaml": yaml.safe_dump(config, sort_keys=False),
        "scripts/task_runner.py": runner,
        "metadata.json": json.dumps(metadata, sort_keys=True, indent=2) + "\n",
    }


def _atomic_install(directory: Path, files: Mapping[str, str]) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".tmp-{directory.name}-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(mode=0o700)
        for relative in CANONICAL_FILES:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(files[relative], encoding="utf-8")
        if directory.exists():
            if not directory.is_dir() or directory.is_symlink():
                raise Top10CanonicalError(f"unsafe candidate destination: {directory}")
            current = {
                name: hashlib.sha256((directory / name).read_bytes()).digest()
                for name in CANONICAL_FILES
            }
            proposed = {
                name: hashlib.sha256((temporary / name).read_bytes()).digest()
                for name in CANONICAL_FILES
            }
            if current != proposed:
                raise Top10CanonicalError(
                    f"deterministic candidate conflicts with existing {directory}"
                )
            return
        os.replace(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def materialize_request(
    request: Top10Request, candidate_root: Path | str
) -> TemplateDraft:
    contract = request.contract
    destination = Path(candidate_root).expanduser().resolve() / contract.contract_hash
    _atomic_install(destination, render_template(request))
    report = validate_generated_template(destination, contract.expected_contract)
    if not report.valid:
        raise Top10CanonicalError(
            "canonical candidate failed static validation: "
            + "; ".join(str(error) for error in report.errors)
        )
    return TemplateDraft(
        destination,
        contract,
        report,
        "source_locked_independent_canonical_factory",
        tuple(_source_artifacts(request.seed_provenance)),
    )


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False
        ) as temporary:
            temporary_name = temporary.name
            yaml.safe_dump(dict(payload), temporary, sort_keys=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def merge_catalog(
    output_path: Path, records: Sequence[Mapping[str, Any]],
    base_catalog: Path | None = None
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
            payload = {"schema_version": "top10_canonical_gpu_promotions_v1", "tasks": []}
        tasks = {
            str(task["id"]): dict(task)
            for task in payload.get("tasks", [])
            if isinstance(task, Mapping) and task.get("id")
        }
        for raw in records:
            record = dict(raw)
            task_id = str(record.get("id") or "")
            bundle = Path(str(record.get("kernel_path") or ""))
            try:
                metadata = json.loads(
                    (bundle / "metadata.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise Top10CanonicalError(f"invalid promoted bundle for {task_id}") from exc
            trust = metadata.get("trust")
            if (
                not isinstance(trust, Mapping)
                or trust.get("trusted") is not True
                or trust.get("contract_hash") != record.get("contract_hash")
                or metadata.get("contract_hash") != record.get("contract_hash")
            ):
                raise Top10CanonicalError(f"refusing to catalog untrusted {task_id}")
            if task_id in tasks and tasks[task_id] != record:
                raise Top10CanonicalError(f"catalog already has conflicting task {task_id}")
            tasks[task_id] = record
        payload["tasks"] = [tasks[key] for key in sorted(tasks)]
        _atomic_yaml(output_path, payload)


def validate_gang_arguments(
    gpu_ids: str | Sequence[int], world_size: int, rank: int | None = None
) -> tuple[int, ...]:
    if world_size not in (2, 4):
        raise Top10CanonicalError("all-reduce world size must be 2 or 4")
    if isinstance(gpu_ids, str):
        try:
            values = tuple(int(value.strip()) for value in gpu_ids.split(","))
        except ValueError as exc:
            raise Top10CanonicalError("GPU IDs must be comma-separated integers") from exc
    else:
        values = tuple(gpu_ids)
    if len(values) != world_size or len(set(values)) != world_size:
        raise Top10CanonicalError(
            "all-reduce requires one distinct GPU ID per process"
        )
    if any(type(value) is not int or value < 0 for value in values):
        raise Top10CanonicalError("GPU IDs must be non-negative integers")
    if rank is not None and (type(rank) is not int or not 0 <= rank < world_size):
        raise Top10CanonicalError("rank must be in [0, world size)")
    return values


@contextlib.contextmanager
def _exclusive_gpu_gang(gpu_ids: Sequence[int]):
    """Hold the same per-GPU locks used by GEAK for the complete gang gate."""
    lock_root = Path("/tmp/team_gpu_locks")
    lock_root.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for gpu_id in sorted(gpu_ids):
            handle = (lock_root / f"gpu_{gpu_id}.lock").open("a+")
            fcntl.flock(handle, fcntl.LOCK_EX)
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def run_all_reduce_gpu_gate(
    draft: TemplateDraft,
    *,
    run_root: Path,
    gpu_ids: Sequence[int],
    world_size: int,
    command_timeout: int,
) -> TemplateGateResult:
    """Compile and verify RCCL all-reduce in one fresh, gang-locked workspace."""
    selected = validate_gang_arguments(gpu_ids, world_size)
    report = validate_generated_template(
        draft.path, draft.contract.expected_contract
    )
    if not report.valid:
        return TemplateGateResult(False, False, False, False, {}, errors=tuple(report.errors))
    workspace = (
        run_root.expanduser().resolve()
        / "top10-all-reduce-gang"
        / f"{draft.contract_hash[:16]}-{uuid.uuid4().hex}"
    )
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(draft.path, workspace)
    container = os.environ.get("GEAK_CONTAINER_NAME", "geak-phase1-vllm")
    visible = ",".join(str(value) for value in selected)
    summaries: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    states = {"compile": False, "correctness": False, "performance": False}
    with _exclusive_gpu_gang(selected):
        for mode in states:
            command = [
                "docker",
                "exec",
                "-e",
                f"HIP_VISIBLE_DEVICES={visible}",
                "-e",
                f"TORCH_EXTENSIONS_DIR={workspace}/.torch_ext",
                "-e",
                "PYTORCH_ROCM_ARCH=gfx942",
                "-w",
                str(workspace),
                container,
                "python3",
                "scripts/task_runner.py",
                mode,
            ]
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=command_timeout,
                )
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                proc = None
                timed_out = True
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
            else:
                stdout, stderr = proc.stdout, proc.stderr
            ok = proc is not None and proc.returncode == 0
            summaries[mode] = {
                "command": " ".join(command),
                "mode": mode,
                "ok": ok,
                "returncode": proc.returncode if proc is not None else None,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": timed_out,
                "gpu_ids": list(selected),
                "world_size": world_size,
            }
            states[mode] = ok
            if not ok:
                errors.append(
                    f"{mode} failed"
                    + (" by timeout" if timed_out else f": {stderr.strip()}")
                )
                break
    per_case_ms: dict[str, float] = {}
    performance_report = workspace / "build" / "performance_report.json"
    if states["performance"] and performance_report.is_file():
        payload = json.loads(performance_report.read_text(encoding="utf-8"))
        for item in payload.get("test_cases", []):
            per_case_ms[str(item["test_case_id"])] = float(
                item["execution_time_ms"]
            )
    performance_valid = states["performance"] and bool(per_case_ms)
    if states["performance"] and not performance_valid:
        errors.append("performance report is missing positive case timing")
    return TemplateGateResult(
        static_valid=True,
        compiled=states["compile"],
        correct=states["correctness"],
        performance_valid=performance_valid,
        per_case_ms=per_case_ms,
        command_summaries=summaries,
        errors=tuple(errors),
        validation_workspace=workspace,
    )


def _selected(
    requests: Sequence[Top10Request], shard_index: int, shard_count: int
) -> list[Top10Request]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise Top10CanonicalError("shard index must be in [0, shard count)")
    return [
        request for index, request in enumerate(requests)
        if index % shard_count == shard_index
    ]


def _trusted_ids(output: Path) -> set[str]:
    if not output.is_file():
        return set()
    trusted = set()
    for task in _mapping_file(output, "output catalog").get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        try:
            metadata = json.loads(
                (Path(str(task["kernel_path"])) / "metadata.json").read_text()
            )
        except (KeyError, OSError, json.JSONDecodeError):
            continue
        trust = metadata.get("trust", {})
        if (
            isinstance(trust, Mapping)
            and trust.get("trusted") is True
            and trust.get("contract_hash") == task.get("contract_hash")
            and metadata.get("contract_hash") == task.get("contract_hash")
        ):
            trusted.add(str(task.get("id")))
    return trusted


def _task_record(
    request: Top10Request, draft: TemplateDraft, promoted: Path
) -> dict[str, Any]:
    metadata = json.loads(
        (promoted / "metadata.json").read_text(encoding="utf-8")
    )
    source_provenance = metadata.get("provenance")
    source_provenance = (
        dict(source_provenance) if isinstance(source_provenance, Mapping) else {}
    )
    source_provenance.update(
        {
            "case_seed": dict(request.seed_provenance),
            "contract_hash": draft.contract_hash,
            "top10_family": request.family,
        }
    )
    return {
        "id": request.request_id,
        "type": "aiter_generated",
        "kernel_path": str(promoted),
        "direction": request.request_text,
        "operator": request.family,
        "top10_family": request.family,
        "backend": request.language,
        "architecture": SUPPORTED_ARCH,
        "contract_hash": draft.contract_hash,
        "recognized_contract": dict(request.recognized_contract),
        "provenance": source_provenance,
    }


def _diagnostic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_yaml(path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "materialize", "gate"))
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--aiter-root", required=True, type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--verified-root", type=Path)
    parser.add_argument("--output-catalog", type=Path)
    parser.add_argument("--base-catalog", type=Path)
    parser.add_argument("--geak-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpu-ids", default="1")
    parser.add_argument("--all-reduce-world-size", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--command-timeout", type=int, default=1800)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requests = load_top10_requests(args.requests)
    verify_locked_source(args.aiter_root, requests)
    selected = _selected(requests, args.shard_index, args.shard_count)
    if args.case_id:
        requested_ids = set(args.case_id)
        known_ids = {request.request_id for request in requests}
        unknown = sorted(requested_ids - known_ids)
        if unknown:
            raise Top10CanonicalError(
                "unknown case IDs: " + ", ".join(unknown)
            )
        selected = [
            request for request in selected if request.request_id in requested_ids
        ]
    summary: dict[str, Any] = {
        **family_counts(requests),
        "source_verified": True,
        "selected": len(selected),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "fail_closed_families": sorted(
            {request.family for request in requests if request.family in FAIL_CLOSED_REASONS}
        ),
    }
    if args.mode == "plan":
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.candidate_root is None:
        raise Top10CanonicalError("materialize and gate require --candidate-root")
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
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise Top10CanonicalError("gate requires " + ", ".join(missing))
    completed = _trusted_ids(args.output_catalog)
    selected = [request for request in selected if request.request_id not in completed]
    records, failures = [], []
    diagnostic_root = args.run_root.expanduser().resolve() / "top10-gate-results"
    for request in selected:
        draft = materialize_request(request, args.candidate_root)
        if request.family in FAIL_CLOSED_REASONS:
            if request.family == "all_reduce":
                frozen = request.recognized_contract.get("contract")
                assert isinstance(frozen, Mapping)
                contract_world_size = int(frozen["world_size"])
                if (
                    args.all_reduce_world_size is not None
                    and args.all_reduce_world_size != contract_world_size
                ):
                    raise Top10CanonicalError(
                        "all-reduce CLI world size does not match frozen contract"
                    )
                if args.all_reduce_world_size is not None:
                    validate_gang_arguments(args.gpu_ids, contract_world_size)
            failures.append(request.request_id)
            _diagnostic(
                diagnostic_root / f"{request.request_id}.yaml",
                {
                    "case_id": request.request_id,
                    "trusted": False,
                    "fail_closed": True,
                    "family": request.family,
                    "reason": FAIL_CLOSED_REASONS[request.family],
                },
            )
            continue
        if request.family == "all_reduce":
            frozen = request.recognized_contract.get("contract")
            assert isinstance(frozen, Mapping)
            contract_world_size = int(frozen["world_size"])
            if (
                args.all_reduce_world_size is not None
                and args.all_reduce_world_size != contract_world_size
            ):
                raise Top10CanonicalError(
                    "all-reduce CLI world size does not match frozen contract"
                )
            gpu_ids = validate_gang_arguments(
                args.gpu_ids, contract_world_size
            )
            result = run_all_reduce_gpu_gate(
                draft,
                run_root=args.run_root,
                gpu_ids=gpu_ids,
                world_size=contract_world_size,
                command_timeout=args.command_timeout,
            )
        else:
            result = run_template_gpu_gate(
                draft,
                geak_root=args.geak_root,
                run_root=args.run_root,
                gpu_ids=args.gpu_ids,
                command_timeout=args.command_timeout,
            )
        _diagnostic(
            diagnostic_root / f"{request.request_id}.yaml",
            {
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
            },
        )
        if not result.trusted:
            failures.append(request.request_id)
            continue
        promoted = promote_validated_template(draft, result, args.verified_root)
        record = _task_record(request, draft, promoted)
        merge_catalog(args.output_catalog, [record], args.base_catalog)
        records.append(record)
    summary.update(
        resumed=len(completed),
        promoted=len(records),
        failed_gate=failures,
        fail_closed_families=sorted(
            {request.family for request in selected if request.family in FAIL_CLOSED_REASONS}
        ),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LOCKED_AITER_SHA",
    "TOP10_FAMILIES",
    "Top10CanonicalError",
    "Top10Request",
    "family_counts",
    "load_top10_requests",
    "main",
    "materialize_request",
    "merge_catalog",
    "render_template",
    "validate_gang_arguments",
    "verify_locked_source",
]
