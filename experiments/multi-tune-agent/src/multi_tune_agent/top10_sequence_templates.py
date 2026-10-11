"""Independent canonical kernels for the Top10 sequence-state families.

The rendered kernel module does not use AITER or PyTorch computational
operators.  Callers provide every output tensor, which also makes cache
mutation and sampling results observable without hidden allocations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SUPPORTED_FAMILIES = frozenset({"rope_kv_cache", "sampling"})
SUPPORTED_LANGUAGES = frozenset({"hip", "triton"})
SUPPORTED_ARCH = "gfx942"


class SequenceTemplateError(ValueError):
    """The requested template is outside the canonical contract."""


def _contract(family: str, contract: Mapping[str, Any]) -> tuple[str, dict[str, int]]:
    if family not in SUPPORTED_FAMILIES:
        raise SequenceTemplateError(f"unsupported sequence family: {family!r}")
    if contract.get("operator") != family:
        raise SequenceTemplateError("family and contract operator must match")
    mode = str(contract.get("mode") or "")
    modes = {
        "rope_kv_cache": {"sbhd", "fused_write"},
        "sampling": {"top_p", "top_k_top_p"},
    }
    if mode not in modes[family]:
        raise SequenceTemplateError(f"unsupported {family} mode: {mode!r}")
    expected_output = "int32" if family == "sampling" else "bf16"
    if (
        contract.get("input_dtype") != "bf16"
        or contract.get("output_dtype") != expected_output
    ):
        raise SequenceTemplateError(
            f"canonical {family} contract requires bf16 input and "
            f"{expected_output} output"
        )
    shape = contract.get("shape")
    if not isinstance(shape, Mapping):
        raise SequenceTemplateError("contract shape must be a mapping")
    dimensions: dict[str, int] = {}
    for name, value in shape.items():
        if type(value) is not int or value <= 0:
            raise SequenceTemplateError("shape dimensions must be positive integers")
        dimensions[str(name)] = value
    if family == "rope_kv_cache":
        required = {"S", "B", "H", "D"} if mode == "sbhd" else {
            "TOKENS", "HEADS", "D", "BLOCK"
        }
        if not required <= dimensions.keys() or dimensions["D"] % 2:
            raise SequenceTemplateError("RoPE shape is incomplete or D is odd")
    elif not {"B", "VOCAB"} <= dimensions.keys():
        raise SequenceTemplateError("sampling shape requires B and VOCAB")
    return mode, dimensions


_TRITON_ROPE = r'''import triton
import triton.language as tl


@triton.jit
def _rope_kv_cache_kernel(
    q, k, value, cos, sin, slots, key_cache, value_cache, q_out, k_out,
    T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    CACHE_BLOCK: tl.constexpr, BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    valid = d < D
    half = D // 2
    partner_d = tl.where(d < half, d + half, d - half)
    sign = tl.where(d < half, -1.0, 1.0)
    base = (token * H + head) * D
    qv = tl.load(q + base + d, mask=valid, other=0.0).to(tl.float32)
    kv = tl.load(k + base + d, mask=valid, other=0.0).to(tl.float32)
    qp = tl.load(q + base + partner_d, mask=valid, other=0.0).to(tl.float32)
    kp = tl.load(k + base + partner_d, mask=valid, other=0.0).to(tl.float32)
    c = tl.load(cos + token * D + d, mask=valid, other=0.0).to(tl.float32)
    s = tl.load(sin + token * D + d, mask=valid, other=0.0).to(tl.float32)
    qr = qv * c + sign * qp * s
    kr = kv * c + sign * kp * s
    tl.store(q_out + base + d, qr, mask=valid)
    tl.store(k_out + base + d, kr, mask=valid)
    slot = tl.load(slots + token)
    cache = (((slot // CACHE_BLOCK) * CACHE_BLOCK + slot % CACHE_BLOCK) * H + head) * D
    tl.store(key_cache + cache + d, kr, mask=valid)
    vv = tl.load(value + base + d, mask=valid, other=0.0)
    tl.store(value_cache + cache + d, vv, mask=valid)


def rope_kv_cache(
    q, k, value, cos, sin, slots, key_cache, value_cache, q_out, k_out
):
    if q.shape != k.shape or q.shape != value.shape or q.ndim != 3:
        raise ValueError("q, k, and value must have identical [T,H,D] shapes")
    tokens, heads, width = q.shape
    if width % 2:
        raise ValueError("rotary width must be even")
    _rope_kv_cache_kernel[(tokens, heads)](
        q, k, value, cos, sin, slots, key_cache, value_cache, q_out, k_out,
        T=tokens, H=heads, D=width, CACHE_BLOCK=key_cache.shape[1],
        BLOCK_D=triton.next_power_of_2(width), num_warps=1,
    )
    return q_out, k_out
'''


_TRITON_SAMPLING = r'''import triton
import triton.language as tl


@triton.jit
def _sampling_kernel(
    probs, uniforms, out, top_k, top_p,
    V: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.arange(0, BLOCK)
    valid = token < V
    probability = tl.load(
        probs + row * V + token, mask=valid, other=-1.0
    ).to(tl.float32)
    # A stable descending rank (token id breaks equal-probability ties).
    prefix = tl.zeros((BLOCK,), tl.float32)
    rank = tl.zeros((BLOCK,), tl.int32)
    for other in range(0, V):
        other_p = tl.load(probs + row * V + other).to(tl.float32)
        before = (other_p > probability) | (
            (other_p == probability) & (other < token)
        )
        rank += before
        prefix += tl.where(before, other_p, 0.0)
    limit = tl.where(top_k <= 0, V, tl.minimum(top_k, V))
    keep = valid & (rank < limit) & (prefix < top_p)
    mass = tl.sum(tl.where(keep, probability, 0.0), axis=0)
    target = tl.load(uniforms + row).to(tl.float32) * mass
    chosen = keep & (target >= prefix) & (target < prefix + probability)
    result = tl.min(tl.where(chosen, token, V), axis=0)
    # u is required to be in [0,1), so this only guards roundoff at the end.
    fallback = tl.min(tl.where(keep & (rank == 0), token, V), axis=0)
    tl.store(out + row, tl.where(result < V, result, fallback))


def sampling(probs, uniforms, top_k, top_p, out):
    if probs.ndim != 2 or uniforms.ndim != 1 or out.ndim != 1:
        raise ValueError("expected probs[B,V], uniforms[B], and out[B]")
    if probs.shape[0] != uniforms.shape[0] or probs.shape[0] != out.shape[0]:
        raise ValueError("sampling batch dimensions differ")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be in (0,1]")
    vocab = probs.shape[1]
    _sampling_kernel[(probs.shape[0],)](
        probs, uniforms, out, int(top_k), float(top_p),
        V=vocab, BLOCK=triton.next_power_of_2(vocab), num_warps=4,
    )
    return out
'''


_HIP_PREAMBLE = r'''from pathlib import Path
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
'''


_HIP_POSTAMBLE = r'''
"""
_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        build = Path(__file__).resolve().parent / "build" / "hip_extension"
        build.mkdir(parents=True, exist_ok=True)
        name = "sequence_" + hashlib.sha256(_SOURCE.encode()).hexdigest()[:16]
        _MODULE = load_inline(
            name=name, cpp_sources="", cuda_sources=_SOURCE, functions=None,
            extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
            with_cuda=True, build_directory=str(build), verbose=False,
        )
    return _MODULE
'''


_HIP_ROPE = _HIP_PREAMBLE + r'''
__global__ void rope_cache_kernel(
    const __hip_bfloat16* q, const __hip_bfloat16* k,
    const __hip_bfloat16* value, const float* cos, const float* sin,
    const int32_t* slots, __hip_bfloat16* key_cache,
    __hip_bfloat16* value_cache, __hip_bfloat16* q_out,
    __hip_bfloat16* k_out, int64_t elements, int H, int D, int block_size) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const int d = index % D;
  const int64_t row = index / D;
  const int token = row / H;
  const int partner = d < D / 2 ? d + D / 2 : d - D / 2;
  const float sign = d < D / 2 ? -1.0f : 1.0f;
  const int64_t partner_index = row * D + partner;
  const float c = cos[token * D + d], s = sin[token * D + d];
  const float qr = __bfloat162float(q[index]) * c +
                   sign * __bfloat162float(q[partner_index]) * s;
  const float kr = __bfloat162float(k[index]) * c +
                   sign * __bfloat162float(k[partner_index]) * s;
  q_out[index] = __float2bfloat16(qr);
  k_out[index] = __float2bfloat16(kr);
  const int slot = slots[token];
  const int64_t cache_index =
      (((int64_t)(slot / block_size) * block_size + slot % block_size) * H
       + row % H) * D + d;
  key_cache[cache_index] = __float2bfloat16(kr);
  value_cache[cache_index] = value[index];
}

void rope_kv_cache(
    torch::Tensor q, torch::Tensor k, torch::Tensor value,
    torch::Tensor cos, torch::Tensor sin, torch::Tensor slots,
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor q_out, torch::Tensor k_out) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3,
              "q must be ROCm bf16 [T,H,D]");
  TORCH_CHECK(k.sizes() == q.sizes() && value.sizes() == q.sizes(),
              "k/value shape mismatch");
  TORCH_CHECK(q.size(2) % 2 == 0 && cos.size(0) == q.size(0) &&
              cos.size(1) == q.size(2) && sin.sizes() == cos.sizes(),
              "invalid rotary tables");
  TORCH_CHECK(slots.scalar_type() == at::kInt && slots.numel() == q.size(0),
              "slots must be int32[T]");
  const int threads = 256;
  hipStream_t stream = at::hip::getCurrentHIPStream();
  rope_cache_kernel<<<(q.numel() + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(k.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(value.data_ptr()),
      cos.data_ptr<float>(), sin.data_ptr<float>(), slots.data_ptr<int32_t>(),
      reinterpret_cast<__hip_bfloat16*>(key_cache.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(value_cache.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(q_out.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(k_out.data_ptr()),
      q.numel(), q.size(1), q.size(2), key_cache.size(1));
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_kv_cache", &rope_kv_cache);
}
''' + _HIP_POSTAMBLE + r'''


def rope_kv_cache(
    q, k, value, cos, sin, slots, key_cache, value_cache, q_out, k_out
):
    _module().rope_kv_cache(
        q, k, value, cos, sin, slots, key_cache, value_cache, q_out, k_out
    )
    return q_out, k_out
'''


_HIP_SAMPLING = _HIP_PREAMBLE + r'''
__global__ void sampling_kernel(
    const __hip_bfloat16* probs, const float* uniforms, int32_t* out,
    int B, int V, int top_k, float top_p) {
  const int row = blockIdx.x;
  if (row >= B || threadIdx.x != 0) return;
  const int limit = top_k <= 0 ? V : (top_k > V ? V : top_k);
  float mass = 0.0f;
  float prefix = 0.0f;
  float previous_p = 3.402823466e+38f;
  int previous_token = -1;
  for (int wanted_rank = 0; wanted_rank < limit; ++wanted_rank) {
    float ranked_p = -1.0f;
    int ranked_token = -1;
    for (int token = 0; token < V; ++token) {
      const float p = __bfloat162float(probs[row * V + token]);
      const bool after_previous =
          p < previous_p || (p == previous_p && token > previous_token);
      if (after_previous &&
          (p > ranked_p || (p == ranked_p && token < ranked_token))) {
        ranked_p = p;
        ranked_token = token;
      }
    }
    if (prefix >= top_p || ranked_token < 0) break;
    mass += ranked_p;
    prefix += ranked_p;
    previous_p = ranked_p;
    previous_token = ranked_token;
  }
  const float target = uniforms[row] * mass;
  int answer = 0;
  prefix = 0.0f;
  previous_p = 3.402823466e+38f;
  previous_token = -1;
  for (int wanted_rank = 0; wanted_rank < limit; ++wanted_rank) {
    int ranked_token = -1;
    float ranked_p = -1.0f;
    for (int token = 0; token < V; ++token) {
      const float p = __bfloat162float(probs[row * V + token]);
      const bool after_previous =
          p < previous_p || (p == previous_p && token > previous_token);
      if (after_previous &&
          (p > ranked_p || (p == ranked_p && token < ranked_token))) {
        ranked_token = token;
        ranked_p = p;
      }
    }
    if (prefix >= top_p || ranked_token < 0) break;
    if (target >= prefix && target < prefix + ranked_p) answer = ranked_token;
    prefix += ranked_p;
    previous_p = ranked_p;
    previous_token = ranked_token;
  }
  out[row] = answer;
}

void sampling(
    torch::Tensor probs, torch::Tensor uniforms, int64_t top_k,
    double top_p, torch::Tensor out) {
  TORCH_CHECK(probs.is_cuda() && probs.scalar_type() == at::kBFloat16 &&
              probs.dim() == 2 && probs.is_contiguous(),
              "probs must be contiguous ROCm bf16 [B,V]");
  TORCH_CHECK(uniforms.scalar_type() == at::kFloat &&
              uniforms.numel() == probs.size(0), "uniforms must be fp32[B]");
  TORCH_CHECK(out.scalar_type() == at::kInt && out.numel() == probs.size(0),
              "out must be int32[B]");
  TORCH_CHECK(top_p > 0.0 && top_p <= 1.0, "top_p must be in (0,1]");
  hipStream_t stream = at::hip::getCurrentHIPStream();
  sampling_kernel<<<probs.size(0), 1, 0, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(probs.data_ptr()),
      uniforms.data_ptr<float>(), out.data_ptr<int32_t>(), probs.size(0),
      probs.size(1), top_k, static_cast<float>(top_p));
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sampling", &sampling);
}
''' + _HIP_POSTAMBLE + r'''


def sampling(probs, uniforms, top_k, top_p, out):
    _module().sampling(probs, uniforms, int(top_k), float(top_p), out)
    return out
'''


def render_sequence_kernel(
    family: str, language: str, contract: Mapping[str, Any]
) -> str:
    """Render a standalone custom-kernel module for a frozen sequence contract."""
    _contract(family, contract)
    normalized = language.lower()
    if normalized not in SUPPORTED_LANGUAGES:
        raise SequenceTemplateError("language must be hip or triton")
    if family == "rope_kv_cache":
        return _HIP_ROPE if normalized == "hip" else _TRITON_ROPE
    return _HIP_SAMPLING if normalized == "hip" else _TRITON_SAMPLING


def _small_dimensions(
    family: str, mode: str, dimensions: Mapping[str, int]
) -> tuple[int, ...]:
    if family == "rope_kv_cache":
        tokens = (
            min(dimensions["S"] * dimensions["B"], 8)
            if mode == "sbhd"
            else min(dimensions["TOKENS"], 8)
        )
        heads = min(dimensions["H"] if mode == "sbhd" else dimensions["HEADS"], 4)
        width = min(dimensions["D"], 32)
        width -= width % 2
        block = min(dimensions.get("BLOCK", 4), 4)
        return tokens, heads, width, block
    return min(dimensions["B"], 8), min(dimensions["VOCAB"], 128)


def render_sequence_runner(family: str, contract: Mapping[str, Any]) -> str:
    """Render the gfx942 compile/correctness/performance gate runner."""
    mode, dimensions = _contract(family, contract)
    shape = _small_dimensions(family, mode, dimensions)
    if family == "rope_kv_cache":
        setup = f'''T, H, D, CACHE_BLOCK = {shape!r}
    torch.manual_seed(9410)
    q = torch.randn((T, H, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    value = torch.randn_like(q)
    angle = torch.arange(T * D, device="cuda", dtype=torch.float32).reshape(T, D) / 37.0
    cos, sin = angle.cos(), angle.sin()
    slots = (torch.arange(T, device="cuda", dtype=torch.int32) * 3) % (T * CACHE_BLOCK)
    blocks = (T * CACHE_BLOCK + CACHE_BLOCK - 1) // CACHE_BLOCK
    key_cache = torch.full((blocks, CACHE_BLOCK, H, D), -17.0, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.full_like(key_cache, -19.0)
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    return (q, k, value, cos, sin, slots, key_cache, value_cache, q_out, k_out)'''
        oracle = '''q, k, value, cos, sin, slots, key_cache, value_cache, q_out, k_out = args
    half = q.shape[-1] // 2
    rotate = lambda x: torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    expected_q = (q.float() * cos[:, None, :] + rotate(q.float()) * sin[:, None, :]).to(torch.bfloat16)
    expected_k = (k.float() * cos[:, None, :] + rotate(k.float()) * sin[:, None, :]).to(torch.bfloat16)
    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()
    block = key_cache.shape[1]
    for token, slot in enumerate(slots.cpu().tolist()):
        expected_key_cache[slot // block, slot % block] = expected_k[token]
        expected_value_cache[slot // block, slot % block] = value[token]
    return expected_q, expected_k, expected_key_cache, expected_value_cache'''
        invoke = "actual_q, actual_k = kernel.rope_kv_cache(*args)"
        checks = '''actual_q, actual_k = result
    expected_q, expected_k, expected_key_cache, expected_value_cache = independent_torch_oracle(args)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=2e-2)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=2e-2)
    torch.testing.assert_close(args[6], expected_key_cache, rtol=0, atol=2e-2)
    torch.testing.assert_close(args[7], expected_value_cache, rtol=0, atol=0)'''
    else:
        batch, vocab = shape
        top_k = 0 if mode == "top_p" else min(17, vocab)
        setup = f'''B, V = {(batch, vocab)!r}
    torch.manual_seed(9410)
    raw = torch.rand((B, V), device="cuda", dtype=torch.float32)
    raw[0, :4] = torch.tensor([0.4, 0.3, 0.2, 0.1], device="cuda")
    probs = (raw / raw.sum(-1, keepdim=True)).to(torch.bfloat16)
    probs = (probs.float() / probs.float().sum(-1, keepdim=True)).to(torch.bfloat16)
    uniforms = torch.rand((B,), device="cuda", dtype=torch.float32)
    uniforms[:4] = torch.tensor([0.0, 0.4, 0.7, 0.999999], device="cuda")
    out = torch.empty((B,), device="cuda", dtype=torch.int32)
    return probs, uniforms, {top_k}, 0.75, out'''
        oracle = '''probs, uniforms, top_k, top_p, _ = args
    order = torch.argsort(probs.float(), dim=-1, descending=True, stable=True)
    sorted_probs = torch.gather(probs.float(), 1, order)
    ranks = torch.arange(probs.shape[1], device="cuda")[None, :]
    limit = probs.shape[1] if int(top_k) <= 0 else min(int(top_k), probs.shape[1])
    keep = ranks < limit
    prefix = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    keep = keep & (prefix < float(top_p))
    kept = sorted_probs * keep
    target = uniforms * kept.sum(-1)
    selected_rank = ((target[:, None] >= prefix) & (target[:, None] < prefix + sorted_probs) & keep).to(torch.int32).argmax(-1)
    return torch.gather(order, 1, selected_rank[:, None]).squeeze(1).to(torch.int32)'''
        invoke = "actual = kernel.sampling(*args)"
        checks = '''actual = result
    expected = independent_torch_oracle(args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if torch.equal(actual, torch.argmax(args[0], dim=-1).to(torch.int32)):
        raise AssertionError("sampling degenerated to argmax for every row")'''
    return f'''import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILY = {family!r}


def require_gfx942():
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx942":
        raise RuntimeError("expected gfx942, found " + arch)


def make_case():
    {setup}


def independent_torch_oracle(args):
    {oracle}


def invoke(kernel, args):
    {invoke}
    return locals().get("actual", (actual_q, actual_k) if FAMILY == "rope_kv_cache" else None)


def verify(kernel, args):
    result = invoke(kernel, args)
    {checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    require_gfx942()
    kernel = importlib.import_module("kernel")
    args = make_case()
    if mode == "compile":
        invoke(kernel, args)
        torch.cuda.synchronize()
        print("Compile: OK")
        return
    if mode == "correctness":
        verify(kernel, args)
        print("Correctness: OK")
        return
    for _ in range(2):
        invoke(kernel, args)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        invoke(kernel, args)
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end) / 10.0
    if not elapsed > 0:
        raise AssertionError("invalid latency")
    case_id = FAMILY + "_" + mode + "_" + "x".join(map(str, {shape!r}))
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    report = {{"test_cases": [{{"test_case_id": case_id, "execution_time_ms": elapsed}}]}}
    (build / "performance_report.json").write_text(json.dumps(report), encoding="utf-8")
    print(f"Perf: {{elapsed:.6f}} ms ({{case_id}})")


if __name__ == "__main__":
    main()
'''


__all__ = [
    "SUPPORTED_ARCH",
    "SUPPORTED_FAMILIES",
    "SUPPORTED_LANGUAGES",
    "SequenceTemplateError",
    "render_sequence_kernel",
    "render_sequence_runner",
]
