"""Independent canonical renderers for the Top10 attention families.

The rendered files intentionally depend only on PyTorch's extension surface or
Triton.  PyTorch operations in the runner are an independent fp32 oracle; the
candidate implementations perform all attention arithmetic in custom kernels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

SUPPORTED_FAMILIES = frozenset({"mha", "mla", "paged_attention"})
SUPPORTED_LANGUAGES = frozenset({"hip", "triton"})
SUPPORTED_ARCH = "gfx942"


def _family(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return {
        "multi_head_attention": "mha",
        "multi_latent_attention": "mla",
        "paged_attn": "paged_attention",
    }.get(normalized, normalized)


def _validate(family: str, language: str | None, contract: Mapping[str, Any]) -> None:
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported attention family: {family!r}")
    if language is not None and language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported attention language: {language!r}")
    target = str(
        contract.get("target_gpu")
        or contract.get("architecture")
        or contract.get("arch")
        or SUPPORTED_ARCH
    ).lower()
    if target != SUPPORTED_ARCH:
        raise ValueError(f"attention templates require {SUPPORTED_ARCH}, got {target!r}")
    operator = _family(contract.get("operator") or family)
    if operator != family:
        raise ValueError(f"family/operator mismatch: {family!r}/{operator!r}")


def _shape(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    value = contract.get("shape") or contract.get("dimensions") or {}
    if not isinstance(value, Mapping):
        raise ValueError("attention contract shape must be a mapping")
    return value


def _positive(shape: Mapping[str, Any], names: tuple[str, ...], default: int) -> int:
    for name in names:
        if name in shape:
            value = shape[name]
            if type(value) is not int or value <= 0:
                raise ValueError(f"shape {name} must be a positive integer")
            return value
    return default


def _dims(family: str, contract: Mapping[str, Any]) -> dict[str, int]:
    """Choose deterministic, small dimensions while preserving contract ratios."""
    shape = _shape(contract)
    if family == "mha":
        b = min(_positive(shape, ("B", "batch"), 2), 2)
        sq = min(_positive(shape, ("SQ", "Q", "query_length"), 8), 16)
        sk = min(_positive(shape, ("SK", "S", "key_length"), sq), 24)
        hq = min(_positive(shape, ("HQ", "H", "query_heads"), 4), 8)
        hk_raw = _positive(shape, ("HK", "KV_HEADS", "kv_heads"), hq)
        hk = min(hk_raw, hq)
        if hq % hk:
            hk = math.gcd(hq, hk) or 1
        d = min(_positive(shape, ("D", "HEAD_DIM", "head_dim"), 32), 64)
        return {"B": b, "SQ": sq, "SK": sk, "HQ": hq, "HK": hk, "D": d}
    if family == "mla":
        b = min(_positive(shape, ("B", "batch"), 2), 2)
        s = min(_positive(shape, ("S", "SK", "sequence"), 12), 24)
        h = min(_positive(shape, ("H", "HQ", "heads"), 4), 8)
        kv = min(_positive(shape, ("KV", "KV_LORA_RANK", "latent_dim"), 32), 64)
        rope = min(_positive(shape, ("ROPE", "QK_ROPE_HEAD_DIM", "rope_dim"), 16), 32)
        return {"B": b, "S": s, "H": h, "KV": kv, "ROPE": rope}
    b = min(_positive(shape, ("B", "batch"), 2), 2)
    hq = min(_positive(shape, ("HQ", "H", "query_heads"), 4), 8)
    hk_raw = _positive(shape, ("HK", "KV_HEADS", "kv_heads"), 1)
    hk = min(hk_raw, hq)
    if hq % hk:
        hk = math.gcd(hq, hk) or 1
    block = min(_positive(shape, ("BLOCK", "BLOCK_SIZE", "block_size"), 8), 16)
    s = min(_positive(shape, ("S", "SK", "sequence"), block * 2), block * 3)
    d = min(_positive(shape, ("D", "HEAD_DIM", "head_dim"), 32), 64)
    return {"B": b, "HQ": hq, "HK": hk, "BLOCK": block, "S": s, "D": d}


_TRITON_MHA = r'''import torch
import triton
import triton.language as tl

@triton.jit
def _mha_kernel(q, k, v, out, SQ: tl.constexpr, SK: tl.constexpr,
                HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    b = row // (SQ * HQ)
    rem = row % (SQ * HQ)
    qi, hq = rem // HQ, rem % HQ
    hk = hq // (HQ // HK)
    s, d = tl.arange(0, BLOCK_S), tl.arange(0, BLOCK_D)
    qv = tl.load(q + ((b * SQ + qi) * HQ + hq) * D + d,
                 mask=d < D, other=0.0).to(tl.float32)
    key_limit = qi + SK - SQ
    valid = (s < SK) & (s <= key_limit)
    kp = k + (((b * SK + s[:, None]) * HK + hk) * D + d[None, :])
    kval = tl.load(kp, mask=valid[:, None] & (d[None, :] < D),
                   other=0.0).to(tl.float32)
    score = tl.sum(kval * qv[None, :], axis=1) * (D ** -0.5)
    score = tl.where(valid, score, -float("inf"))
    score = score - tl.max(score, axis=0)
    weights = tl.exp(score)
    weights = weights / tl.sum(weights, axis=0)
    vp = v + (((b * SK + s[:, None]) * HK + hk) * D + d[None, :])
    vv = tl.load(vp, mask=valid[:, None] & (d[None, :] < D),
                 other=0.0).to(tl.float32)
    result = tl.sum(weights[:, None] * vv, axis=0)
    tl.store(out + ((b * SQ + qi) * HQ + hq) * D + d, result, mask=d < D)

def mha(q, k, v):
    B, SQ, HQ, D = q.shape
    SK, HK = k.shape[1], k.shape[2]
    assert k.shape == (B, SK, HK, D) and v.shape == k.shape and HQ % HK == 0
    out = torch.empty_like(q)
    _mha_kernel[(B * SQ * HQ,)](
        q, k, v, out, SQ=SQ, SK=SK, HQ=HQ, HK=HK, D=D,
        BLOCK_S=triton.next_power_of_2(SK), BLOCK_D=triton.next_power_of_2(D))
    return out
'''


_TRITON_MLA = r'''import torch
import triton
import triton.language as tl

@triton.jit
def _mla_kernel(q_latent, q_rope, latent_kv, rope_k, out,
                S: tl.constexpr, H: tl.constexpr, KV: tl.constexpr,
                ROPE: tl.constexpr, BLOCK_S: tl.constexpr,
                BLOCK_KV: tl.constexpr, BLOCK_R: tl.constexpr):
    row = tl.program_id(0)
    b, h = row // H, row % H
    s = tl.arange(0, BLOCK_S)
    dk, dr = tl.arange(0, BLOCK_KV), tl.arange(0, BLOCK_R)
    qk = tl.load(q_latent + (b * H + h) * KV + dk,
                 mask=dk < KV, other=0.0).to(tl.float32)
    qr = tl.load(q_rope + (b * H + h) * ROPE + dr,
                 mask=dr < ROPE, other=0.0).to(tl.float32)
    kv = tl.load(latent_kv + (b * S + s[:, None]) * KV + dk[None, :],
                 mask=(s[:, None] < S) & (dk[None, :] < KV),
                 other=0.0).to(tl.float32)
    rk = tl.load(rope_k + (b * S + s[:, None]) * ROPE + dr[None, :],
                 mask=(s[:, None] < S) & (dr[None, :] < ROPE),
                 other=0.0).to(tl.float32)
    score = (tl.sum(kv * qk[None, :], axis=1) +
             tl.sum(rk * qr[None, :], axis=1)) * ((KV + ROPE) ** -0.5)
    score = tl.where(s < S, score, -float("inf"))
    score = score - tl.max(score, axis=0)
    weights = tl.exp(score)
    weights = weights / tl.sum(weights, axis=0)
    result = tl.sum(weights[:, None] * kv, axis=0)
    tl.store(out + (b * H + h) * KV + dk, result, mask=dk < KV)

def mla(q_latent, q_rope, latent_kv, rope_k):
    B, H, KV = q_latent.shape
    S, ROPE = latent_kv.shape[1], q_rope.shape[2]
    assert q_rope.shape == (B, H, ROPE)
    assert latent_kv.shape == (B, S, KV) and rope_k.shape == (B, S, ROPE)
    out = torch.empty_like(q_latent)
    _mla_kernel[(B * H,)](
        q_latent, q_rope, latent_kv, rope_k, out, S=S, H=H, KV=KV,
        ROPE=ROPE, BLOCK_S=triton.next_power_of_2(S),
        BLOCK_KV=triton.next_power_of_2(KV),
        BLOCK_R=triton.next_power_of_2(ROPE))
    return out
'''


_TRITON_PAGED = r'''import torch
import triton
import triton.language as tl

@triton.jit
def _paged_attention_kernel(q, k_cache, v_cache, block_table, seq_lens, out,
                            HQ: tl.constexpr, HK: tl.constexpr,
                            BLOCK: tl.constexpr, MAX_BLOCKS: tl.constexpr,
                            D: tl.constexpr, MAX_S: tl.constexpr,
                            BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    b, hq = row // HQ, row % HQ
    hk = hq // (HQ // HK)
    s, d = tl.arange(0, MAX_S), tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + b)
    logical_block, offset = s // BLOCK, s % BLOCK
    physical = tl.load(block_table + b * MAX_BLOCKS + logical_block,
                       mask=(s < length) & (logical_block < MAX_BLOCKS), other=0)
    qv = tl.load(q + (b * HQ + hq) * D + d,
                 mask=d < D, other=0.0).to(tl.float32)
    cache_index = (((physical[:, None] * BLOCK + offset[:, None]) * HK + hk) * D
                   + d[None, :])
    valid = s < length
    kval = tl.load(k_cache + cache_index,
                   mask=valid[:, None] & (d[None, :] < D),
                   other=0.0).to(tl.float32)
    score = tl.sum(kval * qv[None, :], axis=1) * (D ** -0.5)
    score = tl.where(valid, score, -float("inf"))
    score = score - tl.max(score, axis=0)
    weights = tl.exp(score)
    weights = weights / tl.sum(weights, axis=0)
    vv = tl.load(v_cache + cache_index,
                 mask=valid[:, None] & (d[None, :] < D),
                 other=0.0).to(tl.float32)
    result = tl.sum(weights[:, None] * vv, axis=0)
    tl.store(out + (b * HQ + hq) * D + d, result, mask=d < D)

def paged_attention(q, k_cache, v_cache, block_table, seq_lens):
    B, HQ, D = q.shape
    BLOCK, HK = k_cache.shape[1], k_cache.shape[2]
    assert v_cache.shape == k_cache.shape and HQ % HK == 0
    assert block_table.shape[0] == B and seq_lens.shape == (B,)
    out = torch.empty_like(q)
    max_s = block_table.shape[1] * BLOCK
    _paged_attention_kernel[(B * HQ,)](
        q, k_cache, v_cache, block_table, seq_lens, out, HQ=HQ, HK=HK,
        BLOCK=BLOCK, MAX_BLOCKS=block_table.shape[1], D=D,
        MAX_S=triton.next_power_of_2(max_s),
        BLOCK_D=triton.next_power_of_2(D))
    return out
'''


_HIP_SOURCE = r'''import hashlib
import torch
from torch.utils.cpp_extension import load_inline

_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>

__device__ inline float bf(const __hip_bfloat16 x) { return __bfloat162float(x); }

__global__ void mha_kernel(const __hip_bfloat16* q, const __hip_bfloat16* k,
 const __hip_bfloat16* v, __hip_bfloat16* out, int B, int SQ, int SK,
 int HQ, int HK, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * SQ * HQ * D) return;
  int d = i % D, x = i / D, hq = x % HQ, qi = (x / HQ) % SQ, b = x / (HQ * SQ);
  int hk = hq / (HQ / HK), limit = qi + SK - SQ;
  float scale = rsqrtf(float(D)), maximum = -INFINITY;
  for (int s = 0; s < SK && s <= limit; ++s) {
    float score = 0.0f;
    for (int j = 0; j < D; ++j)
      score += bf(q[((b*SQ+qi)*HQ+hq)*D+j]) * bf(k[((b*SK+s)*HK+hk)*D+j]);
    maximum = fmaxf(maximum, score * scale);
  }
  float denominator = 0.0f, numerator = 0.0f;
  for (int s = 0; s < SK && s <= limit; ++s) {
    float score = 0.0f;
    for (int j = 0; j < D; ++j)
      score += bf(q[((b*SQ+qi)*HQ+hq)*D+j]) * bf(k[((b*SK+s)*HK+hk)*D+j]);
    float weight = expf(score * scale - maximum);
    denominator += weight; numerator += weight * bf(v[((b*SK+s)*HK+hk)*D+d]);
  }
  out[i] = __float2bfloat16(numerator / denominator);
}

__global__ void mla_kernel(const __hip_bfloat16* ql, const __hip_bfloat16* qr,
 const __hip_bfloat16* kv, const __hip_bfloat16* rk, __hip_bfloat16* out,
 int B, int S, int H, int KV, int R) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * H * KV) return;
  int d = i % KV, x = i / KV, h = x % H, b = x / H;
  float scale = rsqrtf(float(KV + R)), maximum = -INFINITY;
  for (int s = 0; s < S; ++s) {
    float score = 0.0f;
    for (int j = 0; j < KV; ++j) score += bf(ql[(b*H+h)*KV+j]) * bf(kv[(b*S+s)*KV+j]);
    for (int j = 0; j < R; ++j) score += bf(qr[(b*H+h)*R+j]) * bf(rk[(b*S+s)*R+j]);
    maximum = fmaxf(maximum, score * scale);
  }
  float denominator = 0.0f, numerator = 0.0f;
  for (int s = 0; s < S; ++s) {
    float score = 0.0f;
    for (int j = 0; j < KV; ++j) score += bf(ql[(b*H+h)*KV+j]) * bf(kv[(b*S+s)*KV+j]);
    for (int j = 0; j < R; ++j) score += bf(qr[(b*H+h)*R+j]) * bf(rk[(b*S+s)*R+j]);
    float weight = expf(score * scale - maximum);
    denominator += weight; numerator += weight * bf(kv[(b*S+s)*KV+d]);
  }
  out[i] = __float2bfloat16(numerator / denominator);
}

__global__ void paged_attention_kernel(const __hip_bfloat16* q,
 const __hip_bfloat16* kc, const __hip_bfloat16* vc, const int32_t* table,
 const int32_t* lengths, __hip_bfloat16* out, int B, int HQ, int HK,
 int BLOCK, int MAX_BLOCKS, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * HQ * D) return;
  int d = i % D, x = i / D, hq = x % HQ, b = x / HQ, hk = hq / (HQ / HK);
  int length = lengths[b]; float scale = rsqrtf(float(D)), maximum = -INFINITY;
  for (int s = 0; s < length; ++s) {
    int block = table[b*MAX_BLOCKS+s/BLOCK], off = s % BLOCK; float score = 0.0f;
    for (int j = 0; j < D; ++j)
      score += bf(q[(b*HQ+hq)*D+j]) * bf(kc[((block*BLOCK+off)*HK+hk)*D+j]);
    maximum = fmaxf(maximum, score * scale);
  }
  float denominator = 0.0f, numerator = 0.0f;
  for (int s = 0; s < length; ++s) {
    int block = table[b*MAX_BLOCKS+s/BLOCK], off = s % BLOCK; float score = 0.0f;
    for (int j = 0; j < D; ++j)
      score += bf(q[(b*HQ+hq)*D+j]) * bf(kc[((block*BLOCK+off)*HK+hk)*D+j]);
    float weight = expf(score * scale - maximum);
    denominator += weight; numerator += weight * bf(vc[((block*BLOCK+off)*HK+hk)*D+d]);
  }
  out[i] = __float2bfloat16(numerator / denominator);
}

torch::Tensor mha(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==4, "q must be HIP bf16 BSHD");
  int B=q.size(0), SQ=q.size(1), HQ=q.size(2), D=q.size(3), SK=k.size(1), HK=k.size(2);
  TORCH_CHECK(k.sizes()==at::IntArrayRef({B,SK,HK,D}) && v.sizes()==k.sizes() && HQ%HK==0, "MHA shape mismatch");
  auto out=torch::empty_like(q); int64_t n=out.numel();
  hipLaunchKernelGGL(mha_kernel, dim3((n+255)/256), dim3(256), 0, at::hip::getCurrentHIPStream(),
    (__hip_bfloat16*)q.data_ptr(),(__hip_bfloat16*)k.data_ptr(),(__hip_bfloat16*)v.data_ptr(),
    (__hip_bfloat16*)out.data_ptr(),B,SQ,SK,HQ,HK,D);
  TORCH_CHECK(hipGetLastError()==hipSuccess, "mha launch failed"); return out;
}
torch::Tensor mla(torch::Tensor ql, torch::Tensor qr, torch::Tensor kv, torch::Tensor rk) {
  TORCH_CHECK(ql.is_cuda() && ql.scalar_type()==at::kBFloat16 && ql.dim()==3, "latent query must be HIP bf16");
  int B=ql.size(0), H=ql.size(1), KV=ql.size(2), S=kv.size(1), R=qr.size(2);
  TORCH_CHECK(qr.sizes()==at::IntArrayRef({B,H,R}) && kv.sizes()==at::IntArrayRef({B,S,KV}) &&
              rk.sizes()==at::IntArrayRef({B,S,R}), "MLA shape mismatch");
  auto out=torch::empty_like(ql); int64_t n=out.numel();
  hipLaunchKernelGGL(mla_kernel, dim3((n+255)/256), dim3(256), 0, at::hip::getCurrentHIPStream(),
    (__hip_bfloat16*)ql.data_ptr(),(__hip_bfloat16*)qr.data_ptr(),(__hip_bfloat16*)kv.data_ptr(),
    (__hip_bfloat16*)rk.data_ptr(),(__hip_bfloat16*)out.data_ptr(),B,S,H,KV,R);
  TORCH_CHECK(hipGetLastError()==hipSuccess, "mla launch failed"); return out;
}
torch::Tensor paged_attention(torch::Tensor q, torch::Tensor kc, torch::Tensor vc,
 torch::Tensor table, torch::Tensor lengths) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==3, "q must be HIP bf16 BHD");
  int B=q.size(0), HQ=q.size(1), D=q.size(2), BLOCK=kc.size(1), HK=kc.size(2), MB=table.size(1);
  TORCH_CHECK(vc.sizes()==kc.sizes() && kc.size(3)==D && HQ%HK==0 &&
              table.scalar_type()==at::kInt && lengths.scalar_type()==at::kInt, "paged shape mismatch");
  auto out=torch::empty_like(q); int64_t n=out.numel();
  hipLaunchKernelGGL(paged_attention_kernel, dim3((n+255)/256), dim3(256), 0, at::hip::getCurrentHIPStream(),
    (__hip_bfloat16*)q.data_ptr(),(__hip_bfloat16*)kc.data_ptr(),(__hip_bfloat16*)vc.data_ptr(),
    table.data_ptr<int32_t>(),lengths.data_ptr<int32_t>(),(__hip_bfloat16*)out.data_ptr(),
    B,HQ,HK,BLOCK,MB,D);
  TORCH_CHECK(hipGetLastError()==hipSuccess, "paged attention launch failed"); return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mha", &mha); m.def("mla", &mla); m.def("paged_attention", &paged_attention);
}
"""
_MODULE = None
def _module():
    global _MODULE
    if _MODULE is None:
        name = "geak_attention_" + hashlib.sha256(_SOURCE.encode()).hexdigest()[:12]
        _MODULE = load_inline(name=name, cpp_sources="", cuda_sources=_SOURCE,
            functions=None, extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
            with_cuda=True, verbose=False)
    return _MODULE

def mha(q, k, v): return _module().mha(q, k, v)
def mla(q_latent, q_rope, latent_kv, rope_k): return _module().mla(q_latent, q_rope, latent_kv, rope_k)
def paged_attention(q, k_cache, v_cache, block_table, seq_lens):
    return _module().paged_attention(q, k_cache, v_cache, block_table, seq_lens)
'''


def render_attention_kernel(
    family: str, language: str, contract: Mapping[str, Any]
) -> str:
    """Render a standalone custom HIP or Triton attention candidate."""
    normalized_family = _family(family)
    normalized_language = str(language).strip().lower()
    _validate(normalized_family, normalized_language, contract)
    if normalized_language == "hip":
        return _HIP_SOURCE
    return {
        "mha": _TRITON_MHA,
        "mla": _TRITON_MLA,
        "paged_attention": _TRITON_PAGED,
    }[normalized_family]


_RUNNER = r'''import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILY = __FAMILY__
DIMS = __DIMS__
SEED = 942_031

def require_gfx942():
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("a ROCm GPU is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx942":
        raise RuntimeError(f"expected gfx942, found {arch}")

def make_case(device="cuda"):
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    rand = lambda shape: torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    if FAMILY == "mha":
        d = DIMS
        return (rand((d["B"], d["SQ"], d["HQ"], d["D"])),
                rand((d["B"], d["SK"], d["HK"], d["D"])),
                rand((d["B"], d["SK"], d["HK"], d["D"])))
    if FAMILY == "mla":
        d = DIMS
        return (rand((d["B"], d["H"], d["KV"])),
                rand((d["B"], d["H"], d["ROPE"])),
                rand((d["B"], d["S"], d["KV"])),
                rand((d["B"], d["S"], d["ROPE"])))
    d = DIMS
    blocks_per_seq = (d["S"] + d["BLOCK"] - 1) // d["BLOCK"]
    total_blocks = d["B"] * blocks_per_seq
    q = rand((d["B"], d["HQ"], d["D"]))
    kc = rand((total_blocks, d["BLOCK"], d["HK"], d["D"]))
    vc = rand((total_blocks, d["BLOCK"], d["HK"], d["D"]))
    table = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(d["B"], blocks_per_seq)
    lengths = torch.tensor([max(1, d["S"] - i) for i in range(d["B"])],
                           device=device, dtype=torch.int32)
    return q, kc, vc, table, lengths

def fp32_pytorch_oracle(args):
    if FAMILY == "mha":
        q, k, v = (x.float() for x in args)
        B, SQ, HQ, D = q.shape
        SK, HK = k.shape[1], k.shape[2]
        group = HQ // HK
        k = k.repeat_interleave(group, dim=2)
        v = v.repeat_interleave(group, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(D)
        qpos = torch.arange(SQ, device=q.device) + SK - SQ
        kpos = torch.arange(SK, device=q.device)
        causal = kpos[None, :] <= qpos[:, None]
        probabilities = torch.softmax(scores.masked_fill(~causal[None, None], float("-inf")), dim=-1)
        return torch.einsum("bhqk,bkhd->bqhd", probabilities, v).to(torch.bfloat16)
    if FAMILY == "mla":
        q_latent, q_rope, latent_kv, rope_k = (x.float() for x in args)
        width = q_latent.shape[-1] + q_rope.shape[-1]
        scores = (torch.einsum("bhc,bsc->bhs", q_latent, latent_kv) +
                  torch.einsum("bhr,bsr->bhs", q_rope, rope_k)) / math.sqrt(width)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.einsum("bhs,bsc->bhc", probabilities, latent_kv).to(torch.bfloat16)
    q, kc, vc, table, lengths = args
    q = q.float()
    outputs = []
    block = kc.shape[1]
    group = q.shape[1] // kc.shape[2]
    for b in range(q.shape[0]):
        count = int(lengths[b].item())
        logical = torch.arange(count, device=q.device)
        physical = table[b, torch.div(logical, block, rounding_mode="floor")].long()
        offsets = logical % block
        keys = kc[physical, offsets].float().repeat_interleave(group, dim=1)
        values = vc[physical, offsets].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,shd->hs", q[b], keys) / math.sqrt(q.shape[-1])
        outputs.append(torch.einsum("hs,shd->hd", torch.softmax(scores, dim=-1), values))
    return torch.stack(outputs).to(torch.bfloat16)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    require_gfx942()
    args = make_case()
    candidate = getattr(importlib.import_module("kernel"), FAMILY)
    if mode == "compile":
        candidate(*args); torch.cuda.synchronize(); print("Compile: OK"); return
    if mode == "correctness":
        actual = candidate(*args)
        torch.testing.assert_close(actual, fp32_pytorch_oracle(args), rtol=2e-2, atol=2e-2)
        print("Correctness: OK"); return
    for _ in range(3): candidate(*args)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10): candidate(*args)
    end.record(); end.synchronize()
    latency = start.elapsed_time(end) / 10.0
    if not latency > 0.0: raise AssertionError("invalid latency")
    build = ROOT / "build"; build.mkdir(exist_ok=True)
    report = {"test_cases": [{"test_case_id": FAMILY, "execution_time_ms": latency}]}
    (build / "performance_report.json").write_text(json.dumps(report), encoding="utf-8")
    print(f"Perf: {latency:.6f} ms ({FAMILY})")

if __name__ == "__main__":
    main()
'''


def render_attention_runner(family: str, contract: Mapping[str, Any]) -> str:
    """Render the compile/correctness/performance runner for one family."""
    normalized_family = _family(family)
    _validate(normalized_family, None, contract)
    dims = _dims(normalized_family, contract)
    return _RUNNER.replace("__FAMILY__", repr(normalized_family)).replace(
        "__DIMS__", repr(dims)
    )


__all__ = ["render_attention_kernel", "render_attention_runner"]
