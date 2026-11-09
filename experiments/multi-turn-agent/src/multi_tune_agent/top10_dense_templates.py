"""Standalone dense canonical kernels for the difficult Top10 families.

The public renderers return a complete ``kernel.py`` and GEAK task runner.
Generated candidates contain only independently written HIP or Triton kernels;
PyTorch is used by the runner solely to construct inputs, form an independent
oracle, and time launches.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


SUPPORTED_FAMILIES = frozenset({"fused_moe", "blockscale_gemm"})
SUPPORTED_LANGUAGES = frozenset({"hip", "triton"})
SUPPORTED_ARCH = "gfx942"


class DenseTemplateError(ValueError):
    """The requested dense template has an unsupported or ambiguous contract."""


def _dtype(contract: Mapping[str, Any], role: str) -> str:
    nested = contract.get("dtype")
    value = nested.get(role) if isinstance(nested, Mapping) else None
    value = value or contract.get(f"{role}_dtype")
    return str(value or "").lower()


def _shape(contract: Mapping[str, Any]) -> dict[str, int]:
    raw = contract.get("shape")
    if not isinstance(raw, Mapping):
        raise DenseTemplateError("contract.shape must be a mapping")
    try:
        result = {str(key).upper(): int(value) for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise DenseTemplateError("shape dimensions must be integers") from exc
    if not result or any(value <= 0 for value in result.values()):
        raise DenseTemplateError("shape dimensions must be positive")
    return result


def _normal_weight_dtype(family: str, contract: Mapping[str, Any]) -> str:
    value = _dtype(contract, "weight")
    if family == "blockscale_gemm" and not value:
        value = _dtype(contract, "input")
    mode = str(contract.get("mode") or "").lower()
    if not value:
        if "fp8" in mode:
            value = "fp8_e4m3fnuz"
        elif "int8" in mode:
            value = "int8"
        else:
            value = "bf16"
    aliases = {
        "float8_e4m3fnuz": "fp8_e4m3fnuz",
        "torch.float8_e4m3fnuz": "fp8_e4m3fnuz",
        "fp8_e4m3_arch_specific": "fp8_e4m3fnuz",
        "bfloat16": "bf16",
    }
    value = aliases.get(value, value)
    allowed = {"bf16", "int8", "fp8_e4m3fnuz"}
    if family == "blockscale_gemm":
        allowed.remove("bf16")
    if value not in allowed:
        raise DenseTemplateError(f"unsupported {family} weight dtype {value!r}")
    return value


def _validate(
    family: str, language: str | None, contract: Mapping[str, Any]
) -> tuple[str, dict[str, int], str, int]:
    family = family.strip().lower().replace("-", "_")
    if family not in SUPPORTED_FAMILIES:
        raise DenseTemplateError(f"unsupported dense family {family!r}")
    if language is not None:
        language = language.strip().lower()
        if language not in SUPPORTED_LANGUAGES:
            raise DenseTemplateError(f"unsupported language {language!r}")
    operator = str(contract.get("operator") or family).lower().replace("-", "_")
    if operator != family:
        raise DenseTemplateError(f"operator {operator!r} does not match {family!r}")
    target = str(contract.get("target_gpu") or contract.get("architecture") or SUPPORTED_ARCH)
    if target.lower() != SUPPORTED_ARCH:
        raise DenseTemplateError("dense canonical templates require gfx942")
    shape = _shape(contract)
    expected = (
        {"TOKENS", "MODEL", "INTER", "EXPERTS", "TOPK"}
        if family == "fused_moe"
        else {"M", "N", "K"}
    )
    if not expected.issubset(shape):
        missing = ", ".join(sorted(expected - shape.keys()))
        raise DenseTemplateError("contract shape is missing " + missing)
    if family == "fused_moe" and shape["TOPK"] > shape["EXPERTS"]:
        raise DenseTemplateError("TOPK cannot exceed EXPERTS")
    if _dtype(contract, "output") not in ("", "bf16", "bfloat16"):
        raise DenseTemplateError("dense templates produce bf16")
    weight_dtype = _normal_weight_dtype(family, contract)
    mode = str(contract.get("mode") or "").lower()
    input_dtype = _dtype(contract, "input")
    aliases = {
        "bfloat16": "bf16",
        "float8_e4m3fnuz": "fp8_e4m3fnuz",
        "torch.float8_e4m3fnuz": "fp8_e4m3fnuz",
        "fp8_e4m3_arch_specific": "fp8_e4m3fnuz",
    }
    input_dtype = aliases.get(input_dtype, input_dtype)
    if family == "fused_moe" and input_dtype not in ("", "bf16"):
        raise DenseTemplateError("fused_moe requires bf16 activations")
    if family == "fused_moe" and mode not in {"", "silu", "fp8", "int8"}:
        raise DenseTemplateError(f"unsupported fused_moe mode {mode!r}")
    if family == "blockscale_gemm" and input_dtype not in ("", weight_dtype):
        raise DenseTemplateError(
            "blockscale_gemm activation and weight dtypes must match"
        )
    if family == "blockscale_gemm" and mode not in {
        "",
        "fp8_block128",
        "int8_block128",
        "preshuffle",
    }:
        raise DenseTemplateError(f"unsupported blockscale_gemm mode {mode!r}")
    block_k = int(contract.get("scale_block_k") or 128)
    if family == "blockscale_gemm":
        layout = str(contract.get("layout") or "TN").upper()
        if layout != "TN":
            raise DenseTemplateError("blockscale_gemm requires TN layout")
        accum_dtype = _dtype(contract, "accum")
        expected_accum = "int32" if weight_dtype == "int8" else "fp32"
        if accum_dtype and accum_dtype not in {
            expected_accum,
            "float32" if expected_accum == "fp32" else expected_accum,
        }:
            raise DenseTemplateError(
                f"{weight_dtype} blockscale_gemm requires {expected_accum} accumulation"
            )
        scale = contract.get("scale")
        if not isinstance(scale, Mapping):
            raise DenseTemplateError("blockscale_gemm requires explicit block scales")
        if "activation_block" in scale or "weight_block" in scale:
            if scale.get("activation_block") != [1, 128] or scale.get(
                "weight_block"
            ) != [128, 128]:
                raise DenseTemplateError(
                    "only [1,128]/[128,128] block scales are supported"
                )
        else:
            granularities = {
                str(scale.get(key) or "").lower()
                for key in ("activation", "weight")
            }
            if granularities != {"block128"}:
                raise DenseTemplateError(
                    "only block128 activation/weight scales are supported"
                )
        if block_k != 128:
            raise DenseTemplateError("blockscale_gemm requires scale_block_k=128")
        if mode == "preshuffle" and (
            shape["N"] % 16 != 0 or shape["K"] % 32 != 0
        ):
            raise DenseTemplateError(
                "preshuffled blockscale_gemm requires N%16==0 and K%32==0"
            )
    projection = str(contract.get("projection") or "one_projection").lower()
    if family == "fused_moe" and projection not in {"one_projection", "one-projection"}:
        raise DenseTemplateError("fused_moe supports only the one-projection contract")
    return family, shape, weight_dtype, block_k


def _hip_common_source(body: str, py_body: str, name: str) -> str:
    return f'''from __future__ import annotations

import hashlib
import os

import torch
from torch.utils.cpp_extension import load_inline

_HIP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>

__device__ __forceinline__ float fp8_e4m3fnuz_to_float(uint8_t raw) {{
  if (raw == 0x80) return NAN;
  const float sign = (raw & 0x80) ? -1.0f : 1.0f;
  const int exponent = (raw >> 3) & 0x0f;
  const int mantissa = raw & 0x07;
  if (exponent == 0)
    return sign * ldexpf(static_cast<float>(mantissa), -10);
  return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                       exponent - 8);
}}

{body}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        digest = hashlib.sha256(_HIP_SOURCE.encode("utf-8")).hexdigest()[:16]
        _MODULE = load_inline(
            name="geak_dense_{name}_" + digest,
            cpp_sources="",
            cuda_sources=_HIP_SOURCE,
            functions=None,
            extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
            with_cuda=True,
            verbose=os.environ.get("GEAK_VERBOSE_BUILD") == "1",
        )
    return _MODULE


{py_body}
'''


def _hip_fused_moe(weight_dtype: str) -> str:
    if weight_dtype == "bf16":
        c_weight = "__hip_bfloat16"
        load_weight = "__bfloat162float(weight[index])"
        dtype_code = "0"
    elif weight_dtype == "int8":
        c_weight = "int8_t"
        load_weight = "static_cast<float>(weight[index])"
        dtype_code = "1"
    else:
        c_weight = "uint8_t"
        load_weight = "fp8_e4m3fnuz_to_float(weight[index])"
        dtype_code = "2"
    body = f"""
__global__ void fused_moe_one_projection_kernel(
    const __hip_bfloat16* __restrict__ x,
    const {c_weight}* __restrict__ weight,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ routing_weights,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int T, int D, int I, int E, int TOPK) {{
  const int feature = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = blockIdx.y;
  if (token >= T || feature >= I) return;
  float mixed = 0.0f;
  for (int route = 0; route < TOPK; ++route) {{
    const int expert = expert_ids[token * TOPK + route];
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {{
      const int64_t index =
          (static_cast<int64_t>(expert) * I + feature) * D + d;
      dot += __bfloat162float(x[token * D + d]) * {load_weight};
    }}
    if ({dtype_code} != 0) dot *= weight_scale[expert * I + feature];
    const float silu = dot / (1.0f + expf(-dot));
    mixed += routing_weights[token * TOPK + route] * silu;
  }}
  out[token * I + feature] = __float2bfloat16(mixed);
}}

torch::Tensor fused_moe(
    torch::Tensor x, torch::Tensor weight, torch::Tensor expert_ids,
    torch::Tensor routing_weights, torch::Tensor weight_scale) {{
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && expert_ids.is_cuda() &&
              routing_weights.is_cuda() && weight_scale.is_cuda(),
              "all inputs must be ROCm tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt, "expert_ids must be int32");
  TORCH_CHECK(routing_weights.scalar_type() == at::kFloat,
              "routing_weights must be fp32");
  TORCH_CHECK(weight_scale.scalar_type() == at::kFloat,
              "weight_scale must be fp32");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() &&
              expert_ids.is_contiguous() && routing_weights.is_contiguous() &&
              weight_scale.is_contiguous(), "all inputs must be contiguous");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 3 && expert_ids.dim() == 2 &&
              routing_weights.sizes() == expert_ids.sizes(),
              "expected x[T,D], weight[E,I,D], and routing[T,TOPK]");
  const int T = x.size(0), D = x.size(1), E = weight.size(0);
  const int I = weight.size(1), TOPK = expert_ids.size(1);
  TORCH_CHECK(T > 0 && D > 0 && E > 0 && I > 0 && TOPK > 0 && TOPK <= E,
              "fused_moe dimensions are invalid");
  TORCH_CHECK(weight.size(2) == D && expert_ids.size(0) == T,
              "fused_moe shape mismatch");
  TORCH_CHECK({dtype_code} == 0 || weight_scale.numel() == E * I,
              "quantized weights require one fp32 scale per expert/output");
  auto out = torch::empty({{T, I}}, x.options());
  dim3 block(256), grid((I + 255) / 256, T);
  hipLaunchKernelGGL(
      fused_moe_one_projection_kernel, grid, block, 0,
      at::hip::getCurrentHIPStream(),
      reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const {c_weight}*>(weight.data_ptr()),
      expert_ids.data_ptr<int32_t>(), routing_weights.data_ptr<float>(),
      weight_scale.data_ptr<float>(),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr()), T, D, I, E, TOPK);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "fused_moe launch failed");
  return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("fused_moe", &fused_moe);
}}
"""
    expected = {
        "bf16": "torch.bfloat16",
        "int8": "torch.int8",
        "fp8_e4m3fnuz": "torch.float8_e4m3fnuz",
    }[weight_dtype]
    py_body = f'''def fused_moe(x, weight, expert_ids, routing_weights, weight_scale=None):
    if weight.dtype != {expected}:
        raise TypeError("weight must use the contract dtype")
    if weight_scale is None:
        if weight.dtype != torch.bfloat16:
            raise ValueError("quantized expert weights require weight_scale")
        # The custom kernel does not read this argument for bf16 weights.
        weight_scale = routing_weights
    return _module().fused_moe(
        x, weight, expert_ids, routing_weights, weight_scale
    )'''
    return _hip_common_source(body, py_body, "fused_moe")


def _hip_blockscale(weight_dtype: str, block_k: int, preshuffle: bool) -> str:
    if weight_dtype == "int8":
        scalar = "int8_t"
        load = "static_cast<float>({value})"
        integer = "true"
    else:
        scalar = "uint8_t"
        load = "fp8_e4m3fnuz_to_float({value})"
        integer = "false"
    weight_index = (
        "((((n / 16) * (K / 32) + (k / 32)) * 2 + "
        "((k % 32) / 16)) * 16 + (n % 16)) * 16 + (k % 16)"
        if preshuffle
        else "n * K + k"
    )
    body = f"""
constexpr int SCALE_BLOCK_K = {block_k};
constexpr int SCALE_BLOCK_N = 128;

__global__ void blockscale_gemm_kernel(
    const {scalar}* __restrict__ a,
    const {scalar}* __restrict__ weight,
    const float* __restrict__ a_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int M, int N, int K, int K_BLOCKS) {{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float result = 0.0f;
  for (int kb = 0; kb < K_BLOCKS; ++kb) {{
    const int begin = kb * SCALE_BLOCK_K;
    const int end = begin + SCALE_BLOCK_K < K ? begin + SCALE_BLOCK_K : K;
    {'int32_t block_acc_i = 0;' if weight_dtype == 'int8' else 'float block_acc = 0.0f;'}
    for (int k = begin; k < end; ++k) {{
      {'block_acc_i += static_cast<int32_t>(a[m * K + k]) * static_cast<int32_t>(weight[' + weight_index + ']);' if weight_dtype == 'int8' else f'block_acc += {load.format(value="a[m * K + k]")} * {load.format(value="weight[" + weight_index + "]")};'}
    }}
    const float block_dot = {f'static_cast<float>(block_acc_i)' if weight_dtype == 'int8' else 'block_acc'};
    result += block_dot * a_scale[m * K_BLOCKS + kb] *
              weight_scale[(n / SCALE_BLOCK_N) * K_BLOCKS + kb];
  }}
  out[m * N + n] = __float2bfloat16(result);
}}

torch::Tensor blockscale_gemm(
    torch::Tensor a, torch::Tensor weight, torch::Tensor a_scale,
    torch::Tensor weight_scale) {{
  TORCH_CHECK(a.is_cuda() && weight.is_cuda() && a_scale.is_cuda() &&
              weight_scale.is_cuda(), "all inputs must be ROCm tensors");
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat &&
              weight_scale.scalar_type() == at::kFloat, "scales must be fp32");
  TORCH_CHECK(a.is_contiguous() && weight.is_contiguous() &&
              a_scale.is_contiguous() && weight_scale.is_contiguous(),
              "all inputs must be contiguous");
  TORCH_CHECK(a.dim() == 2 && weight.dim() == 2 && a.size(1) == weight.size(1),
              "expected A[M,K] and weight[N,K]");
  const int M = a.size(0), K = a.size(1), N = weight.size(0);
  TORCH_CHECK({str(not preshuffle).lower()} || (N % 16 == 0 && K % 32 == 0),
              "preshuffled weight requires N%16==0 and K%32==0");
  const int K_BLOCKS = (K + SCALE_BLOCK_K - 1) / SCALE_BLOCK_K;
  const int N_BLOCKS = (N + SCALE_BLOCK_N - 1) / SCALE_BLOCK_N;
  TORCH_CHECK(a_scale.sizes() == at::IntArrayRef({{M, K_BLOCKS}}) &&
              weight_scale.sizes() == at::IntArrayRef({{N_BLOCKS, K_BLOCKS}}),
              "scales must be [M,K_BLOCKS] and [ceil(N/128),K_BLOCKS]");
  auto out = torch::empty({{M, N}}, a.options().dtype(at::kBFloat16));
  dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
  hipLaunchKernelGGL(
      blockscale_gemm_kernel, grid, block, 0, at::hip::getCurrentHIPStream(),
      reinterpret_cast<const {scalar}*>(a.data_ptr()),
      reinterpret_cast<const {scalar}*>(weight.data_ptr()),
      a_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
      reinterpret_cast<__hip_bfloat16*>(out.data_ptr()), M, N, K, K_BLOCKS);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "blockscale_gemm launch failed");
  return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("blockscale_gemm", &blockscale_gemm);
}}
"""
    expected = (
        "torch.int8" if weight_dtype == "int8" else "torch.float8_e4m3fnuz"
    )
    py_body = f'''def blockscale_gemm(a, weight, a_scale, weight_scale):
    if a.dtype != {expected} or weight.dtype != {expected}:
        raise TypeError("A and weight must use the exact contract dtype")
    return _module().blockscale_gemm(a, weight, a_scale, weight_scale)'''
    return _hip_common_source(body, py_body, "blockscale_gemm")


def _triton_fused_moe(weight_dtype: str) -> str:
    quantized = weight_dtype != "bf16"
    expected = {
        "bf16": "torch.bfloat16",
        "int8": "torch.int8",
        "fp8_e4m3fnuz": "torch.float8_e4m3fnuz",
    }[weight_dtype]
    return f'''from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_moe_one_projection_kernel(
    x, weight, expert_ids, routing_weights, weight_scale, out,
    T: tl.constexpr, D: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    feature = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    xv = tl.load(x + token * D + d, mask=d_mask, other=0.0).to(tl.float32)
    mixed = 0.0
    for route in range(0, TOPK):
        expert = tl.load(expert_ids + token * TOPK + route)
        w = tl.load(
            weight + (expert * I + feature) * D + d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(xv * w, axis=0)
        if {quantized!r}:
            dot *= tl.load(weight_scale + expert * I + feature)
        silu = dot * tl.sigmoid(dot)
        mixed += tl.load(routing_weights + token * TOPK + route) * silu
    tl.store(out + token * I + feature, mixed)


def fused_moe(x, weight, expert_ids, routing_weights, weight_scale=None):
    if weight.dtype != {expected}:
        raise TypeError("weight must use the exact contract dtype")
    if x.dtype != torch.bfloat16 or expert_ids.dtype != torch.int32:
        raise TypeError("x must be bf16 and expert_ids must be int32")
    if routing_weights.dtype != torch.float32:
        raise TypeError("routing_weights must be fp32")
    if not (
        x.is_cuda
        and weight.is_cuda
        and expert_ids.is_cuda
        and routing_weights.is_cuda
    ):
        raise ValueError("all inputs must be ROCm tensors")
    if not (
        x.is_contiguous()
        and weight.is_contiguous()
        and expert_ids.is_contiguous()
        and routing_weights.is_contiguous()
    ):
        raise ValueError("all inputs must be contiguous")
    if weight_scale is None:
        if weight.dtype != torch.bfloat16:
            raise ValueError("quantized expert weights require weight_scale")
        # The custom kernel does not read this argument for bf16 weights.
        weight_scale = routing_weights
    T, D = x.shape
    E, I, WD = weight.shape
    if (
        WD != D
        or expert_ids.shape[0] != T
        or routing_weights.shape != expert_ids.shape
        or not 0 < expert_ids.shape[1] <= E
    ):
        raise ValueError("fused_moe shape mismatch")
    if weight_scale.dtype != torch.float32 or not weight_scale.is_contiguous():
        raise TypeError("weight_scale must be contiguous fp32")
    out = torch.empty((T, I), device=x.device, dtype=torch.bfloat16)
    _fused_moe_one_projection_kernel[(T, I)](
        x, weight, expert_ids, routing_weights, weight_scale, out,
        T=T, D=D, I=I, TOPK=expert_ids.shape[1],
        BLOCK_D=triton.next_power_of_2(D), num_warps=4,
    )
    return out
'''


def _triton_blockscale(weight_dtype: str, block_k: int, preshuffle: bool) -> str:
    expected = (
        "torch.int8" if weight_dtype == "int8" else "torch.float8_e4m3fnuz"
    )
    integer = weight_dtype == "int8"
    dot = (
        "tl.sum(av.to(tl.int32) * wv.to(tl.int32), axis=0).to(tl.float32)"
        if integer
        else "tl.sum(av.to(tl.float32) * wv.to(tl.float32), axis=0)"
    )
    weight_offset = (
        "((((n // 16) * (K // 32) + (k // 32)) * 2 + "
        "((k % 32) // 16)) * 16 + (n % 16)) * 16 + (k % 16)"
        if preshuffle
        else "n * K + k"
    )
    return f'''from __future__ import annotations

import torch
import triton
import triton.language as tl

SCALE_BLOCK_K = {block_k}
SCALE_BLOCK_N = 128


@triton.jit
def _blockscale_gemm_kernel(
    a, weight, a_scale, weight_scale, out,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    K_BLOCKS: tl.constexpr, BLOCK_K: tl.constexpr,
    SCALE_K: tl.constexpr, SCALE_N: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_K)
    result = 0.0
    for kb in range(0, K_BLOCKS):
        block_acc = 0.0
        for tile in range(0, SCALE_K // BLOCK_K):
            k = kb * SCALE_K + tile * BLOCK_K + offsets
            av = tl.load(a + m * K + k)
            wv = tl.load(weight + {weight_offset})
            block_acc += {dot}
        # Scales are deliberately consumed here, before the next K block.
        result += block_acc * tl.load(a_scale + m * K_BLOCKS + kb) * tl.load(
            weight_scale + (n // SCALE_N) * K_BLOCKS + kb
        )
    tl.store(out + m * N + n, result)


def blockscale_gemm(a, weight, a_scale, weight_scale):
    if a.dtype != {expected} or weight.dtype != {expected}:
        raise TypeError("A and weight must use the exact contract dtype")
    if a_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        raise TypeError("block scales must be fp32")
    if not (a.is_cuda and weight.is_cuda and a_scale.is_cuda and weight_scale.is_cuda):
        raise ValueError("all inputs must be ROCm tensors")
    if not (
        a.is_contiguous()
        and weight.is_contiguous()
        and a_scale.is_contiguous()
        and weight_scale.is_contiguous()
    ):
        raise ValueError("all inputs must be contiguous")
    M, K = a.shape
    N, WK = weight.shape
    if K % SCALE_BLOCK_K:
        raise ValueError("K must be divisible by the scale block")
    if {preshuffle!r} and (N % 16 or K % 32):
        raise ValueError("preshuffled weight requires N%16==0 and K%32==0")
    K_BLOCKS = triton.cdiv(K, SCALE_BLOCK_K)
    N_BLOCKS = triton.cdiv(N, SCALE_BLOCK_N)
    if (
        WK != K
        or a_scale.shape != (M, K_BLOCKS)
        or weight_scale.shape != (N_BLOCKS, K_BLOCKS)
    ):
        raise ValueError("blockscale_gemm shape mismatch")
    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    _blockscale_gemm_kernel[(M, N)](
        a, weight, a_scale, weight_scale, out,
        M=M, N=N, K=K, K_BLOCKS=K_BLOCKS, BLOCK_K=32,
        SCALE_K=SCALE_BLOCK_K, SCALE_N=SCALE_BLOCK_N, num_warps=1,
    )
    return out
'''


def render_dense_kernel(
    family: str, language: str, contract: Mapping[str, Any]
) -> str:
    """Render a standalone custom-kernel module for one dense Top10 contract."""

    family, _, weight_dtype, block_k = _validate(family, language, contract)
    language = language.strip().lower()
    if family == "fused_moe":
        return (
            _hip_fused_moe(weight_dtype)
            if language == "hip"
            else _triton_fused_moe(weight_dtype)
        )
    return (
        _hip_blockscale(
            weight_dtype, block_k, str(contract.get("mode") or "").lower() == "preshuffle"
        )
        if language == "hip"
        else _triton_blockscale(
            weight_dtype, block_k, str(contract.get("mode") or "").lower() == "preshuffle"
        )
    )


def _runner_dimensions(family: str, shape: Mapping[str, int], block_k: int) -> dict[str, int]:
    if family == "fused_moe":
        return {
            "TOKENS": min(shape["TOKENS"], 8),
            "MODEL": min(shape["MODEL"], 32),
            "INTER": min(shape["INTER"], 24),
            "EXPERTS": min(shape["EXPERTS"], 4),
            "TOPK": min(shape["TOPK"], min(shape["EXPERTS"], 2)),
        }
    # Preserve two independently scaled K blocks, including a partial tail.
    return {
        "M": min(shape["M"], 8),
        "N": min(shape["N"], 16),
        "K": min(shape["K"], block_k + 32),
    }


def render_dense_runner(family: str, contract: Mapping[str, Any]) -> str:
    """Render a deterministic gfx942 compile/correctness/performance runner."""

    family, shape, weight_dtype, block_k = _validate(family, None, contract)
    dims = _runner_dimensions(family, shape, block_k)
    contract_json = json.dumps(dict(contract), sort_keys=True)
    if family == "fused_moe":
        make_case = f'''    T, D, I, E, TOPK = (
        TEST_SHAPE["TOKENS"], TEST_SHAPE["MODEL"], TEST_SHAPE["INTER"],
        TEST_SHAPE["EXPERTS"], TEST_SHAPE["TOPK"],
    )
    x = (torch.randn((T, D), device="cuda") * 0.125).to(torch.bfloat16)
    expert_ids = (
        torch.arange(T * TOPK, device="cuda", dtype=torch.int32).reshape(T, TOPK) % E
    ).contiguous()
    routing = torch.rand((T, TOPK), device="cuda", dtype=torch.float32)
    routing = (routing / routing.sum(dim=1, keepdim=True)).contiguous()
    if WEIGHT_DTYPE == "bf16":
        weight = (torch.randn((E, I, D), device="cuda") * 0.125).to(torch.bfloat16)
        scale = None
    elif WEIGHT_DTYPE == "int8":
        weight = torch.randint(-5, 6, (E, I, D), device="cuda", dtype=torch.int8)
        scale = (torch.rand((E, I), device="cuda") * 0.025 + 0.005).float()
    else:
        raw = torch.randn((E, I, D), device="cuda").clamp(-2, 2)
        weight = raw.to(torch.float8_e4m3fnuz)
        scale = (torch.rand((E, I), device="cuda") * 0.025 + 0.005).float()
    return x, weight.contiguous(), expert_ids, routing, scale'''
        oracle = '''    x, weight, expert_ids, routing, scale = args
    output = torch.zeros(
        (x.shape[0], weight.shape[1]), device=x.device, dtype=torch.float32
    )
    for token in range(x.shape[0]):
        for route in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, route].item())
            projected = torch.matmul(
                weight[expert].float(), x[token].float()
            )
            if scale is not None:
                projected = projected * scale[expert]
            activated = projected * torch.sigmoid(projected)
            output[token] += routing[token, route] * activated
    return output.to(torch.bfloat16)'''
        call = "candidate(*args)"
        tolerance = "rtol=3.0e-2, atol=6.0e-2"
    else:
        dtype = (
            "torch.int8"
            if weight_dtype == "int8"
            else "torch.float8_e4m3fnuz"
        )
        make_case = f'''    M, N, K = TEST_SHAPE["M"], TEST_SHAPE["N"], TEST_SHAPE["K"]
    if WEIGHT_DTYPE == "int8":
        a = torch.randint(-5, 6, (M, K), device="cuda", dtype=torch.int8)
        weight = torch.randint(-5, 6, (N, K), device="cuda", dtype=torch.int8)
    else:
        a = torch.randn((M, K), device="cuda").clamp(-2, 2).to({dtype})
        weight = torch.randn((N, K), device="cuda").clamp(-2, 2).to({dtype})
    if MODE == "preshuffle":
        weight = (
            weight.view(N // 16, 16, K // 32, 2, 16)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(N, K)
        )
    blocks = (K + SCALE_BLOCK_K - 1) // SCALE_BLOCK_K
    n_blocks = (N + SCALE_BLOCK_N - 1) // SCALE_BLOCK_N
    a_scale = (torch.rand((M, blocks), device="cuda") * 0.05 + 0.01).float()
    weight_scale = (torch.rand((n_blocks, blocks), device="cuda") * 0.05 + 0.01).float()
    # Distinct block scales make post-GEMM row/column scaling observably wrong.
    a_scale[:, 0] *= 0.25
    weight_scale[:, -1] *= 2.0
    return a.contiguous(), weight.contiguous(), a_scale.contiguous(), weight_scale.contiguous()'''
        oracle = '''    a, weight, a_scale, weight_scale = args
    M, K = a.shape
    N = weight.shape[0]
    if MODE == "preshuffle":
        weight = (
            weight.view(N // 16, K // 32, 2, 16, 16)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(N, K)
        )
    blocks = (K + SCALE_BLOCK_K - 1) // SCALE_BLOCK_K
    result = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    for kb in range(blocks):
        start = kb * SCALE_BLOCK_K
        stop = min(start + SCALE_BLOCK_K, K)
        if WEIGHT_DTYPE == "int8":
            # Values and block length keep fp32 matmul integer-exact; the
            # int32 round-trip independently pins integer accumulation.
            dot = torch.matmul(
                a[:, start:stop].float(),
                weight[:, start:stop].float().transpose(0, 1),
            ).to(torch.int32).float()
        else:
            dot = torch.matmul(
                a[:, start:stop].float(),
                weight[:, start:stop].float().transpose(0, 1),
            )
        column_scale = weight_scale[:, kb].repeat_interleave(SCALE_BLOCK_N)[:N]
        result += dot * a_scale[:, kb : kb + 1] * column_scale.reshape(1, -1)
    return result.to(torch.bfloat16)'''
        call = "candidate(*args)"
        tolerance = "rtol=3.0e-2, atol=5.0e-2"
    return f'''from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILY = {family!r}
FROZEN_CONTRACT = json.loads({contract_json!r})
TEST_SHAPE = {dims!r}
WEIGHT_DTYPE = {weight_dtype!r}
MODE = {str(contract.get("mode") or "")!r}
SCALE_BLOCK_K = {block_k}
SCALE_BLOCK_N = 128
SUPPORTED_ARCH = "gfx942"
WARMUP = 2
REPEATS = 5


def require_gfx942():
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]
    if arch != SUPPORTED_ARCH:
        raise RuntimeError("expected gfx942, found " + arch)


def make_case(seed):
    torch.manual_seed(seed)
{make_case}


def fp32_int32_oracle(args):
{oracle}


def invoke(candidate, args):
    return {call}


def compile_kernel(candidate):
    invoke(candidate, make_case(94201))
    torch.cuda.synchronize()


def check_correctness(candidate):
    args = make_case(94202)
    actual = invoke(candidate, args)
    expected = fp32_int32_oracle(args)
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise AssertionError("candidate returned the wrong shape or dtype")
    torch.testing.assert_close(actual, expected, {tolerance})


def benchmark(candidate):
    args = make_case(94203)
    for _ in range(WARMUP):
        invoke(candidate, args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke(candidate, args)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    latency = float(sorted(samples)[len(samples) // 2])
    if not latency > 0.0:
        raise AssertionError("benchmark produced invalid latency")
    case_id = FAMILY + "-" + "-".join(
        key.lower() + str(value) for key, value in TEST_SHAPE.items()
    )
    report = {{
        "architecture": SUPPORTED_ARCH,
        "contract": FROZEN_CONTRACT,
        "test_cases": [
            {{"test_case_id": case_id, "execution_time_ms": latency}}
        ],
    }}
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    (build / "performance_report.json").write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )
    print("Perf: %.6f ms (%s)" % (latency, case_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    mode = parser.parse_args().mode
    require_gfx942()
    candidate = getattr(importlib.import_module("kernel"), FAMILY)
    if mode == "compile":
        compile_kernel(candidate)
        print("Compile: OK")
    elif mode == "correctness":
        check_correctness(candidate)
        print("Correctness: OK")
    else:
        benchmark(candidate)


if __name__ == "__main__":
    main()
'''


__all__ = [
    "DenseTemplateError",
    "SUPPORTED_FAMILIES",
    "SUPPORTED_LANGUAGES",
    "render_dense_kernel",
    "render_dense_runner",
]
