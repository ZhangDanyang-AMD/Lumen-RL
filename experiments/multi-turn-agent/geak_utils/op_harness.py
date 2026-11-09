"""Per-operator standard test harnesses for baseline and candidate benchmarking.

Each operator entry provides:
  - shapes: dict of param_name -> tensor shape or scalar value
  - dtypes: dict of param_name -> torch.dtype (default bf16 if absent)
  - call: optional lambda to call the function with generated args
           (used when argument names don't match standard patterns)

Usage:
    from op_harness import get_harness
    harness = get_harness("gemm")
    args = harness.make_args(func)
    ms = harness.time(func)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


# ── Standard shapes ──────────────────────────────────────────────────────────
# Representative sizes that cover common production workloads on gfx950.
# Not exhaustive — chosen to be fast enough to run 20 iters in <30s.

_B = 4        # batch
_S = 512      # seq_len
_H = 32       # heads (query)
_HK = 8       # key/value heads (GQA)
_D = 128      # head_dim
_E = 256      # expert count (MoE)
_M = 4096     # rows (GEMM M)
_N = 8192     # cols (GEMM N)
_K = 4096     # inner (GEMM K)
_EXPERTS = 8  # MoE experts


@dataclass
class OpHarness:
    """Holds shapes + optional custom caller for one operator family."""
    # param_name -> shape tuple or scalar
    shapes: dict[str, Any] = field(default_factory=dict)
    # param_name -> dtype (default torch.bfloat16)
    dtypes: dict[str, torch.dtype] = field(default_factory=dict)
    # If set, ignore `shapes` and call this to build args list directly
    build_args: Callable[[], list[Any]] | None = None
    n_warmup: int = 5
    n_iter: int = 20

    def _tensor(self, name: str, shape: tuple) -> torch.Tensor:
        dt = self.dtypes.get(name, torch.bfloat16)
        if dt in (torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                  torch.int8, torch.int32, torch.int64):
            nl = name.lower()
            if nl in ('topk_ids', 'expert_ids', 'token_ids',
                      'sorted_token_ids'):
                hi = self.shapes.get('num_experts', _EXPERTS)
                if isinstance(hi, tuple):
                    hi = _EXPERTS
                return torch.randint(0, max(hi, 2), shape, dtype=dt, device="cuda")
            return torch.zeros(shape, dtype=dt, device="cuda")
        return torch.randn(shape, dtype=dt, device="cuda")

    def make_args(self, func: Callable) -> list[Any]:
        """Build argument list from function signature + harness shapes."""
        if self.build_args is not None:
            import inspect as _i
            # Support both build_args() and build_args(func)
            try:
                n_params = len(_i.signature(self.build_args).parameters)
            except (ValueError, TypeError):
                n_params = 0
            return self.build_args(func) if n_params > 0 else self.build_args()
        sig = inspect.signature(func)
        params = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        args = []
        for p in params:
            nl = p.name.lower()
            if nl in self.shapes:
                v = self.shapes[nl]
                if isinstance(v, tuple):
                    args.append(self._tensor(nl, v))
                else:
                    args.append(v)
            else:
                # Generic fallback by name pattern
                args.append(self._generic_arg(nl))
        return args

    def _generic_arg(self, name: str) -> Any:
        scalar_names = {
            "m", "n", "k", "k_dim", "size", "dim", "batch", "batch_size",
            "seq_len", "seqlen", "num_heads", "head_dim", "num_experts",
            "top_k", "stride", "num_tokens", "n_tokens",
        }
        bool_names = {"is_causal", "causal", "training", "use_bias", "transpose"}
        float_names = {"scale", "alpha", "beta", "eps", "epsilon", "dropout_p",
                       "softmax_scale", "sm_scale"}
        if name in scalar_names:
            return 128
        if name in bool_names:
            return False
        if name in float_names:
            return 1.0
        return torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

    def time_gpu_ms(self, func: Callable) -> float:
        """Run func and return GPU kernel time in ms via torch.profiler."""
        try:
            args = self.make_args(func)
        except Exception:
            return 0.0
        return _profile(func, args, self.n_warmup, self.n_iter)


def _profile(func: Callable, args: list, n_warmup: int = 5, n_iter: int = 20) -> float:
    """Return total GPU kernel time per call in ms via torch.profiler."""
    try:
        for _ in range(n_warmup):
            func(*args)
        torch.cuda.synchronize()
    except Exception:
        return 0.0

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(n_iter):
                func(*args)
            torch.cuda.synchronize()
    except Exception:
        return 0.0

    events = [e for e in prof.key_averages() if e.device_time_total > 0]
    if not events:
        return 0.0
    total_us = sum(e.device_time_total for e in events)
    return total_us / 1000.0 / n_iter


# ── Harness registry ─────────────────────────────────────────────────────────

_REGISTRY: dict[str, OpHarness] = {}


def _reg(ops: list[str], harness: OpHarness) -> None:
    for op in ops:
        _REGISTRY[op] = harness


def get_harness(operator: str) -> OpHarness:
    """Return harness for operator, or a generic fallback."""
    op = operator.lower().replace("-", "_")
    return _REGISTRY.get(op, _FALLBACK)


# ── GEMM family ──────────────────────────────────────────────────────────────
# A @ B^T where A:(M,K), B:(N,K)  →  output:(M,N)
_reg(["gemm", "splitk_gemm", "block_scale_gemm", "batched_gemm", "deepgemm",
      "gemm_a16w16", "preshuffle_gemm"], OpHarness(
    shapes={
        # common single-letter names
        "a": (_M, _K), "b": (_N, _K), "c": (_M, _N),
        "x": (_M, _K), "w": (_N, _K), "y": (_M, _N),
        "input": (_M, _K), "weight": (_N, _K), "output": (_M, _N),
        "bias": (_N,),
        # FP8 / block-scale variants
        "a_scale":   (_M,), "b_scale":   (_N,),
        "x_scale":   (_M,), "w_scale":   (_N,),
        "input_scale": (_M,), "weight_scale": (_N,),
        "scale_a": (_M,), "scale_b": (_N,),
        "a_scale_block": (_M, _K // 128), "b_scale_block": (_N, _K // 128),
        # int sizes
        "m": _M, "n": _N, "k": _K,
        "stride_am": _K, "stride_bk": _N, "stride_cm": _N,
    },
    dtypes={
        "a": torch.bfloat16, "b": torch.bfloat16,
        "x": torch.bfloat16, "w": torch.bfloat16,
        "input": torch.bfloat16, "weight": torch.bfloat16,
    },
))

# ── INT8 quantized GEMM (a8w8) ────────────────────────────────────────────────
# x:(M,K) int8, w:(N,K) int8, block-scales: (M, K//128) and (N, K//128) fp32
_reg(["gemm_a8w8", "gemm_a8w8_blockscale", "fused_gemm_a8w8", "mxfp8_gemm"], OpHarness(
    build_args=lambda func: _make_a8w8_args(func),
))


def _make_a8w8_args(func) -> list:
    """Build int8/fp8 quantized GEMM args by inspecting parameter names and annotations."""
    import inspect as _inspect
    sig = _inspect.signature(func)
    required_params = [
        p for p in sig.parameters.values()
        if p.default is _inspect.Parameter.empty
        and p.kind not in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD)
    ]
    args = []
    idx = 0
    for p in required_params:
        n = p.name.lower()
        ann = p.annotation
        # If annotation says int, give a scalar
        if ann is int or ann == "<class 'int'>":
            # S1, S2 split sizes: split N in half
            args.append(_N // 2)
            idx += 1
            continue
        # If annotation says float, give scalar
        if ann is float:
            args.append(1.0)
            idx += 1
            continue
        # INT8 input matrices (first positional x and w)
        if n == "x" and idx == 0:
            args.append(torch.randint(-8, 8, (_M, _K), dtype=torch.int8, device="cuda"))
        elif n == "w" and idx <= 2:
            args.append(torch.randint(-8, 8, (_N, _K), dtype=torch.int8, device="cuda"))
        # Block-scale tensors
        elif n in ("x_scales", "a_scales", "input_scales"):
            args.append(torch.ones((_M, _K // 128), dtype=torch.float32, device="cuda"))
        elif n in ("w_scales", "b_scales", "weight_scales"):
            args.append(torch.ones((_N, _K // 128), dtype=torch.float32, device="cuda"))
        # Per-token scales (1D)
        elif n in ("scale_a", "x_scale", "input_scale"):
            args.append(torch.ones((_M,), dtype=torch.float32, device="cuda"))
        elif n in ("scale_b", "w_scale", "weight_scale"):
            args.append(torch.ones((_N,), dtype=torch.float32, device="cuda"))
        # Output / fused-result tensors (M x N bf16)
        elif n in ("y", "c", "output", "out"):
            args.append(torch.zeros((_M, _N), dtype=torch.bfloat16, device="cuda"))
        # Fused mul/add tensors
        elif n in ("a",) and idx > 1:
            args.append(torch.randn((_M, _N), dtype=torch.bfloat16, device="cuda"))
        elif n in ("b",) and idx > 1:
            args.append(torch.randn((_M, _N), dtype=torch.bfloat16, device="cuda"))
        elif n == "bias":
            args.append(torch.zeros((_N,), dtype=torch.bfloat16, device="cuda"))
        elif n in ("m",):
            args.append(_M)
        elif n in ("n",):
            args.append(_N)
        elif n in ("k",):
            args.append(_K)
        elif n in ("s1", "s2"):
            args.append(_N // 2)
        elif n == "dtype":
            args.append(torch.bfloat16)
        else:
            # generic fallback: bf16 tensor
            args.append(torch.randn((128, 256), dtype=torch.bfloat16, device="cuda"))
        idx += 1
    return args

# ── Attention family ─────────────────────────────────────────────────────────
# Q:(B,H,S,D), K:(B,HK,S,D), V:(B,HK,S,D) → O:(B,H,S,D)
_reg(["attention", "fmha", "mha", "mla", "paged_attention"], OpHarness(
    shapes={
        "q": (_B, _H, _S, _D), "k": (_B, _HK, _S, _D), "v": (_B, _HK, _S, _D),
        "o": (_B, _H, _S, _D), "out": (_B, _H, _S, _D), "output": (_B, _H, _S, _D),
        # flat/varlen variants
        "query": (_B * _S, _H, _D), "key": (_B * _S, _HK, _D),
        "value": (_B * _S, _HK, _D),
        # scales
        "q_scale": (_B, _H), "k_scale": (_B, _HK), "v_scale": (_B, _HK),
        "softmax_scale": 1.0 / (_D ** 0.5),
        "sm_scale":      1.0 / (_D ** 0.5),
        # seqlen info
        "seq_len": _S, "seqlen": _S, "seqlen_q": _S, "seqlen_k": _S,
        "batch_size": _B, "num_heads": _H, "num_kv_heads": _HK, "head_dim": _D,
        "is_causal": False,
        # cu_seqlens for varlen
        "cu_seqlens_q": None, "cu_seqlens_k": None,
        # lse / softmax_lse
        "lse": (_B, _H, _S), "softmax_lse": (_B, _H, _S),
    },
    dtypes={
        "q": torch.bfloat16, "k": torch.bfloat16, "v": torch.bfloat16,
        "query": torch.bfloat16, "key": torch.bfloat16, "value": torch.bfloat16,
        "lse": torch.float32, "softmax_lse": torch.float32,
    },
))

# ── KV-cache ─────────────────────────────────────────────────────────────────
_reg(["kv_cache"], OpHarness(
    shapes={
        "k":         (_B, _HK, _S, _D), "v": (_B, _HK, _S, _D),
        "k_cache":   (_B, _HK, _S, _D), "v_cache": (_B, _HK, _S, _D),
        "key":       (_B, _HK, 1, _D),  "value": (_B, _HK, 1, _D),
        "slots":     (_B,),
        "block_tables": (_B, _S // 16),
        "seq_lens":  (_B,),
        "num_heads": _HK, "head_dim": _D, "block_size": 16,
    },
    dtypes={
        "k": torch.bfloat16, "v": torch.bfloat16,
        "k_cache": torch.bfloat16, "v_cache": torch.bfloat16,
        "key": torch.bfloat16, "value": torch.bfloat16,
        "slots": torch.int32, "seq_lens": torch.int32,
        "block_tables": torch.int32,
    },
))

# ── Normalization family ─────────────────────────────────────────────────────
_reg(["rmsnorm", "layernorm", "normalization"], OpHarness(
    shapes={
        "x":          (_M, _N), "input": (_M, _N),
        "hidden_states": (_M, _N),
        "weight":     (_N,), "gamma": (_N,), "scale": (_N,),
        "bias":       (_N,), "beta":  (_N,),
        "residual":   (_M, _N),
        "output":     (_M, _N), "out": (_M, _N),
        # output scale for fused quant
        "x_scale":    (_M,), "output_scale": (_M,),
        "eps": 1e-5, "epsilon": 1e-5,
        "n_cols": _N, "hidden_size": _N,
    },
    dtypes={
        "x": torch.bfloat16, "input": torch.bfloat16,
        "hidden_states": torch.bfloat16, "residual": torch.bfloat16,
        "weight": torch.bfloat16, "gamma": torch.bfloat16,
        "bias": torch.bfloat16, "beta": torch.bfloat16,
        "output": torch.bfloat16, "out": torch.bfloat16,
        "x_scale": torch.float32, "output_scale": torch.float32,
    },
))

# ── MoE / MoE-sorting ────────────────────────────────────────────────────────
_reg(["moe", "moe_sorting", "topk", "moe_routing", "moe_grouped_gemm",
      "moe_reduce", "dispatch_combine"], OpHarness(
    shapes={
        # token routing
        "hidden_states": (_M, _K), "input": (_M, _K),
        "x":             (_M, _K),
        # weights per expert
        "w1": (_EXPERTS, _N, _K), "w2": (_EXPERTS, _K, _N),
        "w3": (_EXPERTS, _N, _K),
        "gate_up_weight": (_EXPERTS, _N * 2, _K),
        "down_weight": (_EXPERTS, _K, _N),
        # routing
        "topk_weights":  (_M, 2), "topk_ids": (_M, 2),
        "expert_ids":    (_M * 2,), "token_ids": (_M * 2,),
        "sorted_token_ids": (_M * 2,), "expert_weights": (_M, 2),
        # expert output (M*top_k slots, hidden_dim)
        "expert_output": (_M * 2, _K), "dispatched": (_M * 2, _K),
        # sizes
        "num_tokens": _M, "num_experts": _EXPERTS, "top_k": 2,
        "hidden_size": _K, "intermediate_size": _N,
    },
    dtypes={
        "hidden_states": torch.bfloat16, "input": torch.bfloat16,
        "x": torch.bfloat16,
        "w1": torch.bfloat16, "w2": torch.bfloat16, "w3": torch.bfloat16,
        "gate_up_weight": torch.bfloat16, "down_weight": torch.bfloat16,
        "topk_weights": torch.float32,
        "topk_ids": torch.int32, "expert_ids": torch.int32,
        "token_ids": torch.int32, "sorted_token_ids": torch.int32,
        "expert_weights": torch.float32,
    },
))

# ── Conv ─────────────────────────────────────────────────────────────────────
_reg(["conv"], OpHarness(
    shapes={
        # (N, C, L) causal conv1d
        "x":       (_B, 256, _S), "input": (_B, 256, _S),
        "weight":  (256, 1, 4),   "conv_weight": (256, 1, 4),
        "bias":    (256,),
        "state":   (_B, 256, 4),  "conv_state": (_B, 256, 4),
        # 2-D conv
        "a": (_B, 256, _S), "b": (256, 1, 4),
        "d_model": 256, "d_conv": 4,
    },
    dtypes={
        "x": torch.bfloat16, "input": torch.bfloat16,
        "weight": torch.bfloat16, "conv_weight": torch.bfloat16,
        "state": torch.bfloat16, "conv_state": torch.bfloat16,
    },
))

# ── Quantization ─────────────────────────────────────────────────────────────
_reg(["quantization"], OpHarness(
    shapes={
        "x":      (_M, _N), "input": (_M, _N),
        "output": (_M, _N), "out": (_M, _N),
        "scale":  (_M,), "x_scale": (_M,), "scales": (_M,),
        "amax":   (_M,), "n_cols": _N, "num_cols": _N,
    },
    dtypes={
        "x": torch.bfloat16, "input": torch.bfloat16,
        "output": torch.float8_e4m3fnuz,
        "out": torch.float8_e4m3fnuz,
        "scale": torch.float32, "x_scale": torch.float32,
        "scales": torch.float32, "amax": torch.float32,
    },
))

# ── All-reduce ────────────────────────────────────────────────────────────────
_reg(["all_reduce"], OpHarness(
    shapes={
        "x":      (_M, _N), "input": (_M, _N),
        "output": (_M, _N), "out": (_M, _N),
        "buffer": (_M * _N * 4,),
        "rank_data": (8 * 8192 * 1024,),
        "world_size": 8, "rank": 0,
    },
    dtypes={"x": torch.bfloat16, "input": torch.bfloat16,
            "output": torch.bfloat16, "out": torch.bfloat16,
            "buffer": torch.uint8, "rank_data": torch.uint8},
))

# ── Sampling ──────────────────────────────────────────────────────────────────
_reg(["sampling"], OpHarness(
    shapes={
        "logits":      (_B, 32000), "probs": (_B, 32000),
        "temperatures": (_B,), "top_p": (_B,), "top_k_val": (_B,),
        "output":      (_B,), "seeds": (_B,),
        "vocab_size": 32000, "batch_size": _B,
    },
    dtypes={
        "logits": torch.float32, "probs": torch.float32,
        "temperatures": torch.float32, "top_p": torch.float32,
        "output": torch.int32, "seeds": torch.int64,
    },
))

# ── Delta rule ────────────────────────────────────────────────────────────────
_reg(["delta_rule"], OpHarness(
    shapes={
        "q": (_B, _H, _S, _D), "k": (_B, _H, _S, _D),
        "v": (_B, _H, _S, _D), "beta": (_B, _H, _S),
        "output": (_B, _H, _S, _D),
        "h0": (_B, _H, _D, _D),
        "seq_len": _S, "num_heads": _H, "head_dim": _D,
    },
    dtypes={
        "q": torch.bfloat16, "k": torch.bfloat16,
        "v": torch.bfloat16, "output": torch.bfloat16,
        "h0": torch.bfloat16, "beta": torch.bfloat16,
    },
))

# ── Generic fallback ──────────────────────────────────────────────────────────
_FALLBACK = OpHarness(
    shapes={
        "a": (_M, _K), "b": (_N, _K), "c": (_M, _N),
        "x": (_M, _K), "y": (_M, _N),
        "input": (_M, _K), "output": (_M, _N),
        "weight": (_N, _K), "bias": (_N,),
        "q": (_B, _H, _S, _D), "k": (_B, _HK, _S, _D), "v": (_B, _HK, _S, _D),
    },
)
