"""DeepSeek-V4.1 text backbone as a trainable full-sequence PyTorch module.

This mirrors the released `inference/model.py` prefill path (start_pos == 0) operation for
operation and dtype for dtype, so that its logits can be checked against the release, but:

* no KV caches: a training forward sees the whole sequence;
* the tensors V4.1 shares across layers (compressed KV, index keys, top-k, candidate blocks)
  travel in an explicit `SharedAttention` instead of a process-global, so a caller can carry
  them across recomputation or pipeline boundaries;
* nothing is written in place into storage another layer reads, so autograd works;
* the inference-time activation quantization is applied as fake quantization with a
  straight-through gradient, and each point can be switched off (`QuantPoints`).

It is the reference the Megatron implementation is checked against, and small enough to train
directly on a truncated slice. MoE here is a per-expert loop; the Megatron path replaces it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from lumenrl.models.deepseek_v41 import ops
from lumenrl.models.deepseek_v41.config import AttnMode, DSV41Config, LayerSpec
from lumenrl.models.deepseek_v41.engram import Engram, EngramHasher, EngramLayout

FP8_BLOCK = 32
FP4_BLOCK = 32
KV_FP4_BLOCK = 16
INDEXER_QUERY_CHUNK = 1024


@dataclass(frozen=True)
class QuantPoints:
    """Where the rollout quantizes activations, reproduced here as fake quantization. Defaults
    are the release's: SWA KV in FP8, compressed KV in FP4 (E4M3 scales), indexer q/k in MXFP4,
    the attention output in FP8 before `wo_a`, and the input of every linear layer the release
    stores in FP8 or FP4 in FP8 (`linear_act_fp8`: attention `wq_a` / `wq_b` / `wkv` / `wo_b`,
    the indexer's `wq_b`, routed and shared experts, Engram's `wkv`)."""

    window_kv_fp8: bool = True
    compressed_kv_fp4: bool = True
    indexer_fp4: bool = True
    attn_out_fp8: bool = True
    linear_act_fp8: bool = True

    @classmethod
    def none(cls) -> QuantPoints:
        return cls(False, False, False, False, False)


@dataclass
class SharedAttention:
    """What sources hand to the layers after them within one forward."""

    compress_kv: Tensor | None = None
    index_k: Tensor | None = None
    topk_idxs: Tensor | None = None
    candidates: Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(8)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rotate(x: Tensor, freqs_cis: Tensor, rope_dim: int, inverse: bool = False) -> Tensor:
    """RoPE on the last `rope_dim` channels of x ([b, s, d] or [b, s, h, d]), out of place."""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    c = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)))
    f = freqs_cis.conj() if inverse else freqs_cis
    f = f.view(1, c.size(1), *([1] * (c.ndim - 3)), c.size(-1))
    tail = torch.view_as_real(c * f).flatten(-2).to(x.dtype)
    return torch.cat([head, tail], dim=-1)


def window_topk_idxs(window: int, bsz: int, seqlen: int, device) -> Tensor:
    end = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (end - window + 1).clamp(0) + torch.arange(min(seqlen, window), device=device)
    idxs = torch.where(idxs > end, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def select_candidate_blocks(logits: Tensor, compress_lens: Tensor, topk_blocks: int, block_size: int) -> Tensor:
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _fake(enabled: bool, x: Tensor, fn) -> Tensor:
    return ops.straight_through(x, fn(x.detach())) if enabled else x


class Compressor(nn.Module):
    """Pools `ratio` consecutive tokens into one KV latent (pre-RoPE). Ratio 1 is a projection."""

    def __init__(self, cfg: DSV41Config, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        # ratio 1 stays in the checkpoint's bf16; the fp32 pooling above ratio 1 promotes to fp32
        wdtype = torch.float32 if ratio > 1 else torch.bfloat16
        self.wkv = nn.Linear(cfg.dim, cfg.head_dim, bias=False, dtype=wdtype)
        if ratio > 1:
            self.wgate = nn.Linear(cfg.dim, cfg.head_dim, bias=False, dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor | None:
        seqlen, ratio, dtype = x.size(1), self.ratio, x.dtype
        if ratio == 1:
            return self.norm(self.wkv(x))
        if seqlen < ratio:
            return None
        xf = x.float()
        cutoff = seqlen - seqlen % ratio
        kv, score = self.wkv(xf)[:, :cutoff], self.wgate(xf)[:, :cutoff]
        kv = (kv.unflatten(1, (-1, ratio)) * score.unflatten(1, (-1, ratio)).softmax(dim=2)).sum(dim=2)
        return self.norm(kv.to(dtype))


class Indexer(nn.Module):
    """Picks the `index_topk` best compressed positions per query, optionally within the
    candidate blocks of an earlier layer."""

    def __init__(self, cfg: DSV41Config, spec: LayerSpec, quant: QuantPoints):
        super().__init__()
        self.owns_k = spec.mode is AttnMode.FULL
        self.ratio = spec.compress_ratio
        self.is_candidate_source = spec.is_candidate_source
        self.uses_candidates = spec.candidate_source is not None
        self.candidate_topk_blocks, self.candidate_block_size = cfg.candidate_topk_blocks, cfg.candidate_block_size
        self.n_heads, self.head_dim, self.rope_dim = cfg.index_n_heads, cfg.index_head_dim, cfg.rope_head_dim
        self.index_topk, self.softmax_scale = cfg.index_topk, cfg.index_head_dim**-0.5
        self.quant = quant
        self.wq_b = ops.QLinear(cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim, act_fp8=quant.linear_act_fp8)
        self.weights_proj = nn.Linear(cfg.dim, cfg.index_n_heads, bias=False, dtype=torch.bfloat16)
        if self.owns_k:
            self.wk = nn.Linear(cfg.head_dim, cfg.index_head_dim, bias=False, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(cfg.index_head_dim, cfg.norm_eps)

    def _q4(self, x: Tensor) -> Tensor:
        return _fake(self.quant.indexer_fp4, x, lambda t: ops.fp4_fake_quant(t, FP4_BLOCK, "e8m0"))

    @torch.no_grad()
    def forward(self, x: Tensor, qr: Tensor, latent: Tensor | None, freqs: Tensor, offset: int, shared: SharedAttention) -> Tensor:
        """No gradient reaches the indexer -- what it returns is a discrete top-k -- so it
        runs without autograd, and over chunks of `INDEXER_QUERY_CHUNK` queries: at a ratio-1
        layer the score of a whole sequence is [s, heads, s]."""
        seqlen, ratio, rd = x.size(1), self.ratio, self.rope_dim
        if self.owns_k and latent is not None:
            k = self.k_norm(self.wk(latent))
            k = self._q4(rotate(k, freqs[: seqlen - seqlen % ratio : ratio], rd))
            shared.index_k = k
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = self._q4(rotate(q, freqs[:seqlen], rd))
        index_k = shared.index_k[:, : seqlen // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        positions = torch.arange(seqlen // ratio, device=x.device)
        topk = min(self.index_topk, seqlen // ratio)
        idxs, candidates = [], []
        for start in range(0, seqlen, INDEXER_QUERY_CHUNK):
            rows = slice(start, min(start + INDEXER_QUERY_CHUNK, seqlen))
            score = torch.einsum("bshd,btd->bsht", q[:, rows], index_k)
            score = (score.relu() * weights[:, rows].unsqueeze(-1)).sum(dim=2)
            compress_lens = (torch.arange(rows.start + 1, rows.stop + 1, device=x.device) // ratio).unsqueeze(-1)
            score = score.masked_fill(positions >= compress_lens, -torch.inf)
            if self.is_candidate_source:
                candidates.append(
                    select_candidate_blocks(score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size)
                )
            elif self.uses_candidates:
                score = score.masked_fill(~shared.candidates[:, rows], -torch.inf)
            top = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
            idxs.append(torch.where(top < compress_lens, top + offset, -1).int())
        if self.is_candidate_source:
            shared.candidates = torch.cat(candidates, dim=1)
        return torch.cat(idxs, dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: DSV41Config, spec: LayerSpec, quant: QuantPoints):
        super().__init__()
        self.spec, self.quant = spec, quant
        self.n_heads, self.head_dim, self.rope_dim = cfg.n_heads, cfg.head_dim, cfg.rope_head_dim
        self.n_groups, self.o_lora_rank, self.window = cfg.o_groups, cfg.o_lora_rank, cfg.window_size
        self.ratio, self.softmax_scale = spec.compress_ratio, cfg.head_dim**-0.5
        self.attn_sink = nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))
        act = quant.linear_act_fp8
        self.wq_a = ops.QLinear(cfg.dim, cfg.q_lora_rank, act_fp8=act)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        self.wq_b = ops.QLinear(cfg.q_lora_rank, cfg.n_heads * cfg.head_dim, act_fp8=act)
        self.wkv = ops.QLinear(cfg.dim, cfg.head_dim, act_fp8=act)
        self.kv_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.wo_a = nn.Linear(
            cfg.n_heads * cfg.head_dim // cfg.o_groups, cfg.o_groups * cfg.o_lora_rank, bias=False, dtype=torch.bfloat16
        )
        self.wo_b = ops.QLinear(cfg.o_groups * cfg.o_lora_rank, cfg.dim, act_fp8=act)
        self.compressor = Compressor(cfg, self.ratio) if spec.mode is AttnMode.FULL else None
        self.indexer = Indexer(cfg, spec, quant) if spec.mode in (AttnMode.FULL, AttnMode.REINDEX) else None
        if self.ratio:
            orig, theta = cfg.original_seq_len, cfg.compress_rope_theta
        else:
            orig, theta = 0, cfg.rope_theta
        self._rope = (cfg.rope_head_dim, orig, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)

    def freqs(self, seqlen: int, device) -> Tensor:
        dim, orig, theta, factor, bf, bs = self._rope
        return precompute_freqs_cis(dim, seqlen, orig, theta, factor, bf, bs).to(device)

    def forward(self, x: Tensor, shared: SharedAttention) -> Tensor:
        bsz, seqlen, _ = x.size()
        rd, freqs = self.rope_dim, self.freqs(seqlen, x.device)
        qr = self.q_norm(self.wq_a(x))
        q = rotate(self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim)), freqs, rd)

        kv = rotate(self.kv_norm(self.wkv(x)), freqs, rd)
        kv = _fake(self.quant.window_kv_fp8, kv, lambda t: ops.fp8_fake_quant(t, FP8_BLOCK, pow2_scale=True))
        topk_idxs = window_topk_idxs(self.window, bsz, seqlen, x.device)
        if self.ratio:
            compress_kv, compress_idxs = self._compress(x, qr, freqs, kv.size(1), shared)
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = ops.sparse_attention(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        o = rotate(o, freqs, rd, inverse=True)
        o = o.view(bsz, seqlen, self.n_groups, -1)
        o = _fake(self.quant.attn_out_fp8, o, lambda t: ops.fp8_fake_quant(t, FP8_BLOCK, pow2_scale=True))
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))

    def _compress(self, x: Tensor, qr: Tensor, freqs: Tensor, offset: int, shared: SharedAttention):
        seqlen, ratio = x.size(1), self.ratio
        compress_len = seqlen // ratio
        latent = self.compressor(x) if self.compressor is not None else None
        if self.indexer is None:
            idxs = shared.topk_idxs
        elif compress_len == 0:
            idxs = torch.empty(x.size(0), seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            idxs = self.indexer(x, qr, latent, freqs, offset, shared)
        if self.indexer is not None:
            shared.topk_idxs = idxs
        if latent is not None:
            latent = rotate(latent, freqs[: seqlen - seqlen % ratio : ratio], self.rope_dim)
            latent = _fake(
                self.quant.compressed_kv_fp4, latent, lambda t: ops.fp4_fake_quant(t, KV_FP4_BLOCK, "e4m3")
            )
            shared.compress_kv = latent
        if self.compressor is not None and latent is None:
            shared.compress_kv = x.new_zeros(x.size(0), 0, self.head_dim)
        return shared.compress_kv[:, :compress_len], idxs


class Gate(nn.Module):
    def __init__(self, cfg: DSV41Config):
        super().__init__()
        self.topk, self.score_func = cfg.n_activated_experts, cfg.score_func
        self.norm_topk_prob, self.route_scale = cfg.norm_topk_prob, cfg.route_scale
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.dim))
        self.bias = nn.Parameter(torch.empty(cfg.n_routed_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float, act_fp8: bool = False):
        super().__init__()
        self.w1 = ops.QLinear(dim, inter_dim, act_fp8=act_fp8)
        self.w2 = ops.QLinear(inter_dim, dim, act_fp8=act_fp8)
        self.w3 = ops.QLinear(dim, inter_dim, act_fp8=act_fp8)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: Tensor, weights: Tensor | None = None) -> Tensor:
        dtype = x.dtype
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights * y
        return self.w2(y.to(dtype))


class MoE(nn.Module):
    """Reference MoE: routed experts in a loop plus the shared expert."""

    def __init__(self, cfg: DSV41Config, quant: QuantPoints | None = None):
        super().__init__()
        self.dim, self.n_routed_experts = cfg.dim, cfg.n_routed_experts
        act = quant is not None and quant.linear_act_fp8
        self.gate = Gate(cfg)
        self.experts = nn.ModuleList(
            Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit, act) for _ in range(cfg.n_routed_experts)
        )
        self.shared_experts = Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit, act)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        for i, expert in enumerate(self.experts):
            idx, top = torch.where(indices == i)
            if idx.numel():
                y = y.index_add(0, idx, expert(x[idx], weights[idx, top, None]).float())
        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """Single-pass mHC block: each sublayer's coefficients mix the input of the next one."""

    def __init__(
        self,
        cfg: DSV41Config,
        spec: LayerSpec,
        quant: QuantPoints,
        layout: EngramLayout | None,
        ffn: Callable[[LayerSpec], nn.Module] | None = None,
        engram_on_host: bool = False,
    ):
        super().__init__()
        self.norm_eps, self.hc_mult = cfg.norm_eps, cfg.hc_mult
        self.hc_sinkhorn_iters, self.hc_eps = cfg.hc_sinkhorn_iters, cfg.hc_eps
        self.attn = Attention(cfg, spec, quant)
        self.ffn = MoE(cfg, quant) if ffn is None else ffn(spec)
        self.engram = (
            Engram(cfg, layout, spec.engram_hash_index, act_fp8=quant.linear_act_fp8, table_on_host=engram_on_host)
            if spec.engram_hash_index is not None else None
        )
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        mix_hc, hc_dim = (2 + cfg.hc_mult) * cfg.hc_mult, cfg.hc_mult * cfg.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_mixes(self, x: Tensor, fn: Tensor, scale: Tensor, base: Tensor):
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, fn) * rsqrt
        return ops.hc_split_sinkhorn(mixes, scale, base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    @staticmethod
    def hc_pre(x: Tensor, pre_mix: Tensor) -> Tensor:
        return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)

    @staticmethod
    def hc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: Tensor, pre_mix: Tensor, shared: SharedAttention) -> tuple[Tensor, Tensor]:
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn(self.attn_norm(self.hc_pre(x, pre_mix)), shared)
        x = self.hc_post(x, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn(self.ffn_norm(self.hc_pre(x, attn_pre)))
        return self.hc_post(x, residual, ffn_post, ffn_comb), ffn_pre


SHARED_FIELDS = ("compress_kv", "index_k", "topk_idxs", "candidates")


def _pre_mix_width(dtype: torch.dtype) -> int:
    size = torch.empty((), dtype=dtype).element_size()
    if 4 % size:
        raise TypeError(f"cannot carry an fp32 pre_mix in {dtype}")
    return 4 // size


class _Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h: Tensor, pre_mix: Tensor) -> Tensor:
        ctx.dim = h.size(-1)
        return torch.cat([h, pre_mix.float().unsqueeze(-1).view(h.dtype)], dim=-1)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return grad[..., : ctx.dim], grad[..., ctx.dim :].contiguous().view(torch.float32).squeeze(-1)


class _Unpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed: Tensor) -> tuple[Tensor, Tensor]:
        width = _pre_mix_width(packed.dtype)
        ctx.shape, ctx.dtype, ctx.width = packed.shape, packed.dtype, width
        h = packed[..., :-width].contiguous()
        return h, packed[..., -width:].contiguous().view(torch.float32).squeeze(-1)

    @staticmethod
    def backward(ctx, grad_h: Tensor | None, grad_pre: Tensor | None):
        device = (grad_h if grad_h is not None else grad_pre).device
        out = torch.zeros(ctx.shape, dtype=ctx.dtype, device=device)
        if grad_h is not None:
            out[..., : -ctx.width] = grad_h
        if grad_pre is not None:
            out[..., -ctx.width :] = grad_pre.float().unsqueeze(-1).view(ctx.dtype)
        return out


def pack_stream(h: Tensor, pre_mix: Tensor) -> Tensor:
    """What crosses a pipeline stage boundary: the residual stream [b, s, hc, d] and the fp32
    `pre_mix` [b, s, hc] the last FFN computed for the next attention (it cannot be recomputed
    from `h`), as one tensor [b, s, hc, d + w] -- each fp32 reinterpreted as `w` elements of
    `h`'s dtype, so a single-tensor send carries both bit for bit. Gradients travel packed the
    same way."""
    return _Pack.apply(h, pre_mix)


def unpack_stream(packed: Tensor) -> tuple[Tensor, Tensor]:
    return _Unpack.apply(packed)


def _run_layer(layer: Block, hashes: Tensor | None, h: Tensor, pre_mix: Tensor, *fields):
    shared = SharedAttention(*fields)
    if layer.engram is not None:
        h = layer.engram(h, hashes[:, :, layer.engram.hash_index, :])
    h, pre_mix = layer(h, pre_mix, shared)
    return (h, pre_mix, *(getattr(shared, f) for f in SHARED_FIELDS))


class DSV41Stage(nn.Module):
    """Layers [start, end) of `cfg`, as one pipeline stage holds them. The first stage embeds and
    expands to hc_mult copies; the last collapses them and projects to the vocabulary; any other
    returns `pack_stream(h, pre_mix)`. A layer whose KV, index or candidate source is in an
    earlier stage is refused, so nothing else ever has to cross.

    Each layer gets the shared attention tensors as explicit inputs and returns them, which is
    what makes `recompute` (one non-reentrant checkpoint per layer) correct: in backward a layer
    is recomputed from what the earlier layers produced. Parameter names keep the model's layer
    numbering, so a stage loads with `convert.load_released` as the whole model does.

    `dtype` is what the release keeps in bf16 (most weights, norms, embedding); the fp32 tensors
    (hc mixes, sinks, gate bias, ratio-2 compressor, head) are fp32 regardless.

    `ffn(spec)` builds each layer's MoE in place of the reference one (whose experts every rank
    would hold); `engram_on_host` keeps the Engram tables out of the module state, in host
    memory (see `EngramTable`)."""

    def __init__(
        self,
        cfg: DSV41Config,
        start: int = 0,
        end: int | None = None,
        token_map: Tensor | None = None,
        quant: QuantPoints | None = None,
        dtype: torch.dtype = torch.bfloat16,
        recompute: bool = False,
        ffn: Callable[[LayerSpec], nn.Module] | None = None,
        engram_on_host: bool = False,
    ):
        super().__init__()
        end = cfg.n_layers if end is None else end
        if not 0 <= start < end <= cfg.n_layers:
            raise ValueError(f"stage [{start}, {end}) is not inside {cfg.n_layers} layers")
        specs = cfg.layer_specs()[start:end]
        for s in specs:
            for what in ("kv_source", "index_source", "candidate_source"):
                src = getattr(s, what)
                if src is not None and src < start:
                    raise ValueError(
                        f"stage [{start}, {end}): layer {s.layer} takes its {what} from layer {src}, "
                        "which is in an earlier stage"
                    )
        self.cfg, self.quant, self.recompute = cfg, quant or QuantPoints(), recompute
        self.start, self.end = start, end
        self.pre_process, self.post_process = start == 0, end == cfg.n_layers
        self.layout = EngramLayout.from_config(cfg)
        self.hasher = None
        if any(s.engram_hash_index is not None for s in specs):
            if token_map is None:
                raise ValueError("an engram model needs the compressed token map")
            self.hasher = EngramHasher(cfg, self.layout, token_map)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            if self.pre_process:
                self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
            self.layers = nn.ModuleDict(
                {str(s.layer): Block(cfg, s, self.quant, self.layout, ffn, engram_on_host) for s in specs}
            )
            if self.post_process:
                self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        finally:
            torch.set_default_dtype(previous)
        if self.post_process:
            self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False, dtype=torch.float32)

    def forward(self, input_ids: Tensor, stream: Tensor | None = None, return_hidden: bool = False) -> Tensor:
        """input_ids [b, s] (every stage: Engram hashes them); `stream` is the previous stage's
        output. Returns fp32 logits [b, s, vocab] from the last stage -- or with `return_hidden`
        the fp32 hidden states [b, s, dim] the head projects, for a caller that applies
        `head.weight` itself a chunk of positions at a time."""
        if self.pre_process:
            h = self.embed(input_ids).unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
            pre_mix = h.new_zeros(*h.shape[:2], self.cfg.hc_mult, dtype=torch.float32)
            pre_mix[..., 0] = 1.0
        elif stream is None:
            raise ValueError(f"stage [{self.start}, {self.end}) needs the previous stage's stream")
        else:
            h, pre_mix = unpack_stream(stream)
        hashes = self.hasher(input_ids) if self.hasher is not None else None
        fields = (None,) * len(SHARED_FIELDS)
        for layer in self.layers.values():
            if self.recompute and torch.is_grad_enabled():
                h, pre_mix, *fields = checkpoint(_run_layer, layer, hashes, h, pre_mix, *fields, use_reentrant=False)
            else:
                h, pre_mix, *fields = _run_layer(layer, hashes, h, pre_mix, *fields)
        if not self.post_process:
            return pack_stream(h, pre_mix)
        h = self.norm(Block.hc_pre(h, pre_mix)).float()
        return h if return_hidden else F.linear(h, self.head.weight)


class DSV41Model(DSV41Stage):
    """The whole model: embed -> hc_mult copies -> blocks (Engram before its layers) -> collapse
    -> norm -> head. input_ids [B, S] -> fp32 logits [B, S, vocab]."""

    def __init__(
        self,
        cfg: DSV41Config,
        token_map: Tensor | None = None,
        quant: QuantPoints | None = None,
        dtype: torch.dtype = torch.bfloat16,
        recompute: bool = False,
        ffn: Callable[[LayerSpec], nn.Module] | None = None,
        engram_on_host: bool = False,
    ):
        super().__init__(cfg, 0, cfg.n_layers, token_map, quant, dtype, recompute, ffn, engram_on_host)
