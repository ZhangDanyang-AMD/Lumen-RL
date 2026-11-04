"""Vocab-parallel selected log-probs for TP-sharded logits (score centering, P7).

With Megatron ``GPTModel(parallel_output=True)`` each tensor-parallel rank
holds a contiguous ``[R, V/tp]`` slice of the logits. Scoring ``[y, H]``
(sampled token + sampler head ids) then needs the full-vocab log-sum-exp and
the logit of each id from whichever rank owns it:

forward (per row chunk), three all-reduces over the TP group:
    max_r      = max over ranks of the local max         (MAX)
    sumexp_r   = sum over ranks of sum exp(z - max_r)    (SUM)
    picked_r,m = sum over ranks of the owner's logit     (SUM; non-owners add 0)
    out        = picked - (max + log sumexp)

backward: no communication. The gradient w.r.t. this rank's logit shard is
``(onehot_local(ids) . g - softmax_local * sum_m g) / T``; Megatron's
column-parallel output layer consumes exactly that local slice.

This is the slime / Miles / torchtitan pattern. Today every Lumen-RL Megatron
engine builds ``parallel_output=False`` (full-vocab logits on each TP rank, see
``packing.selected_log_probs``), so this op is not yet wired; it is the piece
needed to switch to ``parallel_output=True`` at TP > 1 (design doc, P7).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = ["vocab_parallel_selected_log_probs"]


class _VocabParallelSelectedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, ids, vocab_start, temperature, chunk_rows, group):
        rows, v_local = logits.shape
        cdt = torch.promote_types(logits.dtype, torch.float32)
        local_ids = ids - vocab_start
        owned = (local_ids >= 0) & (local_ids < v_local)
        safe_ids = local_ids.clamp(0, v_local - 1)
        out = torch.empty(ids.shape, dtype=cdt, device=logits.device)
        lse = torch.empty(rows, dtype=cdt, device=logits.device)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            z = logits[start:end].to(cdt)
            if temperature != 1.0:
                z = z / temperature
            m = z.max(dim=-1).values
            dist.all_reduce(m, op=dist.ReduceOp.MAX, group=group)
            se = torch.exp(z - m[:, None]).sum(-1)
            dist.all_reduce(se, op=dist.ReduceOp.SUM, group=group)
            lse[start:end] = m + se.log()
            picked = z.gather(-1, safe_ids[start:end]) * owned[start:end]
            dist.all_reduce(picked, op=dist.ReduceOp.SUM, group=group)
            out[start:end] = picked - lse[start:end, None]
        ctx.save_for_backward(logits, safe_ids, owned, lse)
        ctx.temperature = temperature
        ctx.chunk_rows = chunk_rows
        return out

    @staticmethod
    def backward(ctx, g_out):
        logits, safe_ids, owned, lse = ctx.saved_tensors
        t, chunk_rows = ctx.temperature, ctx.chunk_rows
        grad = torch.empty_like(logits)
        rows = logits.shape[0]
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            z = logits[start:end].to(lse.dtype)
            if t != 1.0:
                z = z / t
            g = g_out[start:end].to(lse.dtype)
            owns_z = t != 1.0 or logits.dtype != lse.dtype
            gz = z.sub_(lse[start:end, None]) if owns_z else z - lse[start:end, None]
            gz.exp_().mul_(-g.sum(-1, keepdim=True))
            gz.scatter_add_(-1, safe_ids[start:end], g * owned[start:end])
            if t != 1.0:
                gz.div_(t)
            grad[start:end] = gz.to(grad.dtype)
        return grad, None, None, None, None, None


def vocab_parallel_selected_log_probs(
    logits: torch.Tensor,
    ids: torch.Tensor,
    *,
    vocab_start: int,
    temperature: float = 1.0,
    chunk_rows: int = 4096,
    group=None,
) -> torch.Tensor:
    """Log-probs ``[R, m]`` at global ``ids`` under the full-vocab softmax of
    TP-sharded ``logits [R, V/tp]`` (this rank owns ``[vocab_start, vocab_start
    + V/tp)``). Identical on every rank of ``group``; gradient is local."""
    return _VocabParallelSelectedLogProbs.apply(
        logits, ids.long(), int(vocab_start), float(temperature), int(chunk_rows), group,
    )
