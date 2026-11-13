"""DeepSeek-V4 sparse MQA attention in pure PyTorch, autograd only.

Replaces miles' ``sparse_attn_tilelang`` (the one hard tilelang dependency on the
training path). Semantics were read off the tilelang kernel in
``miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_fwd.py``:

  for each (b, s, h):
      idxs   = topk_idxs[b, s]                     # int32, -1 marks a padding slot
      scores = q[b,s,h] . kv[b, idxs] * sm_scale   # invalid slots -> -inf
      probs  = softmax([scores, attn_sink[h]])     # gpt-oss sink: one extra column
      out    = sum_j probs[j] * kv[b, idxs[j]]     # sink column dropped

K and V are the same tensor (shared-KV MQA), and ``attn_sink`` enters as a raw
logit -- it is *not* multiplied by ``sm_scale`` -- matching both the kernel
(which only rebases it to log2) and the transformers reference.

Implemented by gathering rather than by masking a full [S, S_kv] score matrix:

  * a mask cannot represent a repeated index, but a gather counts it twice,
    which is what the kernel does;
  * cost is O(S * topk) instead of O(S * S_kv). At topk=512 and S=4096 that is
    8x less, and the gap grows with sequence length.

If every slot of a row is invalid the sink still carries the whole softmax mass,
so the output is 0 rather than NaN -- again matching the kernel.
"""

import torch
from torch import Tensor


def sparse_mqa_attention(
    q: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    sm_scale: float | None = None,
    q_chunk: int | None = None,
) -> Tensor:
    """q [B,S,H,D] / kv [B,S_kv,D] / attn_sink [H] fp32 / topk_idxs [B,S,T] int -> [B,S,H,D].

    ``sm_scale`` is the natural-log scale (the caller passes ``head_dim**-0.5``);
    the kernel's extra log2(e) factor is an artifact of its use of ``exp2``.
    ``q_chunk`` splits the query axis to bound the size of the gathered KV.
    """
    b, s, h, d = q.shape
    if sm_scale is None:
        sm_scale = d**-0.5

    if q_chunk is None or q_chunk >= s:
        return _sparse_mqa_chunk(q, kv, attn_sink, topk_idxs, sm_scale)

    outs = [
        _sparse_mqa_chunk(
            q[:, i : i + q_chunk], kv, attn_sink, topk_idxs[:, i : i + q_chunk], sm_scale
        )
        for i in range(0, s, q_chunk)
    ]
    return torch.cat(outs, dim=1)


def _sparse_mqa_chunk(
    q: Tensor, kv: Tensor, attn_sink: Tensor, topk_idxs: Tensor, sm_scale: float
) -> Tensor:
    b, s, h, _ = q.shape

    valid = topk_idxs >= 0
    gather_idx = topk_idxs.clamp_min(0).long()
    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1)

    # Gather from an fp32 view rather than from bf16. Backward through the gather
    # scatter-adds up to `topk` contributions onto each KV position, and doing
    # that accumulation in bf16 costs an order of magnitude of accuracy in the KV
    # gradient. Casting once up front also avoids the two fp32 temporaries the
    # einsums would otherwise each materialize.
    kv_g = kv.float()[batch_idx, gather_idx]  # [B, S, T, D] fp32

    scores = torch.einsum('bshd,bstd->bsht', q.float(), kv_g) * sm_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float('-inf'))

    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    combined = torch.cat([scores, sink], dim=-1)
    # Subtracting the row max keeps this finite in bf16; transformers' reference
    # does the same, and it also makes the all-invalid row resolve to 0.
    combined = combined - combined.amax(dim=-1, keepdim=True).detach()
    probs = torch.softmax(combined, dim=-1)[..., :-1]

    return torch.einsum('bsht,bstd->bshd', probs, kv_g).to(q.dtype)


def sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Drop-in for miles' entry point of the same name."""
    return sparse_mqa_attention(q, kv, attn_sink, topk_idxs, sm_scale)
