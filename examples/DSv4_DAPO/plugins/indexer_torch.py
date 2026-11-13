"""DeepSeek-V4 DSA indexer scoring in pure PyTorch.

Replaces miles' tilelang ``batched_indexer_fwd``. Semantics from
``miles_plugins/models/deepseek_v4/ops/kernel/tilelang_indexer_fwd.py``:

    logits[s, kv] = sum_h relu(q[s, h] . k[kv]) * weights[s, h]

masked to -inf outside each query's causal range ``[cu_k_s[s], cu_k_e[s])``. The
kernel leaves the out-of-range entries uninitialized and a second pass
(``clean_logits``) writes the -inf, so the masking is part of the contract, not
an optimization.

Only the forward exists in miles too: the indexer picks top-k indices, and
``topk_indices`` is an integer selection that carries no gradient.
"""

import torch
from torch import Tensor


def make_causal_cu_seqlens(
    seq_len_q: int, seq_len_kv: int, compress_ratio: int, device
) -> tuple[Tensor, Tensor]:
    """Copied from miles: query p may see compressed groups [0, (p+1)//ratio)."""
    positions = torch.arange(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ks = torch.zeros(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ke = ((positions + 1) // compress_ratio).to(torch.int32)
    return cu_seqlen_ks, cu_seqlen_ke


def indexer_fwd(
    q: Tensor, kv: Tensor, weights: Tensor, cu_seqlen_ks: Tensor, cu_seqlen_ke: Tensor
) -> Tensor:
    """q [S,H,D] bf16 / kv [Kv,D] bf16 / weights [S,H] fp32 -> logits [S,Kv] fp32."""
    scores = torch.einsum('shd,kd->shk', q.float(), kv.float())
    logits = (scores.clamp_min(0) * weights.float().unsqueeze(-1)).sum(dim=1)

    kv_pos = torch.arange(kv.shape[0], device=q.device, dtype=torch.int32)
    in_range = (kv_pos.unsqueeze(0) >= cu_seqlen_ks.unsqueeze(1)) & (
        kv_pos.unsqueeze(0) < cu_seqlen_ke.unsqueeze(1)
    )
    return logits.masked_fill(~in_range, float('-inf'))


def batched_indexer_fwd(
    q: Tensor, k: Tensor, weights: Tensor, cu_seqlen_ks: Tensor, cu_seqlen_ke: Tensor
) -> Tensor:
    """q [S,B,H,D] / k [Kv,B,D] / weights [S,B,H] -> logits [B,S,Kv] fp32."""
    batch = q.shape[1]
    return torch.stack(
        [
            indexer_fwd(q[:, b], k[:, b], weights[:, b], cu_seqlen_ks, cu_seqlen_ke)
            for b in range(batch)
        ],
        dim=0,
    )


# miles' kernel module exports this name with a leading underscore.
_make_causal_cu_seqlens = make_causal_cu_seqlens
