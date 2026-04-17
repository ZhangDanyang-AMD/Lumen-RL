"""MoE diagnostics: load balancing, entropy, and lightweight router utilities."""

from __future__ import annotations

import logging
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def _extract_router_logits(output: Any) -> Tensor | None:
    """Best-effort extraction of router logits from a MoE layer output."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple):
        for part in reversed(output):
            if isinstance(part, Tensor) and part.ndim >= 2:
                return part
    return None


def iter_moe_modules(model: nn.Module) -> Iterator[tuple[int, str, nn.Module]]:
    """Yield ``(layer_index, qualified_name, module)`` for likely MoE blocks."""
    idx = 0
    for name, module in model.named_modules():
        cls = type(module).__name__
        if "MoE" in cls or "MoeBlock" in cls or "SparseMoe" in cls:
            yield idx, name, module
            idx += 1
            continue
        if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "w_gate")):
            yield idx, name, module
            idx += 1


def compute_load_balance_loss(router_logits: Tensor, num_experts: int, top_k: int) -> Tensor:
    """Switch-style load balancing loss on router assignments.

    Args:
        router_logits: Tensor shaped ``[..., num_experts]``.
        num_experts: Number of routed experts.
        top_k: Top-k routing width (1 reduces to hard argmax counts).
    """
    logits = router_logits.reshape(-1, num_experts)
    probs = F.softmax(logits, dim=-1)
    k = max(1, min(int(top_k), num_experts))
    top = torch.topk(probs, k, dim=-1)
    counts = torch.zeros_like(probs)
    counts.scatter_(1, top.indices, top.values)
    f = counts.sum(dim=0) / float(counts.shape[0])
    loss = float(num_experts) * (f * f).sum()
    return loss


def compute_router_entropy(router_logits: Tensor) -> Tensor:
    """Mean Shannon entropy of the router softmax over tokens."""
    logits = router_logits.reshape(-1, router_logits.shape[-1])
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)
    return ent.mean()


def check_expert_utilization(router_logits: Tensor, num_experts: int) -> dict[str, Any]:
    """Summarize how uniformly experts are used under a softmax router."""
    logits = router_logits.reshape(-1, num_experts)
    p = F.softmax(logits, dim=-1)
    mean_p = p.mean(dim=0)
    std_p = p.std(dim=0)
    max_expert = int(mean_p.argmax().item())
    min_expert = int(mean_p.argmin().item())
    util = {
        "num_tokens": int(logits.shape[0]),
        "num_experts": int(num_experts),
        "mean_softmax_mass_per_expert": mean_p.detach().cpu().tolist(),
        "std_softmax_mass_per_expert": std_p.detach().cpu().tolist(),
        "argmax_expert_mass_mean": float(mean_p[max_expert].item()),
        "argmin_expert_mass_mean": float(mean_p[min_expert].item()),
    }
    logger.debug("Expert utilization summary: max_mean=%.4f min_mean=%.4f", util["argmax_expert_mass_mean"], util["argmin_expert_mass_mean"])
    return util
