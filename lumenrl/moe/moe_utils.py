"""MoE diagnostics: load balancing, entropy, and lightweight router utilities."""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")


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
    """Yield ``(layer_index, qualified_name, module)`` for likely MoE blocks.

    Handles HF-style blocks (``experts`` + ``gate``/``w_gate``) and Megatron-Core
    ``MoELayer`` (``experts`` + ``router``). The layer index is parsed from the
    module path (``decoder.layers.{i}.mlp``) when available, else a running count.
    """
    running = 0
    for name, module in model.named_modules():
        cls = type(module).__name__
        is_moe = (
            "MoE" in cls or "MoeBlock" in cls or "SparseMoe" in cls
            or (hasattr(module, "experts") and hasattr(module, "router"))       # Megatron MoELayer
            or (hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "w_gate")))
        )
        if not is_moe:
            continue
        m = _LAYER_IDX_RE.search(name)
        layer_idx = int(m.group(1)) if m else running
        yield layer_idx, name, module
        running += 1


def iter_megatron_routers(model: nn.Module) -> Iterator[tuple[int, nn.Module]]:
    """Yield ``(layer_index, router_module)`` for Megatron-Core MoE routers.

    A router is detected by the presence of ``routing`` + ``gating`` callables
    (``megatron.core.transformer.moe.router.TopKRouter``). The layer index is
    parsed from the module path (local per-PP-stage numbering)."""
    for name, module in model.named_modules():
        if callable(getattr(module, "routing", None)) and callable(getattr(module, "gating", None)):
            m = _LAYER_IDX_RE.search(name)
            if m:
                yield int(m.group(1)), module


@contextmanager
def megatron_record_router_logits(model: nn.Module, store: dict[int, Tensor]):
    """Record each Megatron MoE layer's router logits into ``store`` (by layer idx).

    Wraps every router's ``routing(logits, ...)`` so the pre-routing gating logits
    are captured (detached) as they flow through the forward. Works under EP/CP/TP
    (each rank records its LOCAL token logits). Restores the methods on exit.
    """
    saved: list[tuple[nn.Module, Any]] = []

    def make_wrap(layer_idx: int, orig):
        def wrapped(logits, *args, **kwargs):
            store[layer_idx] = logits.detach()
            return orig(logits, *args, **kwargs)
        return wrapped

    try:
        for layer_idx, router in iter_megatron_routers(model):
            orig = router.routing
            saved.append((router, orig))
            router.routing = make_wrap(layer_idx, orig)
        if not saved:
            logger.warning("megatron_record_router_logits: no Megatron routers found.")
        yield store
    finally:
        for router, orig in saved:
            router.routing = orig


@contextmanager
def megatron_replay_router_logits(model: nn.Module, recorded: dict[int, Tensor]):
    """Replay recorded router logits into each Megatron MoE layer's ``routing``.

    Substitutes the passed ``logits`` with ``recorded[layer_idx]`` (shape/dtype/
    device matched) so training reuses the recorded routing decisions. Restores
    the methods on exit. Requires the current forward's per-rank token layout to
    match the recording forward (same microbatch construction)."""
    saved: list[tuple[nn.Module, Any]] = []

    def make_wrap(layer_idx: int, orig):
        def wrapped(logits, *args, **kwargs):
            rec = recorded.get(layer_idx)
            if rec is not None and rec.numel() == logits.numel():
                logits = rec.to(device=logits.device, dtype=logits.dtype).reshape(logits.shape)
            elif rec is not None:
                logger.warning(
                    "megatron_replay_router_logits: layer %d shape mismatch rec=%s cur=%s; using current.",
                    layer_idx, tuple(rec.shape), tuple(logits.shape),
                )
            return orig(logits, *args, **kwargs)
        return wrapped

    try:
        for layer_idx, router in iter_megatron_routers(model):
            orig = router.routing
            saved.append((router, orig))
            router.routing = make_wrap(layer_idx, orig)
        yield
    finally:
        for router, orig in saved:
            router.routing = orig


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
