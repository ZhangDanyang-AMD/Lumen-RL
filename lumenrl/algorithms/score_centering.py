"""Score centering (Marek & Ryabinin, arXiv 2609.20807).

Plain policy gradient under a sampler ``q`` that differs from the trainer ``p``
splits into a covariance term (the signal) and a drift term
``E_q[A] * sum_v q_v grad log p_v`` that distils the trainer toward the biased
sampler. Score centering subtracts the sampler's expected score so only the
covariance remains. Only the sampler's top-k head ``H`` is stored; the tail is
modelled as the trainer's tail rescaled, ``q_v = rho * p_v`` for ``v`` not in
``H``. Because ``E_p[grad log p] = 0`` the tail folds into the head (eq. 14):

    L = -A * ( sg[w_y] log p_y - sum_{v in H} sg[q_v w_v - alpha p_v] log p_v )

    w_v = f(p_v / q_v),  alpha = rho * f(1 / rho),
    rho = (1 - sum_H q) / (1 - sum_H p)   (both tails floored at eps)

``f`` is the importance-sampling rule: 1 (none), ``min(r, c)`` (token TIS) or
``r * 1[lo <= r <= hi]`` (IcePop). Design notes: ``docs/score_centering_design.md``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch
from torch import Tensor

from lumenrl.algorithms.loss_functions import agg_loss

if TYPE_CHECKING:
    from lumenrl.core.config import LumenRLConfig

logger = logging.getLogger(__name__)

__all__ = [
    "make_weight_fn",
    "score_centering_token_loss",
    "score_centering_pg_loss",
    "score_centering_loss_from_config",
    "score_centering_replaces_loss",
    "validate_score_centering",
    "validate_topk_rows",
    "densify_rollout_topk",
    "vllm_engine_kwargs",
]

WeightFn = Callable[[Tensor], Tensor]

# Log-ratio clamp for the sampled-token weight, as in the DAPO loss.
_LOG_RATIO_CLAMP = 20.0


def make_weight_fn(
    is_mode: str,
    *,
    tis_clip: float = 2.0,
    icepop_low: float = 0.5,
    icepop_high: float = 5.0,
) -> WeightFn:
    """Importance-weight rule ``f(r)`` applied to ``r = p / q``."""
    if is_mode == "none":
        return torch.ones_like
    if is_mode == "tis":
        return lambda r: r.clamp(max=tis_clip)
    if is_mode == "icepop":
        return lambda r: r * ((r >= icepop_low) & (r <= icepop_high)).to(r.dtype)
    raise ValueError(f"score_centering is_mode must be none|tis|icepop, got {is_mode!r}")


def score_centering_token_loss(
    logp_y: Tensor,
    head_logp: Tensor,
    q_y: Tensor,
    q_head: Tensor,
    advantages: Tensor,
    *,
    weight_fn: WeightFn,
    eps: float = 1e-6,
    support_complete: bool = False,
    rho_pin_tail: float = 0.0,
    correction_only: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-token REINFORCE + score-centering loss (before aggregation).

    Args:
        logp_y: Trainer log-prob of the sampled token ``[N, L]`` (with grad).
        head_logp: Trainer log-probs at the sampler's head ids ``[N, L, k]``
            (with grad). Values at invalid slots are ignored.
        q_y: Sampler log-prob of the sampled token ``[N, L]``.
        q_head: Sampler head log-probs ``[N, L, k]``; ``-inf`` marks an invalid
            slot (prompt/pad positions, filler under filtered sampling).
        advantages: ``[N, L]`` (or broadcastable).
        weight_fn: ``f(r)`` from :func:`make_weight_fn`.
        eps: Floor on both tail masses.
        rho_pin_tail: ``rho = 1`` where both tail masses are below this (their
            ratio is fp32 rounding noise there); 0 disables.
        correction_only: Return only the additive term ``A * correction`` (for
            ``base_objective: ppo_clip_additive``, added to the clip loss),
            i.e. the REINFORCE loss without its ``-A * w_y * log p_y`` part.
        support_complete: The valid head slots are the sampler's whole support
            ``S`` (filtered sampling whose support fits in the head; ``q_head``
            already renormalized over ``S``). The trainer is renormalized over
            the same ``S`` -- sampled token included -- and ``rho = 1``: the
            correction is exact and out-of-support logits get no gradient.
            Completeness is only *proven* where the head has a ``-inf`` filler
            slot (vLLM ran out of support before filling the head); a full
            head may have cut a support enlarged by ties at the sampler's k-th
            logit. Such positions are still renormalized over the head and
            reported as ``support_unproven``.

    Returns:
        ``(pg [N, L], stats)`` where ``stats`` holds detached per-token tensors
        ``w_y, ratio_y, rho, alpha, correction, p_mass, q_mass``.
    """
    dtype = torch.promote_types(head_logp.dtype, torch.float32)
    valid = torch.isfinite(q_head)
    if support_complete:
        # log p_S(v) = log p(v) - logsumexp_{u in S} log p(u), with grad.
        # Rows with no valid slot (prompt / pad) get lse = 0 via one dummy slot.
        has = valid.any(-1, keepdim=True)
        first = torch.zeros_like(valid)
        first[..., 0] = True
        lse_mask = valid | (~has & first)
        lse = torch.logsumexp(
            head_logp.to(dtype).masked_fill(~lse_mask, float("-inf")), dim=-1
        ) * has.squeeze(-1)
        head_logp = head_logp.to(dtype) - lse.unsqueeze(-1)
        logp_y = logp_y.to(dtype) - lse
    with torch.no_grad():
        lp = head_logp.detach().to(dtype).masked_fill(~valid, float("-inf"))
        lq = q_head.to(dtype)
        p = lp.exp()
        q = lq.exp()
        p_mass = p.sum(-1)
        q_mass = q.sum(-1)
        if support_complete:
            rho = torch.ones_like(q_mass)  # never 0/0 from fp32 noise (Miles #4204)
        else:
            rho = (1.0 - q_mass).clamp_min(eps) / (1.0 - p_mass).clamp_min(eps)
            if rho_pin_tail > 0:
                noise = ((1.0 - q_mass) < rho_pin_tail) & ((1.0 - p_mass) < rho_pin_tail)
                rho = torch.where(noise, torch.ones_like(rho), rho)
        alpha = rho * weight_fn(rho.reciprocal())
        log_ratio = torch.where(valid, lp - lq, torch.zeros_like(lp))
        w_head = weight_fn(log_ratio.clamp(-_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP).exp())
        residual = torch.where(valid, q * w_head - alpha.unsqueeze(-1) * p, torch.zeros_like(p))
        ratio_y = (logp_y.detach().to(dtype) - q_y.to(dtype)).clamp(
            -_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP
        ).exp()
        w_y = weight_fn(ratio_y)

    head_lp = head_logp.to(dtype).masked_fill(~valid, 0.0)
    correction = (residual * head_lp).sum(-1)
    if correction_only:
        pg = advantages.detach().to(dtype) * correction
    else:
        pg = -advantages.detach().to(dtype) * (w_y * logp_y.to(dtype) - correction)
    stats = {
        "support_unproven": (valid.all(-1) if support_complete else None),
        "w_y": w_y,
        "ratio_y": ratio_y,
        "rho": rho,
        "alpha": alpha,
        "correction": correction.detach(),
        "p_mass": p_mass,
        "q_mass": q_mass,
    }
    return pg, stats


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def score_centering_pg_loss(
    logp_y: Tensor,
    head_logp: Tensor,
    q_y: Tensor,
    q_head: Tensor,
    advantages: Tensor,
    *,
    mask: Tensor,
    is_mode: str = "tis",
    tis_clip: float = 2.0,
    icepop_low: float = 0.5,
    icepop_high: float = 5.0,
    eps: float = 1e-6,
    support_complete: bool = False,
    rho_pin_tail: float = 0.0,
    correction_only: bool = False,
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """REINFORCE + score centering, aggregated like the DAPO loss.

    Aggregation goes through :func:`agg_loss` with the same ``loss_agg_mode`` /
    ``batch_num_tokens`` / ``dp_size`` contract as ``asymmetric_clip_loss``, so
    the gradient scale matches today's loss at ratio 1.

    Returns:
        ``(loss, metrics)``; metrics are detached 0-dim tensors (token-masked
        means over ``mask``) under the ``sc/`` prefix.
    """
    weight_fn = make_weight_fn(
        is_mode, tis_clip=tis_clip, icepop_low=icepop_low, icepop_high=icepop_high
    )
    pg, st = score_centering_token_loss(
        logp_y, head_logp, q_y, q_head, advantages, weight_fn=weight_fn, eps=eps,
        support_complete=support_complete,
        rho_pin_tail=rho_pin_tail,
        correction_only=correction_only,
    )
    loss_mask = mask.to(pg.dtype)
    loss = agg_loss(
        pg,
        loss_mask,
        loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
    )

    with torch.no_grad():
        sel = loss_mask.bool()
        q_mass_sel = st["q_mass"][sel]
        metrics = {
            "sc/sampler_head_mass": _masked_mean(st["q_mass"], loss_mask),
            "sc/train_head_mass": _masked_mean(st["p_mass"], loss_mask),
            "sc/tail_ratio_mean": _masked_mean(st["rho"], loss_mask),
            "sc/correction_abs": _masked_mean(st["correction"].abs(), loss_mask),
            "sc/sample_weight_mean": _masked_mean(st["w_y"], loss_mask),
            # Additive coverage signal (the quantile below cannot be merged
            # across rows / workers): share of tokens whose sampler head
            # holds < 90 % of the mass, i.e. where the tail model carries weight.
            "sc/frac_low_head_mass": _masked_mean((st["q_mass"] < 0.9).to(pg.dtype), loss_mask),
            "sc/head_coverage_p01": (
                torch.quantile(q_mass_sel[:16_000_000].float(), 0.01).to(pg.dtype)
                if q_mass_sel.numel() > 0
                else torch.zeros((), dtype=pg.dtype, device=pg.device)
            ),
        }
        if support_complete:
            metrics["sc/frac_support_unproven"] = _masked_mean(
                st["support_unproven"].to(pg.dtype), loss_mask
            )
        if is_mode == "tis":
            metrics["sc/frac_truncated"] = _masked_mean(
                (st["ratio_y"] > tis_clip).to(pg.dtype), loss_mask
            )
        elif is_mode == "icepop":
            metrics["sc/frac_masked"] = _masked_mean(
                (st["w_y"] == 0).to(pg.dtype), loss_mask
            )
    return loss, metrics


def score_centering_loss_from_config(
    sc: dict,
    *,
    token_lp: Tensor,
    head_lp: Optional[Tensor],
    q_y: Optional[Tensor],
    q_head: Optional[Tensor],
    advantages: Tensor,
    mask: Optional[Tensor],
    algo_name: str,
    loss_agg_mode: str,
    dp_size: int,
    batch_num_tokens: Optional[int],
    global_batch_size: Optional[int],
) -> tuple[Tensor, dict[str, float]]:
    """Engine-facing entry point shared by the Megatron and FSDP2 loss paths.

    For ``base_objective: reinforce`` this is the whole policy loss; for
    ``ppo_clip_additive`` it is only the correction term, which the caller adds
    to the algorithm's own clip loss (see :func:`score_centering_replaces_loss`).

    ``sc`` is ``algorithm.score_centering`` as a plain dict (from
    ``meta["algo_config"]``). Aggregation follows the clip loss each algorithm
    uses today: GRPO honours ``loss_agg_mode``, DAPO is token-mean.

    Returns ``(loss, stats)`` where ``stats`` holds token *sums*
    (``sc/<name>_sum``) and the token count ``sc_tok``, so micro-batches, rows
    and workers can be merged into exact token means on the controller.
    """
    if head_lp is None or q_y is None or q_head is None:
        raise RuntimeError(
            "score_centering needs head_log_probs (trainer), rollout_log_probs and "
            "rollout_topk_logprobs (sampler); got "
            f"head_log_probs={head_lp is not None}, rollout_log_probs={q_y is not None}, "
            f"rollout_topk_logprobs={q_head is not None}"
        )
    if mask is None:
        mask = torch.ones_like(token_lp)
    agg_mode = loss_agg_mode if algo_name == "grpo" else "token-mean"
    loss, metrics = score_centering_pg_loss(
        token_lp, head_lp, q_y, q_head, advantages,
        mask=mask,
        is_mode=str(sc.get("is_mode", "tis")),
        tis_clip=float(sc.get("tis_clip", 2.0)),
        icepop_low=float(sc.get("icepop_low", 0.5)),
        icepop_high=float(sc.get("icepop_high", 5.0)),
        eps=float(sc.get("eps", 1e-6)),
        support_complete=bool(sc.get("support_replay", False)),
        rho_pin_tail=float(sc.get("rho_pin_tail", 0.0)),
        correction_only=sc.get("base_objective", "reinforce") == "ppo_clip_additive",
        loss_agg_mode=agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
    )
    tok = float(mask.sum())
    stats = {f"{k}_sum": float(v) * tok for k, v in metrics.items() if k != "sc/head_coverage_p01"}
    stats["sc_tok"] = tok
    return loss, stats


def score_centering_replaces_loss(sc: Optional[dict]) -> bool:
    """True when score centering *is* the policy loss (``base_objective: reinforce``);
    False when it is added to the clip loss, or disabled."""
    return sc is not None and sc.get("base_objective", "reinforce") == "reinforce"


def score_centering_enabled(algo_config: dict) -> Optional[dict]:
    """``algorithm.score_centering`` (plain dict) when enabled, else ``None``."""
    sc = (algo_config or {}).get("score_centering")
    return sc if isinstance(sc, dict) and sc.get("enabled") else None


# --------------------------------------------------------------------------- #
# Setup-time guards
# --------------------------------------------------------------------------- #

# Backends whose loss path is wired for score centering. ATOM rollout capture
# (P6) follows; see docs/score_centering_design.md section 5.
_SUPPORTED_TRAINING_BACKENDS = ("megatron_native", "fsdp2")
_SUPPORTED_GENERATION_BACKENDS = ("vllm",)
_SUPPORTED_ALGORITHMS = ("dapo", "grpo")


def validate_score_centering(cfg: "LumenRLConfig") -> None:
    """Fail at setup on combinations score centering v1 does not support.

    No-op when ``algorithm.score_centering.enabled`` is false.
    """
    sc = cfg.algorithm.score_centering
    if not sc.enabled:
        return

    algo = cfg.algorithm
    pol = cfg.policy
    vcfg = pol.generation.vllm_cfg
    rc = cfg.quantization.rollout_correction
    errors: list[str] = []

    def need(ok: bool, msg: str) -> None:
        if not ok:
            errors.append(msg)

    name = str(algo.name).lower()
    need(
        str(pol.generation_backend) in _SUPPORTED_GENERATION_BACKENDS,
        f"policy.generation_backend={pol.generation_backend!r}; v1 captures the "
        f"sampler head from {_SUPPORTED_GENERATION_BACKENDS} only (ATOM is phase P6)",
    )
    need(
        str(pol.training_backend) in _SUPPORTED_TRAINING_BACKENDS,
        f"policy.training_backend={pol.training_backend!r}; the score-centering loss "
        f"is wired for {_SUPPORTED_TRAINING_BACKENDS} only",
    )
    need(
        name in _SUPPORTED_ALGORITHMS,
        f"algorithm.name={algo.name!r}; score centering replaces the policy loss of "
        f"{_SUPPORTED_ALGORITHMS} only",
    )
    need(
        not (name == "dapo" and algo.dapo.loss_mode == "gmpo"),
        "algorithm.dapo.loss_mode='gmpo' is a different objective; use 'token_level'",
    )
    need(
        algo.loss_mode in ("", "vanilla"),
        f"algorithm.loss_mode={algo.loss_mode!r}; score centering defines its own policy loss",
    )
    kl = algo.dapo.kl_coeff if name == "dapo" else algo.grpo.kl_coeff
    need(
        kl == 0.0,
        f"algorithm.{name}.kl_coeff={kl}; a KL loss term is not supported with score centering in v1",
    )
    need(
        not algo.distillation.enabled,
        "algorithm.distillation.enabled; distillation losses are not supported with score centering",
    )
    need(
        bool(cfg.controller.ray.enabled),
        "controller.ray.enabled must be true; the sampler head is captured on the Ray "
        "controller rollout path only",
    )
    need(
        str(vcfg.transport) == "ray_http",
        f"policy.generation.vllm_cfg.transport={vcfg.transport!r}; the sampler head is "
        "captured by the ray_http vLLM server only",
    )
    need(
        bool(vcfg.calculate_log_probs),
        "policy.generation.vllm_cfg.calculate_log_probs must be true (the sampled-token "
        "rollout log-prob is q_y)",
    )
    # The proportional tail model needs unfiltered sampling. Filtered sampling
    # needs support replay (design doc, appendix A.5): exact only when the
    # sampler's support always fits in the captured head, i.e. vLLM
    # top_k <= score_centering.top_k (top_p then only shrinks it further).
    s_top_k = int(vcfg.top_k)
    s_filtered = s_top_k > 0 or float(vcfg.top_p) < 1.0
    if sc.support_replay:
        need(
            s_filtered,
            "score_centering.support_replay needs filtered sampling "
            f"(policy.generation.vllm_cfg.top_k in [1, {sc.top_k})); sampling is unfiltered, "
            "so set support_replay: false",
        )
        need(
            not s_filtered or 0 < s_top_k < int(sc.top_k),
            f"policy.generation.vllm_cfg.top_k={s_top_k}, top_p={vcfg.top_p}; "
            "score_centering.support_replay needs the sampler support to fit in the head: "
            f"top_k in [1, score_centering.top_k={sc.top_k}) -- strictly smaller, because vLLM "
            "keeps every token tied at the k-th logit, and top-p without such a top_k can "
            "exceed the head",
        )
    else:
        need(
            float(vcfg.top_p) == 1.0,
            f"policy.generation.vllm_cfg.top_p={vcfg.top_p}; filtered sampling needs "
            "score_centering.support_replay (with top_k <= score_centering.top_k)",
        )
        need(
            s_top_k in (-1, 0),
            f"policy.generation.vllm_cfg.top_k={s_top_k}; filtered sampling needs "
            "score_centering.support_replay: true (and top_k <= score_centering.top_k)",
        )
    # The trainer coerces temperature 0 to 1 for DAPO/GRPO (_ray_sampling_params).
    need(
        float(vcfg.temperature) >= 0.0,
        f"policy.generation.vllm_cfg.temperature={vcfg.temperature} must be >= 0",
    )
    need(
        rc.rollout_is in ("", None, "token"),
        f"quantization.rollout_correction.rollout_is={rc.rollout_is!r}; score centering "
        "applies its IS rule per token inside the correction (set rollout_is to '' or 'token')",
    )
    need(
        not rc.rollout_is_batch_normalize,
        "quantization.rollout_correction.rollout_is_batch_normalize is not supported with "
        "score centering",
    )

    if errors:
        raise ValueError(
            "algorithm.score_centering.enabled=true is not supported with this config:\n  - "
            + "\n  - ".join(errors)
        )

    if rc.rollout_is == "token" and sc.base_objective == "reinforce":
        logger.warning(
            "score_centering: controller token-IS weights are not applied; the IS rule is "
            "algorithm.score_centering.is_mode=%r (tis_clip=%s)",
            sc.is_mode,
            sc.tis_clip,
        )


# --------------------------------------------------------------------------- #
# Rollout capture helpers (controller side)
# --------------------------------------------------------------------------- #


def validate_topk_rows(ids: Tensor, logprobs: Tensor, *, tag: str) -> None:
    """Sanity-check one response's sampler head ``[R, k]`` before training on it.

    ``-inf`` slots are allowed (filler); everything else must be a real
    log-probability, ids must be distinct per position, and the head mass must
    not exceed 1 (beyond fp32 rounding).
    """
    ids = torch.as_tensor(ids)
    lps = torch.as_tensor(logprobs).float()
    if ids.ndim != 2 or ids.shape != lps.shape:
        raise ValueError(
            f"score_centering {tag}: head shape mismatch ids={tuple(ids.shape)} "
            f"logprobs={tuple(lps.shape)}"
        )
    valid = torch.isfinite(lps)
    if torch.isnan(lps).any() or torch.isposinf(lps).any():
        raise ValueError(f"score_centering {tag}: head log-probs must be finite or -inf")
    if (lps[valid] > 1e-5).any():
        raise ValueError(f"score_centering {tag}: head log-prob > 0 (max {lps[valid].max():.3g})")
    mass = torch.where(valid, lps.exp(), torch.zeros_like(lps)).sum(-1)
    if (mass > 1.0001).any():
        raise ValueError(f"score_centering {tag}: head mass {mass.max():.6f} > 1")
    # Distinct ids among valid slots: sort, then no equal neighbours.
    keyed = torch.where(valid, ids.long(), -1 - torch.arange(ids.shape[1]).expand_as(ids))
    srt = keyed.sort(-1).values
    if ids.shape[1] > 1 and (srt[:, 1:] == srt[:, :-1]).any():
        raise ValueError(f"score_centering {tag}: duplicate token ids in the head")


def densify_rollout_topk(
    rows: list,
    seq_mask: Tensor,
    prompt_lengths: list[int],
    k: int,
) -> tuple[Tensor, Tensor]:
    """Place per-response heads ``(ids [R_i, k], logprobs [R_i, k])`` into the
    left-padded ``[B, S-1, k]`` layout of ``rollout_log_probs``.

    Row ``i`` occupies columns ``start + plen_i - 1 ...`` with
    ``start = S - seq_len_i`` (same rule as the rollout log-prob assembly).
    Elsewhere ids are 0 and log-probs ``-inf`` (invalid slot).
    """
    b, s = seq_mask.shape
    out_ids = torch.zeros(b, s - 1, k, dtype=torch.int32)
    out_lps = torch.full((b, s - 1, k), float("-inf"), dtype=torch.float32)
    seq_lens = seq_mask.sum(-1).tolist()
    for i, (ids, lps) in enumerate(rows):
        ids = torch.as_tensor(ids)
        lps = torch.as_tensor(lps)
        col = s - int(seq_lens[i]) + int(prompt_lengths[i]) - 1
        n = min(ids.shape[0], s - 1 - col)
        if n <= 0:
            continue
        out_ids[i, col:col + n] = ids[:n].to(torch.int32)
        out_lps[i, col:col + n] = lps[:n].to(torch.float32)
    return out_ids, out_lps


def vllm_engine_kwargs(sc) -> dict:
    """Extra vLLM engine args for score centering (empty when disabled).

    ``processed_logprobs`` reports log-probs after temperature (and after any
    top-k/top-p masking), i.e. the distribution actually sampled; vLLM's
    default ``raw_logprobs`` is pre-temperature. ``max_logprobs`` defaults to
    20 in vLLM, below the default head size.
    """
    if not sc.enabled:
        return {}
    return {"logprobs_mode": "processed_logprobs", "max_logprobs": max(20, int(sc.top_k))}
