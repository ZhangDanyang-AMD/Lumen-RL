"""Score-centering estimator (arXiv 2609.20807, eq. 14).

The checks mirror the ones every reference implementation (verl, slime, Vime,
Miles, torchtitan) converged on:

* k = V equals exact full-vocabulary centering;
* constant advantage gives zero *expected* gradient when the tail model holds;
* on-policy (p == q) reduces to plain REINFORCE, bitwise;
* f(r) = r (exact IS) makes the correction vanish;
* masked / -inf slots, y outside the head, empty rows stay finite;
* gradient scale matches the DAPO clip loss at ratio 1.
"""

from __future__ import annotations

import math

import pytest
import torch

from lumenrl.algorithms.loss_functions import asymmetric_clip_loss
from lumenrl.algorithms.score_centering import (
    make_weight_fn,
    score_centering_pg_loss,
    score_centering_token_loss,
)

F64 = torch.float64


def _tis(c: float = 2.0):
    return make_weight_fn("tis", tis_clip=c)


def _dists(v: int = 12, k: int = 5, seed: int = 0, tail_exact: bool = True):
    """Trainer logits ``z`` (leaf), sampler log-probs ``lq`` and sampler head ``H``.

    With ``tail_exact`` the sampler's tail is exactly ``rho * p`` so the
    proportional tail model is correct and centering is exact.
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(v, generator=g, dtype=F64)
    p = torch.softmax(z, -1)
    q = torch.softmax(z + 0.7 * torch.randn(v, generator=g, dtype=F64), -1)
    head = torch.topk(q, k).indices
    if tail_exact:
        in_head = torch.zeros(v, dtype=torch.bool)
        in_head[head] = True
        rho = (1 - q[in_head].sum()) / (1 - p[in_head].sum())
        q = torch.where(in_head, q, rho * p)
    return z.requires_grad_(True), q.log(), head


def _token_inputs(z, lq, head, ys):
    """One token per sampled id in ``ys`` (same prefix), shapes ``[1, len(ys)]``."""
    lp = torch.log_softmax(z, -1)
    n = len(ys)
    ys = torch.as_tensor(ys)
    logp_y = lp[ys].view(1, n)
    q_y = lq[ys].view(1, n)
    head_logp = lp[head].expand(n, -1).view(1, n, -1)
    q_head = lq[head].expand(n, -1).view(1, n, -1)
    return logp_y, head_logp, q_y, q_head


@pytest.mark.parametrize("mode", ["none", "tis", "icepop"])
def test_full_vocab_head_equals_exact_centering(mode: str) -> None:
    v = 10
    z, lq, _ = _dists(v=v, k=v, tail_exact=False)
    head = torch.arange(v)
    wf = make_weight_fn(mode, tis_clip=2.0, icepop_low=0.5, icepop_high=5.0)
    y, adv = 3, torch.tensor([[0.7]], dtype=F64)

    logp_y, head_logp, q_y, q_head = _token_inputs(z, lq, head, [y])
    pg, _ = score_centering_token_loss(logp_y, head_logp, q_y, q_head, adv, weight_fn=wf)
    (g_sc,) = torch.autograd.grad(pg.sum(), z)

    lp = torch.log_softmax(z, -1)
    q = lq.exp()
    with torch.no_grad():
        w = wf((lp - lq).exp())
    ref = -adv.squeeze() * (w[y] * lp[y] - (q * w * lp).sum())
    (g_ref,) = torch.autograd.grad(ref, z)

    torch.testing.assert_close(g_sc, g_ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", ["none", "tis", "icepop"])
def test_constant_advantage_has_zero_expected_gradient(mode: str) -> None:
    v = 12
    z, lq, head = _dists(v=v, k=4, seed=1, tail_exact=True)
    wf = make_weight_fn(mode, tis_clip=1.5, icepop_low=0.8, icepop_high=1.25)
    ys = list(range(v))  # every outcome, weighted by its sampler probability
    logp_y, head_logp, q_y, q_head = _token_inputs(z, lq, head, ys)
    adv = torch.full((1, v), 1.3, dtype=F64)

    pg, _ = score_centering_token_loss(logp_y, head_logp, q_y, q_head, adv, weight_fn=wf)
    expected = (pg * lq.exp().view(1, v)).sum()
    (g,) = torch.autograd.grad(expected, z)

    assert g.abs().max().item() < 1e-12


def test_constant_advantage_plain_reinforce_has_nonzero_drift() -> None:
    """Sanity check that the previous test can fail: without the correction
    the expected gradient under p != q is not zero."""
    v = 12
    z, lq, head = _dists(v=v, k=4, seed=1, tail_exact=True)
    lp = torch.log_softmax(z, -1)
    expected = -(lq.exp() * lp).sum()
    (g,) = torch.autograd.grad(expected, z)
    assert g.abs().max().item() > 1e-3


@pytest.mark.parametrize("mode", ["none", "tis", "icepop"])
def test_on_policy_is_plain_reinforce_bitwise(mode: str) -> None:
    torch.manual_seed(0)
    n, l, v, k = 2, 3, 20, 6
    z = torch.randn(n, l, v, requires_grad=True)
    lp = torch.log_softmax(z, -1)
    ids = torch.topk(lp.detach(), k, -1).indices
    y = torch.randint(0, v, (n, l))
    logp_y = lp.gather(-1, y[..., None]).squeeze(-1)
    head_logp = lp.gather(-1, ids)
    adv = torch.randn(n, l)
    mask = torch.ones(n, l)
    wf = make_weight_fn(mode, tis_clip=2.0, icepop_low=0.5, icepop_high=5.0)

    pg, _ = score_centering_token_loss(
        logp_y, head_logp, logp_y.detach(), head_logp.detach(), adv, weight_fn=wf
    )
    (g_sc,) = torch.autograd.grad((pg * mask).sum(), z, retain_graph=True)
    (g_ref,) = torch.autograd.grad((-adv * logp_y * mask).sum(), z)

    assert torch.equal(g_sc, g_ref)


def test_exact_importance_weight_makes_correction_vanish() -> None:
    z, lq, head = _dists(v=12, k=4, seed=2, tail_exact=False)
    logp_y, head_logp, q_y, q_head = _token_inputs(z, lq, head, [0, 5, 11])
    adv = torch.ones(1, 3, dtype=F64)

    _, stats = score_centering_token_loss(
        logp_y, head_logp, q_y, q_head, adv, weight_fn=lambda r: r
    )
    assert stats["correction"].abs().max().item() < 1e-12


def test_weight_fn_modes() -> None:
    r = torch.tensor([0.1, 0.6, 1.0, 3.0, 9.0])
    assert torch.equal(make_weight_fn("none")(r), torch.ones_like(r))
    assert torch.equal(make_weight_fn("tis", tis_clip=2.0)(r), torch.tensor([0.1, 0.6, 1.0, 2.0, 2.0]))
    icepop = make_weight_fn("icepop", icepop_low=0.5, icepop_high=5.0)(r)
    assert torch.equal(icepop, torch.tensor([0.0, 0.6, 1.0, 3.0, 0.0]))
    with pytest.raises(ValueError):
        make_weight_fn("sequence")


def test_tail_coefficient_matches_closed_form() -> None:
    """alpha = rho * f(1/rho): min(1, c*rho) for TIS, rho for none."""
    z, lq, head = _dists(v=12, k=3, seed=3, tail_exact=False)
    logp_y, head_logp, q_y, q_head = _token_inputs(z, lq, head, [0])
    adv = torch.ones(1, 1, dtype=F64)
    p_mass = head_logp.detach().exp().sum()
    q_mass = q_head.exp().sum()
    rho = ((1 - q_mass) / (1 - p_mass)).item()

    for mode, c, alpha in [("none", 2.0, rho), ("tis", 0.5, min(1.0, 0.5 * rho))]:
        _, stats = score_centering_token_loss(
            logp_y, head_logp, q_y, q_head, adv, weight_fn=make_weight_fn(mode, tis_clip=c)
        )
        assert math.isclose(stats["rho"].item(), rho, rel_tol=1e-12)
        assert math.isclose(stats["alpha"].item(), alpha, rel_tol=1e-12)


def test_invalid_slots_and_sample_outside_head_stay_finite() -> None:
    torch.manual_seed(0)
    n, l, v, k = 2, 4, 30, 5
    z = torch.randn(n, l, v, requires_grad=True)
    lp = torch.log_softmax(z, -1)
    lq = torch.log_softmax(z.detach() + 0.3 * torch.randn(n, l, v), -1)
    ids = torch.topk(lq, k, -1).indices
    q_head = lq.gather(-1, ids)
    # Row 0 / pos 0: prompt position, every slot invalid. Row 1 / pos 2: two
    # filler slots with garbage ids (as vLLM returns under filtered sampling).
    q_head[0, 0] = float("-inf")
    q_head[1, 2, -2:] = float("-inf")
    ids[1, 2, -2:] = 0
    y = torch.argmin(lq, -1)  # least likely token: never in the head
    logp_y = lp.gather(-1, y[..., None]).squeeze(-1)
    q_y = lq.gather(-1, y[..., None]).squeeze(-1)
    head_logp = lp.gather(-1, ids)
    mask = torch.ones(n, l)
    mask[0, 0] = 0

    loss, metrics = score_centering_pg_loss(
        logp_y, head_logp, q_y, q_head, torch.randn(n, l),
        mask=mask, is_mode="tis", tis_clip=2.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(z.grad).all()
    assert all(torch.isfinite(m).all() for m in metrics.values())


def test_all_masked_batch_returns_zero_fp32_loss() -> None:
    n, l, k = 1, 3, 4
    head_logp = torch.full((n, l, k), -2.0, requires_grad=True)
    logp_y = torch.full((n, l), -2.0, requires_grad=True)
    q_head = torch.full((n, l, k), float("-inf"))
    loss, metrics = score_centering_pg_loss(
        logp_y, head_logp, torch.zeros(n, l), q_head, torch.ones(n, l),
        mask=torch.zeros(n, l), is_mode="tis", batch_num_tokens=8, dp_size=2,
    )
    loss.backward()
    assert loss.dtype == torch.float32
    assert loss.item() == 0.0
    assert all(torch.isfinite(m).all() for m in metrics.values())


def test_gradient_scale_matches_dapo_clip_loss_on_policy() -> None:
    torch.manual_seed(1)
    n, l, v, k = 3, 5, 16, 4
    z = torch.randn(n, l, v, requires_grad=True)
    lp = torch.log_softmax(z, -1)
    ids = torch.topk(lp.detach(), k, -1).indices
    y = torch.randint(0, v, (n, l))
    logp_y = lp.gather(-1, y[..., None]).squeeze(-1)
    head_logp = lp.gather(-1, ids)
    adv = torch.randn(n, l)
    mask = (torch.rand(n, l) > 0.3).float()
    kw = dict(mask=mask, batch_num_tokens=40, dp_size=4, loss_agg_mode="token-mean")

    sc, _ = score_centering_pg_loss(
        logp_y, head_logp, logp_y.detach(), head_logp.detach(), adv, is_mode="tis", **kw
    )
    (g_sc,) = torch.autograd.grad(sc, z, retain_graph=True)
    dapo = asymmetric_clip_loss(logp_y, logp_y.detach(), adv, 0.2, 0.28, clip_ratio_c=3.0, **kw)
    (g_dapo,) = torch.autograd.grad(dapo, z)

    torch.testing.assert_close(g_sc, g_dapo)


def test_metrics_report_head_mass_and_truncation() -> None:
    z, lq, head = _dists(v=12, k=4, seed=4, tail_exact=False)
    ys = list(range(12))
    logp_y, head_logp, q_y, q_head = _token_inputs(z, lq, head, ys)
    _, metrics = score_centering_pg_loss(
        logp_y, head_logp, q_y, q_head, torch.ones(1, 12, dtype=F64),
        mask=torch.ones(1, 12, dtype=F64), is_mode="tis", tis_clip=1.0,
    )
    ratio = (logp_y - q_y).exp().detach()
    assert math.isclose(metrics["sc/sampler_head_mass"].item(), q_head[0, 0].exp().sum().item(), rel_tol=1e-9)
    assert math.isclose(metrics["sc/train_head_mass"].item(), head_logp[0, 0].exp().sum().item(), rel_tol=1e-9)
    assert math.isclose(metrics["sc/frac_truncated"].item(), (ratio > 1.0).double().mean().item(), rel_tol=1e-9)
    for key in ("sc/tail_ratio_mean", "sc/correction_abs", "sc/sample_weight_mean", "sc/head_coverage_p01"):
        assert key in metrics


# --------------------------------------------------------------------------- #
# Support replay (filtered sampling whose support fits in the head)
# --------------------------------------------------------------------------- #


def _filtered(v=12, k=5, support=4, seed=5):
    """Sampler that only draws from its top-``support`` tokens (vLLM top-k sampling).

    Returns trainer logits ``z`` (leaf), the sampler's processed log-probs ``lq``
    (renormalized over S, ``-inf`` elsewhere) and the head ids (S + filler).
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(v, generator=g, dtype=F64)
    q_raw = torch.log_softmax(z + 0.8 * torch.randn(v, generator=g, dtype=F64), -1)
    s_ids = torch.topk(q_raw, support).indices
    lq = torch.full((v,), float("-inf"), dtype=F64)
    lq[s_ids] = torch.log_softmax(q_raw[s_ids], -1)
    head = torch.topk(lq, k).indices  # S first, then -inf filler
    return z.requires_grad_(True), lq, head, s_ids


@pytest.mark.parametrize("support", [4, 5])
@pytest.mark.parametrize("mode", ["none", "tis", "icepop"])
def test_support_replay_has_zero_expected_gradient(mode, support) -> None:
    z, lq, head, s_ids = _filtered(support=support)
    wf = make_weight_fn(mode, tis_clip=1.5, icepop_low=0.8, icepop_high=1.25)
    ys = s_ids.tolist()  # the sampler can only produce tokens in S
    logp_y, head_logp, q_y, q_head = _token_inputs(z, lq, head, ys)
    adv = torch.full((1, len(ys)), 0.9, dtype=F64)

    pg, st = score_centering_token_loss(
        logp_y, head_logp, q_y, q_head, adv, weight_fn=wf, support_complete=True
    )
    expected = (pg * lq[s_ids].exp().view(1, -1)).sum()
    (g,) = torch.autograd.grad(expected, z)

    assert g.abs().max().item() < 1e-12
    assert torch.equal(st["rho"], torch.ones_like(st["rho"]))


def test_support_replay_matches_exact_renormalized_centering() -> None:
    z, lq, head, s_ids = _filtered(support=4)
    wf = make_weight_fn("tis", tis_clip=2.0)
    y, adv = int(s_ids[2]), torch.tensor([[0.6]], dtype=F64)
    logp_y, head_logp, q_y, q_head = _token_inputs(z, lq, head, [y])
    pg, _ = score_centering_token_loss(
        logp_y, head_logp, q_y, q_head, adv, weight_fn=wf, support_complete=True
    )
    (g_sc,) = torch.autograd.grad(pg.sum(), z)

    lp = torch.log_softmax(z, -1)
    lp_s = lp[s_ids] - torch.logsumexp(lp[s_ids], -1)  # trainer renormalized over S
    q_s = lq[s_ids].exp()
    with torch.no_grad():
        w = wf((lp_s - lq[s_ids]).exp())
    j = int((s_ids == y).nonzero())
    ref = -adv.squeeze() * (w[j] * lp_s[j] - (q_s * w * lp_s).sum())
    (g_ref,) = torch.autograd.grad(ref, z)
    torch.testing.assert_close(g_sc, g_ref, rtol=1e-10, atol=1e-12)


def test_support_replay_leaves_out_of_support_logits_without_gradient() -> None:
    z, lq, head, s_ids = _filtered(support=4)
    logp_y, head_logp, q_y, q_head = _token_inputs(z, lq, head, s_ids.tolist())
    pg, _ = score_centering_token_loss(
        logp_y, head_logp, q_y, q_head, torch.randn(1, 4, dtype=F64),
        weight_fn=make_weight_fn("tis"), support_complete=True,
    )
    (g,) = torch.autograd.grad(pg.sum(), z)
    outside = torch.ones(12, dtype=torch.bool)
    outside[s_ids] = False
    assert g[outside].abs().max().item() == 0.0
    assert g[s_ids].abs().max().item() > 0.0


def test_support_replay_on_policy_is_reinforce_on_the_truncated_policy() -> None:
    z, _, head, s_ids = _filtered(support=4)
    lp = torch.log_softmax(z, -1)
    lq = torch.full((12,), float("-inf"), dtype=F64)
    lq[s_ids] = (lp[s_ids] - torch.logsumexp(lp[s_ids], -1)).detach()  # sampler == p_S
    ys = s_ids.tolist()
    logp_y, head_logp, q_y, q_head = _token_inputs(z, lq, head, ys)
    adv = torch.randn(1, 4, dtype=F64)
    pg, _ = score_centering_token_loss(
        logp_y, head_logp, q_y, q_head, adv, weight_fn=make_weight_fn("tis"),
        support_complete=True,
    )
    (g_sc,) = torch.autograd.grad(pg.sum(), z, retain_graph=True)
    lp_s = lp[s_ids] - torch.logsumexp(lp[s_ids], -1)
    (g_ref,) = torch.autograd.grad((-adv.view(-1) * lp_s).sum(), z)
    torch.testing.assert_close(g_sc, g_ref, rtol=1e-10, atol=1e-12)


def test_support_replay_rows_without_a_head_stay_finite() -> None:
    head_logp = torch.randn(1, 3, 4, requires_grad=True)
    q_head = torch.log_softmax(torch.randn(1, 3, 4), -1)
    q_head[0, 0] = float("-inf")  # prompt position
    loss, metrics = score_centering_pg_loss(
        head_logp[..., 0].detach().requires_grad_(True), head_logp, q_head[..., 0].nan_to_num(neginf=0.0),
        q_head, torch.ones(1, 3), mask=torch.tensor([[0.0, 1.0, 1.0]]), is_mode="tis",
        support_complete=True,
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(head_logp.grad).all()
    assert all(torch.isfinite(m).all() for m in metrics.values())


def test_support_replay_reports_positions_whose_support_is_unproven() -> None:
    """Completeness is proven by a -inf filler slot (vLLM ran out of support
    before filling the head); a full head may have cut the support (ties at the
    sampler's k-th logit keep extra tokens)."""
    head_logp = torch.log_softmax(torch.randn(1, 4, 6), -1)[..., :3].requires_grad_(True)
    q_head = torch.log_softmax(torch.randn(1, 4, 3), -1)
    q_head[0, 1, 2] = float("-inf")   # proven
    q_head[0, 3, 1:] = float("-inf")  # proven
    _, metrics = score_centering_pg_loss(
        head_logp[..., 0], head_logp, q_head[..., 0], q_head, torch.ones(1, 4),
        mask=torch.ones(1, 4), is_mode="tis", support_complete=True,
    )
    assert metrics["sc/frac_support_unproven"].item() == pytest.approx(0.5)
    _, plain = score_centering_pg_loss(
        head_logp[..., 0], head_logp, q_head[..., 0], q_head, torch.ones(1, 4),
        mask=torch.ones(1, 4), is_mode="tis",
    )
    assert "sc/frac_support_unproven" not in plain


def test_rho_is_pinned_when_both_tails_are_rounding_noise() -> None:
    """Near-deterministic tokens: both tails ~1e-6, their ratio is fp32 noise
    (measured: every rho < 0.5 on real rollouts came from tails < 1e-5)."""
    head_logp = torch.log(torch.tensor([[[0.999999, 0.0000005]]], dtype=F64)).requires_grad_(True)
    q_head = torch.log(torch.tensor([[[0.9999998, 0.0000001]]], dtype=F64))  # tails 1e-7 vs 5e-7
    args = (head_logp[..., 0], head_logp, q_head[..., 0], q_head, torch.ones(1, 1, dtype=F64))
    _, raw = score_centering_token_loss(*args, weight_fn=_tis(), eps=1e-12)
    _, pinned = score_centering_token_loss(*args, weight_fn=_tis(), eps=1e-12, rho_pin_tail=1e-5)
    assert raw["rho"].item() < 0.5
    assert pinned["rho"].item() == 1.0 and pinned["alpha"].item() == 1.0


def test_rho_pin_leaves_real_tails_alone() -> None:
    z, lq, head = _dists(v=12, k=3, seed=3, tail_exact=False)
    args = _token_inputs(z, lq, head, [0])
    _, a = score_centering_token_loss(*args, torch.ones(1, 1, dtype=F64), weight_fn=_tis())
    _, b = score_centering_token_loss(*args, torch.ones(1, 1, dtype=F64), weight_fn=_tis(), rho_pin_tail=1e-5)
    assert torch.equal(a["rho"], b["rho"])


# --------------------------------------------------------------------------- #
# Correction-only term (base_objective: ppo_clip_additive)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["none", "tis", "icepop"])
def test_correction_only_is_the_reinforce_loss_minus_its_sample_term(mode) -> None:
    """REINFORCE+SC = -A*w_y*log p_y + A*correction, so the additive term must be
    exactly that loss plus A*w_y*log p_y -- in value and in gradient."""
    z, lq, head = _dists(v=12, k=4, seed=6, tail_exact=False)
    args = _token_inputs(z, lq, head, [0, 3, 7, 11])
    adv = torch.randn(1, 4, dtype=F64)
    wf = make_weight_fn(mode, tis_clip=1.5, icepop_low=0.8, icepop_high=1.25)

    full, st = score_centering_token_loss(*args, adv, weight_fn=wf)
    corr, _ = score_centering_token_loss(*args, adv, weight_fn=wf, correction_only=True)
    ref = full + adv * st["w_y"] * args[0]

    torch.testing.assert_close(corr, ref)
    (g1,) = torch.autograd.grad(corr.sum(), z, retain_graph=True)
    (g2,) = torch.autograd.grad(ref.sum(), z)
    torch.testing.assert_close(g1, g2)


def test_correction_only_vanishes_on_policy() -> None:
    torch.manual_seed(0)
    lp = torch.log_softmax(torch.randn(2, 3, 20), -1)
    ids = torch.topk(lp, 5, -1).indices
    head = lp.gather(-1, ids)
    corr, _ = score_centering_token_loss(
        lp[..., 0], head, lp[..., 0], head, torch.randn(2, 3),
        weight_fn=make_weight_fn("tis"), correction_only=True,
    )
    assert torch.equal(corr, torch.zeros_like(corr))
