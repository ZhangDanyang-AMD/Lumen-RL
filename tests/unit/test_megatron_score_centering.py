"""Score-centering wiring in the Megatron engines (P4), tested without Megatron."""

from __future__ import annotations

import pytest
import torch

from lumenrl.algorithms.score_centering import score_centering_pg_loss
from lumenrl.engine.training.megatron_base_engine import MegatronBaseEngine
from lumenrl.trainer.rl_trainer import RLTrainer

SC = {"enabled": True, "top_k": 3, "is_mode": "tis", "tis_clip": 2.0, "icepop_low": 0.5,
      "icepop_high": 5.0, "eps": 1e-6, "chunk_rows": 2, "validate_rows": True}


def _engine(chunk=0):
    e = object.__new__(MegatronBaseEngine)
    e._logprob_chunk_size = chunk
    return e


def _cfg(sc):
    algo = {"dapo": {"clip_ratio_low": 0.2, "clip_ratio_high": 0.28, "clip_ratio_c": 3.0},
            "score_centering": sc}

    def fn(key, default):
        v = algo["dapo"].get(key, algo.get(key, default))
        return default if v is None else v

    return fn


def _batch(b=2, s=7, k=3, v=11, seed=0):
    """Left-padded batch in the controller's [B, S-1(, k)] layout."""
    g = torch.Generator().manual_seed(seed)
    starts = [2, 0]
    resp_mask = torch.zeros(b, s - 1)
    resp_mask[0, 3:] = 1
    resp_mask[1, 2:] = 1
    q = torch.log_softmax(torch.randn(b, s - 1, v, generator=g), -1)
    ids = torch.topk(q, k, -1).indices.int()
    head_q = q.gather(-1, ids.long())
    head_q[resp_mask == 0] = float("-inf")
    t = {
        "old_log_probs": torch.randn(b, s - 1, generator=g) - 2,
        "rollout_log_probs": torch.randn(b, s - 1, generator=g) - 2,
        "response_mask": resp_mask,
        "advantages": torch.randn(b, s - 1, generator=g),
        "rollout_is_weights": torch.full((b, s - 1), 7.0),  # must be ignored under SC
        "rollout_topk_ids": ids,
        "rollout_topk_logprobs": head_q,
    }
    return t, starts


def test_token_and_head_logprobs_share_one_normalization() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, 11, requires_grad=True)
    targets = torch.randint(0, 11, (5,))
    head_ids = torch.randint(0, 11, (5, 3))
    e = _engine()
    tok, head = e._token_head_logprob_train(logits, targets, head_ids, chunk_rows=2)
    torch.testing.assert_close(tok, e._token_logprob_train(logits, targets))
    torch.testing.assert_close(head, torch.log_softmax(logits, -1).gather(-1, head_ids))


def test_row_head_ids_slice_matches_logit_positions() -> None:
    t, _ = _batch()
    ids = MegatronBaseEngine._sc_row_head_ids(t, 0, start=2, n=4, device="cpu")
    torch.testing.assert_close(ids, t["rollout_topk_ids"][0, 2:6].long())


@pytest.mark.parametrize("row", [0, 1])
def test_row_policy_loss_uses_score_centering(row) -> None:
    t, starts = _batch()
    start = starts[row]
    lm = t["old_log_probs"].shape[1] - start
    torch.manual_seed(row)
    logits = torch.randn(lm, 11, requires_grad=True)
    lp = torch.log_softmax(logits, -1)
    targets = torch.randint(0, 11, (lm,))
    head_ids = t["rollout_topk_ids"][row, start:].long()
    token_lp = lp.gather(-1, targets[:, None]).view(1, -1)
    head_lp = lp.gather(-1, head_ids).view(1, lm, -1)

    loss, stats = _engine()._row_policy_loss(
        t, row, start, token_lp, "dapo", _cfg(SC), 40, 4, "token-mean", None, head_lp=head_lp,
    )
    ref, ref_m = score_centering_pg_loss(
        token_lp, head_lp,
        t["rollout_log_probs"][row, start:].view(1, -1),
        t["rollout_topk_logprobs"][row, start:].view(1, lm, -1),
        t["advantages"][row, start:].view(1, -1),
        mask=t["response_mask"][row, start:].view(1, -1),
        is_mode="tis", tis_clip=2.0, batch_num_tokens=40, dp_size=4,
    )
    torch.testing.assert_close(loss, ref)
    tok = float(t["response_mask"][row].sum())
    assert stats["sc_tok"] == tok
    assert stats["sc/sampler_head_mass_sum"] == pytest.approx(float(ref_m["sc/sampler_head_mass"]) * tok)
    assert "sc/frac_low_head_mass_sum" in stats
    assert stats["rc_kl_tok"] == tok  # rollout_corr/kl keeps working


def test_row_policy_loss_without_score_centering_is_unchanged() -> None:
    t, starts = _batch()
    lm = t["old_log_probs"].shape[1]
    token_lp = torch.randn(1, lm, requires_grad=True)
    off = dict(SC, enabled=False)
    a, sa = _engine()._row_policy_loss(t, 1, 0, token_lp, "dapo", _cfg(off), 40, 4, "token-mean", None)
    b, sb = _engine()._row_policy_loss(t, 1, 0, token_lp, "dapo", _cfg(None), 40, 4, "token-mean", None)
    torch.testing.assert_close(a, b)
    assert "sc_tok" not in sa


def test_row_policy_loss_requires_the_head_when_enabled() -> None:
    t, _ = _batch()
    del t["rollout_topk_logprobs"]
    lm = t["old_log_probs"].shape[1]
    with pytest.raises(RuntimeError, match="rollout_topk_logprobs"):
        _engine()._row_policy_loss(
            t, 1, 0, torch.randn(1, lm), "dapo", _cfg(SC), 40, 4, "token-mean", None,
            head_lp=torch.randn(1, lm, 3),
        )


def test_controller_turns_sc_sums_into_token_means() -> None:
    m = {"sc/sampler_head_mass_sum": 9.0, "sc/correction_abs_sum": 3.0, "sc_tok": 10.0, "loss": 1.0}
    RLTrainer._finalize_score_centering_metrics(m)
    assert m == {"sc/sampler_head_mass": 0.9, "sc/correction_abs": 0.3, "loss": 1.0}


def test_row_policy_loss_ppo_clip_additive_is_clip_plus_correction() -> None:
    t, starts = _batch()
    lm = t["old_log_probs"].shape[1]
    torch.manual_seed(4)
    logits = torch.randn(lm, 11, requires_grad=True)
    lp = torch.log_softmax(logits, -1)
    token_lp = lp[:, 0].view(1, -1)
    head_lp = lp.gather(-1, t["rollout_topk_ids"][1].long()).view(1, lm, -1)
    add = dict(SC, base_objective="ppo_clip_additive")

    total, stats = _engine()._row_policy_loss(
        t, 1, 0, token_lp, "dapo", _cfg(add), 40, 4, "token-mean", None, head_lp=head_lp,
    )
    clip, _ = _engine()._row_policy_loss(
        t, 1, 0, token_lp, "dapo", _cfg(None), 40, 4, "token-mean", None,
    )
    corr, _ = score_centering_pg_loss(
        token_lp, head_lp, t["rollout_log_probs"][1].view(1, -1),
        t["rollout_topk_logprobs"][1].view(1, lm, -1), t["advantages"][1].view(1, -1),
        mask=t["response_mask"][1].view(1, -1), is_mode="tis", tis_clip=2.0,
        batch_num_tokens=40, dp_size=4, correction_only=True,
    )
    torch.testing.assert_close(total, clip + corr)
    assert stats["sc_tok"] == float(t["response_mask"][1].sum())
