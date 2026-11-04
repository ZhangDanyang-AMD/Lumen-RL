"""Score-centering wiring in the FSDP2 engine and actor worker (P5), on CPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lumenrl.algorithms.score_centering import score_centering_pg_loss, validate_score_centering
from lumenrl.core.config import LumenRLConfig, ScoreCenteringConfig
from lumenrl.engine.training.fsdp_engine import FSDP2Engine as FSDPEngine
from lumenrl.engine.training.packing import pack_sequences
from lumenrl.workers.actor_worker import LumenActorWorker

SC = {"enabled": True, "top_k": 3, "is_mode": "tis", "tis_clip": 2.0, "icepop_low": 0.5,
      "icepop_high": 5.0, "eps": 1e-6, "chunk_rows": 4096, "validate_rows": True}


def _micro(b=2, s=6, v=13, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    am = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])[:b, :s]
    ids = torch.randint(0, v, (b, s), generator=g) * am
    head = torch.randint(0, v, (b, s - 1, k), generator=g).int()
    return {"input_ids": ids, "attention_mask": am, "rollout_topk_ids": head,
            "temperature": 0.7, "_padded_seq_len": s}


def test_packed_outputs_add_head_log_probs_aligned_with_log_probs() -> None:
    mb = _micro()
    packed = pack_sequences(mb["input_ids"], mb["attention_mask"])
    torch.manual_seed(0)
    logits = torch.randn(1, packed.input_ids.shape[1], 13)
    eng = object.__new__(FSDPEngine)

    out = eng._prepare_packed_outputs(logits, packed, mb)
    plain = eng._prepare_packed_outputs(logits, packed, {k: v for k, v in mb.items() if k != "rollout_topk_ids"})

    torch.testing.assert_close(out["log_probs"], plain["log_probs"])
    assert out["head_log_probs"].shape == (2, 5, 3)
    # Row 1 is unpadded: column j is the distribution after token j.
    lsm = torch.log_softmax(logits[0] / 0.7, -1)
    off = int(packed.cu_seqlens[1])
    for j in range(5):
        torch.testing.assert_close(
            out["head_log_probs"][1, j], lsm[off + j][mb["rollout_topk_ids"][1, j].long()]
        )
    # Row 0 has 4 real tokens -> 3 predictions in columns 2..4.
    torch.testing.assert_close(out["head_log_probs"][0, 2], lsm[0][mb["rollout_topk_ids"][0, 2].long()])
    assert (out["head_log_probs"][0, :2] == 0).all()


def test_unpacked_forward_rejects_score_centering() -> None:
    eng = object.__new__(FSDPEngine)
    mb = _micro()
    with pytest.raises(NotImplementedError, match="packed"):
        eng._prepare_model_outputs(torch.randn(2, 6, 13), mb["input_ids"], mb)


def test_forward_step_does_not_keep_the_head_in_step_outputs() -> None:
    mb = _micro()
    packed = pack_sequences(mb["input_ids"], mb["attention_mask"])
    mb["_packed"] = packed
    eng = object.__new__(FSDPEngine)
    eng.module = lambda **kw: SimpleNamespace(logits=torch.randn(1, packed.input_ids.shape[1], 13))
    seen = {}

    def loss_fn(model_output, data):
        seen.update(model_output)
        return torch.zeros(()), {}

    _, meta = eng.forward_step(mb, loss_fn, forward_only=True)
    assert "head_log_probs" in seen
    assert "head_log_probs" not in meta["model_output"]


def _worker(sc):
    w = object.__new__(LumenActorWorker)
    w._pol_meta = {"algorithm": "dapo", "batch_num_tokens": 40, "dp_size": 4,
                   "algo_config": {"dapo": {"clip_ratio_low": 0.2, "clip_ratio_high": 0.28,
                                            "clip_ratio_c": 3.0}, "score_centering": sc}}
    return w


def _data(b=2, s=6, k=3, v=13, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(b, s - 1)
    mask[0, 2:] = 1
    mask[1, 1:] = 1
    q = torch.log_softmax(torch.randn(b, s - 1, v, generator=g), -1)
    ids = torch.topk(q, k, -1).indices
    qh = q.gather(-1, ids)
    qh[mask == 0] = float("-inf")
    return {
        "old_log_probs": torch.randn(b, s - 1, generator=g) - 2,
        "rollout_log_probs": torch.randn(b, s - 1, generator=g) - 2,
        "advantages": torch.randn(b, s - 1, generator=g),
        "response_mask": mask,
        "rollout_is_weights": torch.full((b, s - 1), 9.0),
        "rollout_topk_ids": ids.int(),
        "rollout_topk_logprobs": qh,
    }


def test_policy_loss_uses_score_centering_and_reports_token_sums() -> None:
    d = _data()
    torch.manual_seed(2)
    z = torch.randn(2, 5, 13, requires_grad=True)
    lp = torch.log_softmax(z, -1)
    tok = lp[..., 0]
    head = lp.gather(-1, d["rollout_topk_ids"].long())

    loss, m = _worker(SC)._policy_loss_fn({"log_probs": tok, "head_log_probs": head}, d)
    ref, ref_m = score_centering_pg_loss(
        tok, head, d["rollout_log_probs"], d["rollout_topk_logprobs"], d["advantages"],
        mask=d["response_mask"], is_mode="tis", tis_clip=2.0, batch_num_tokens=40, dp_size=4,
    )
    torch.testing.assert_close(loss, ref)
    n = float(d["response_mask"].sum())
    assert m["sc_tok"] == n
    assert m["sc/sampler_head_mass_sum"] == pytest.approx(float(ref_m["sc/sampler_head_mass"]) * n)
    assert m["rollout_corr_kl_tok"] == n


def test_policy_loss_requires_head_tensors_when_enabled() -> None:
    d = _data()
    tok = torch.randn(2, 5)
    with pytest.raises(RuntimeError, match="head_log_probs"):
        _worker(SC)._policy_loss_fn({"log_probs": tok}, d)


def test_policy_loss_without_score_centering_is_unchanged() -> None:
    d = _data()
    tok = torch.randn(2, 5)
    a, _ = _worker(dict(SC, enabled=False))._policy_loss_fn({"log_probs": tok}, d)
    b, _ = _worker(None)._policy_loss_fn({"log_probs": tok}, d)
    torch.testing.assert_close(a, b)


def test_guard_accepts_fsdp2() -> None:
    cfg = LumenRLConfig()
    cfg.algorithm.name = "dapo"
    cfg.policy.training_backend = "fsdp2"
    cfg.policy.generation_backend = "vllm"
    cfg.policy.generation.vllm_cfg.calculate_log_probs = True
    cfg.policy.generation.vllm_cfg.transport = "ray_http"
    cfg.controller.ray.enabled = True
    cfg.algorithm.score_centering = ScoreCenteringConfig(enabled=True)
    validate_score_centering(cfg)


def test_ppo_clip_additive_is_the_clip_loss_plus_the_correction() -> None:
    from lumenrl.algorithms.loss_functions import asymmetric_clip_loss
    from lumenrl.algorithms.score_centering import score_centering_pg_loss as _pg

    d = _data()
    torch.manual_seed(3)
    z = torch.randn(2, 5, 13, requires_grad=True)
    lp = torch.log_softmax(z, -1)
    tok = lp[..., 0]
    head = lp.gather(-1, d["rollout_topk_ids"].long())
    add = dict(SC, base_objective="ppo_clip_additive")

    loss, m = _worker(add)._policy_loss_fn({"log_probs": tok, "head_log_probs": head}, d)

    clip = asymmetric_clip_loss(
        tok, d["old_log_probs"], d["advantages"], 0.2, 0.28, mask=d["response_mask"],
        clip_ratio_c=3.0, batch_num_tokens=40, dp_size=4, rollout_is_weights=d["rollout_is_weights"],
    )
    corr, _ = _pg(
        tok, head, d["rollout_log_probs"], d["rollout_topk_logprobs"], d["advantages"],
        mask=d["response_mask"], is_mode="tis", tis_clip=2.0, batch_num_tokens=40, dp_size=4,
        correction_only=True,
    )
    torch.testing.assert_close(loss, clip + corr)
    assert m["sc_tok"] == float(d["response_mask"].sum())
