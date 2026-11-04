"""Score-centering rollout capture (P3): vLLM top-k head -> dense batch tensors."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lumenrl.algorithms.score_centering import densify_rollout_topk, validate_topk_rows
from lumenrl.core.protocol import DataProto
from lumenrl.engine.inference.vllm_ray_server import VLLMRayServer, extract_flat_logprobs


def _flat(rows: list[list[tuple[int, float]]]):
    """A FlatLogprobs-shaped object: one ``[(id, logprob), ...]`` list per position."""
    st, en, ids, lps = [], [], [], []
    for row in rows:
        st.append(len(ids))
        for t, lp in row:
            ids.append(t)
            lps.append(lp)
        en.append(len(ids))
    return SimpleNamespace(start_indices=st, end_indices=en, token_ids=ids, logprobs=lps)


def _rows(tokens, k, seed=0):
    """vLLM order per position: sampled first, then top-1..top-k (duplicates kept)."""
    rng = np.random.default_rng(seed)
    out, heads = [], []
    for t in tokens:
        head = rng.choice(np.arange(100, 200), size=k, replace=False)
        lp = np.sort(np.log(rng.dirichlet(np.ones(k + 4))[:k]))[::-1]
        out.append([(t, -0.5)] + list(zip(head.tolist(), lp.tolist())))
        heads.append((head, lp))
    return out, heads


def test_extract_flat_logprobs_splits_sampled_and_head() -> None:
    tokens = [5, 9, 7]
    rows, heads = _rows(tokens, k=4)
    lps, ids, head_lps = extract_flat_logprobs(_flat(rows), tokens, k=4)
    assert lps == [-0.5, -0.5, -0.5]
    assert ids.dtype == np.int32 and head_lps.dtype == np.float32
    assert ids.shape == head_lps.shape == (3, 4)
    np.testing.assert_array_equal(ids[1], heads[1][0])
    np.testing.assert_allclose(head_lps[2], heads[2][1], rtol=1e-6)


def test_extract_flat_logprobs_k0_returns_sampled_only() -> None:
    tokens = [3, 4]
    lps, ids, head_lps = extract_flat_logprobs(_flat([[(3, -1.0)], [(4, -2.0)]]), tokens, k=0)
    assert lps == [-1.0, -2.0] and ids is None and head_lps is None


def test_extract_flat_logprobs_rejects_misaligned_rows() -> None:
    rows, _ = _rows([5, 9], k=3)
    with pytest.raises(ValueError, match="sampled"):
        extract_flat_logprobs(_flat(rows), [5, 8], k=3)  # position 1 sampled 9, not 8
    with pytest.raises(ValueError, match="entries"):
        extract_flat_logprobs(_flat([rows[0], rows[1][:2]]), [5, 9], k=3)


class _FakeEngine:
    def __init__(self, logprobs):
        self.logprobs = logprobs
        self.seen = None

    async def generate(self, prompt, sp, request_id=None):
        self.seen = sp
        comp = SimpleNamespace(
            text="", token_ids=[5, 9, 7], logprobs=self.logprobs(sp)
        )
        yield SimpleNamespace(outputs=[comp], prompt_token_ids=[1, 2])


def _server(engine) -> VLLMRayServer:
    srv = VLLMRayServer.__new__(VLLMRayServer)
    srv.engine = engine
    return srv


def test_generate_requests_flat_topk_and_returns_head_arrays() -> None:
    rows, heads = _rows([5, 9, 7], k=4)
    eng = _FakeEngine(lambda sp: _flat(rows))
    out = asyncio.run(
        _server(eng).generate([1, 2], {"max_tokens": 3, "logprobs": 0, "topk_logprobs": 4})
    )
    assert eng.seen.logprobs == 4 and eng.seen.flat_logprobs is True
    assert eng.seen.detokenize is False  # 33 token decodes per position otherwise
    assert out["logprobs"] == [-0.5, -0.5, -0.5]
    np.testing.assert_array_equal(out["topk_ids"][0], heads[0][0])
    assert out["topk_logprobs"].shape == (3, 4)


def test_generate_without_topk_keeps_the_old_contract() -> None:
    eng = _FakeEngine(lambda sp: _flat([[(5, -1.0)], [(9, -2.0)], [(7, -3.0)]]))
    out = asyncio.run(_server(eng).generate([1, 2], {"max_tokens": 3, "logprobs": 0}))
    assert eng.seen.logprobs == 0 and eng.seen.detokenize is True
    assert out["logprobs"] == [-1.0, -2.0, -3.0]
    assert "topk_ids" not in out


# --------------------------------------------------------------------------- #


def _good_head(r=3, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(50, generator=g)[:k] for _ in range(r)]).int()
    lps = torch.log(torch.softmax(torch.randn(r, k + 3, generator=g), -1)[:, :k])
    return ids, lps


def test_validate_topk_rows_accepts_a_real_head() -> None:
    validate_topk_rows(*_good_head(), tag="sample 0")


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (lambda i, l: (i.index_fill(1, torch.tensor([1]), int(i[0, 0])), l), "duplicate"),
        (lambda i, l: (i, l.index_fill(1, torch.tensor([0]), float("nan"))), "finite"),
        (lambda i, l: (i, l.index_fill(1, torch.tensor([0]), 0.5)), "> 0"),
        (lambda i, l: (i, torch.full_like(l, -0.1)), "mass"),
        (lambda i, l: (i[:, :3], l), "shape"),
    ],
)
def test_validate_topk_rows_rejects_corrupt_rows(corrupt, match) -> None:
    ids, lps = corrupt(*_good_head())
    with pytest.raises(ValueError, match=match):
        validate_topk_rows(ids, lps, tag="sample 7")


def test_densify_aligns_with_rollout_logprob_positions() -> None:
    # Left-padded, S = 8: row 0 = 3 prompt + 4 response tokens; row 1 = 2 + 2.
    s, k = 8, 3
    seq_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]])
    plens = [3, 2]
    rows = [
        (torch.arange(12).view(4, 3).int(), -torch.arange(12).view(4, 3).float() - 1),
        (torch.arange(6).view(2, 3).int() + 50, torch.full((2, 3), -2.0)),
    ]
    ids, lps = densify_rollout_topk(rows, seq_mask, plens, k)
    assert ids.shape == lps.shape == (2, s - 1, k)
    assert ids.dtype == torch.int32 and lps.dtype == torch.float32
    # Row 0: start=1, response targets start at column start + plen - 1 = 3.
    torch.testing.assert_close(lps[0, 3:7], rows[0][1])
    assert torch.isneginf(lps[0, :3]).all()
    # Row 1: start=4, first response target at column 5.
    torch.testing.assert_close(ids[1, 5:7], rows[1][0])
    assert torch.isneginf(lps[1, :5]).all()
    assert (ids[1, :5] == 0).all()


def test_dense_head_tensors_split_with_their_rows() -> None:
    ids = torch.arange(4 * 5 * 2).view(4, 5, 2).int()
    proto = DataProto(tensors={"rollout_topk_ids": ids, "response_mask": torch.ones(4, 5)})
    parts = proto.split(2)
    torch.testing.assert_close(parts[1]["rollout_topk_ids"], ids[2:])
    torch.testing.assert_close(proto.select_idxs([3, 0])["rollout_topk_ids"], ids[[3, 0]])


# --------------------------------------------------------------------------- #
# Trainer wiring
# --------------------------------------------------------------------------- #

from lumenrl.algorithms.score_centering import vllm_engine_kwargs  # noqa: E402
from lumenrl.core.config import LumenRLConfig, ScoreCenteringConfig  # noqa: E402
from lumenrl.trainer.rl_trainer import RLTrainer  # noqa: E402


def test_vllm_engine_kwargs_use_processed_logprobs_and_raise_the_cap() -> None:
    assert vllm_engine_kwargs(ScoreCenteringConfig()) == {}
    kw = vllm_engine_kwargs(ScoreCenteringConfig(enabled=True, top_k=32))
    assert kw == {"logprobs_mode": "processed_logprobs", "max_logprobs": 32}
    assert vllm_engine_kwargs(ScoreCenteringConfig(enabled=True, top_k=8))["max_logprobs"] == 20


def _trainer(sc_enabled: bool, k: int = 3):
    cfg = LumenRLConfig()
    cfg.algorithm.name = "dapo"
    cfg.policy.max_response_length = 4
    cfg.policy.generation.vllm_cfg.calculate_log_probs = True
    cfg.algorithm.score_centering = ScoreCenteringConfig(enabled=sc_enabled, top_k=k)
    t = object.__new__(RLTrainer)
    t.config = cfg
    return t


def test_rollout_sampling_params_request_the_head_only_with_score_centering() -> None:
    assert "topk_logprobs" not in _trainer(False)._ray_sampling_params(True)
    sp = _trainer(True, k=3)._ray_sampling_params(True)
    assert sp["logprobs"] == 0 and sp["topk_logprobs"] == 3


class _Tok:
    pad_token_id = 0

    def __call__(self, p, **_):
        return {"input_ids": [1, 2]}


def _rollout_trainer(results, sc_enabled=True, k=3):
    t = _trainer(sc_enabled, k)
    t._tokenizer = _Tok()
    t._device = torch.device("cpu")
    t._ray_vllm_engine = SimpleNamespace(_sleeping=False, generate_tokens=lambda ids, sp: results)
    return t


def _result(n_resp: int, k: int, seed: int):
    ids, lps = _good_head(n_resp, k, seed)
    return {
        "prompt_token_ids": [1, 2], "token_ids": list(range(10, 10 + n_resp)),
        "logprobs": [-0.1] * n_resp, "topk_ids": ids.numpy(), "topk_logprobs": lps.numpy(),
    }


def test_ray_rollout_carries_the_head_per_row() -> None:
    results = [_result(3, 3, 0), _result(2, 3, 1)]
    *_, ragged = _rollout_trainer(results)._rollout_with_ray_vllm(["p"], 2)
    heads = ragged["rollout_topk"]
    assert len(heads) == 2 and heads[0][0].shape == (3, 3) and heads[1][1].shape == (2, 3)
    torch.testing.assert_close(heads[1][0], torch.as_tensor(results[1]["topk_ids"]))


def test_ray_rollout_rejects_a_corrupt_head() -> None:
    bad = _result(3, 3, 0)
    bad["topk_logprobs"] = np.full((3, 3), -0.1, np.float32)  # mass 2.7
    with pytest.raises(ValueError, match="mass"):
        _rollout_trainer([bad])._rollout_with_ray_vllm(["p"], 1)


def test_ray_rollout_rejects_a_head_of_the_wrong_length() -> None:
    bad = _result(3, 3, 0)
    bad["topk_ids"], bad["topk_logprobs"] = bad["topk_ids"][:2], bad["topk_logprobs"][:2]
    with pytest.raises(ValueError, match="rows"):
        _rollout_trainer([bad])._rollout_with_ray_vllm(["p"], 1)


def test_eval_rollouts_do_not_capture_the_head() -> None:
    res = _result(3, 3, 0)
    del res["topk_ids"], res["topk_logprobs"]
    t = _rollout_trainer([res])
    *_, ragged = t._rollout_with_ray_vllm(["p"], 1, sampling_params={"max_tokens": 4})
    assert "rollout_topk" not in ragged


def test_batch_gets_dense_head_and_ragged_payload_is_dropped() -> None:
    t = _trainer(True, k=3)
    t._device = torch.device("cpu")
    seq_mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    plens = [2, 3]
    response_mask = torch.zeros(2, 5)
    response_mask[0, 2:5] = 1
    response_mask[1, 2:5] = 1
    ragged = {"rollout_topk": [_good_head(3, 3, 0), _good_head(3, 3, 1)], "rollout_routing": [1, 2]}
    tensors: dict = {}

    t._attach_score_centering_head(tensors, ragged, seq_mask, plens, response_mask)

    assert set(ragged) == {"rollout_routing"}
    assert tensors["rollout_topk_ids"].shape == (2, 5, 3)
    assert torch.isfinite(tensors["rollout_topk_logprobs"][response_mask.bool()]).all()


def test_batch_rejects_a_head_that_misses_response_tokens() -> None:
    t = _trainer(True, k=3)
    t._device = torch.device("cpu")
    seq_mask = torch.ones(1, 6, dtype=torch.long)
    response_mask = torch.zeros(1, 5)
    response_mask[0, 1:5] = 1  # 4 response targets, but the head covers 3
    with pytest.raises(ValueError, match="response"):
        t._attach_score_centering_head(
            {}, {"rollout_topk": [_good_head(3, 3, 0)]}, seq_mask, [2], response_mask
        )


def test_batch_without_score_centering_is_untouched() -> None:
    t = _trainer(False)
    tensors: dict = {}
    t._attach_score_centering_head(tensors, {}, torch.ones(1, 4), [1], torch.ones(1, 3))
    assert tensors == {}


def test_dense_head_stays_on_the_host() -> None:
    """The controller only reorders / slices / ships these rows; keeping
    [B, S-1, k] off the controller GPU saves ~2.8 GB on long-run configs."""
    t = _trainer(True, k=3)
    t._device = torch.device("meta")  # any non-CPU device
    seq_mask = torch.ones(1, 6, dtype=torch.long)
    response_mask = torch.zeros(1, 5)
    response_mask[0, 1:4] = 1
    tensors: dict = {}
    t._attach_score_centering_head(
        tensors, {"rollout_topk": [_good_head(3, 3, 0)]}, seq_mask, [2], response_mask
    )
    assert tensors["rollout_topk_ids"].device.type == "cpu"
    assert tensors["rollout_topk_logprobs"].device.type == "cpu"
