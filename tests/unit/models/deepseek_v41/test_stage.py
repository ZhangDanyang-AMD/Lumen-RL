"""Pipeline stages and per-layer recomputation of the V4.1 backbone, on the tiny 8-layer config.

Layers 0-3 are the encoder part (window-only, then a KV/index source at 2 and a reuse layer),
4-7 the decoder part (the candidate source at 4, a reindex layer at 6), so [0, 4) + [4, 8) is
the split the real model gets at layer 20.
"""

import pytest
import torch

from lumenrl.models.deepseek_v41 import model as model_module
from lumenrl.models.deepseek_v41 import ops
from lumenrl.models.deepseek_v41.engram import EngramLayout
from lumenrl.models.deepseek_v41.model import (
    DSV41Model,
    DSV41Stage,
    pack_stream,
    unpack_stream,
)
from tests.unit.models.deepseek_v41.test_model_vs_reference import (
    SEQLEN,
    TINY,
    _config,
    _init,
    _token_map,
)


def _rows():
    layout = EngramLayout.from_config(_config((1,)))
    return (sum(p for per_ngram in layout.primes[0] for p in per_ngram),)


@pytest.fixture
def model():
    m = DSV41Model(_config(_rows()), token_map=torch.tensor(_token_map()))
    with torch.no_grad():
        _init(m, 0)
    return m


def _ids(seed):
    return torch.randint(0, TINY["vocab_size"], (2, SEQLEN), generator=torch.Generator().manual_seed(seed))


def _grads(module):
    return {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}


def _backward(logits, seed):
    w = torch.randn(logits.shape, generator=torch.Generator().manual_seed(seed))
    (logits * w).sum().backward()


def test_pack_round_trips_bit_for_bit_and_so_does_its_gradient():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(2, 5, 4, 8, generator=g).to(torch.bfloat16).requires_grad_()
    pre = (torch.rand(2, 5, 4, generator=g) * 1e-3).requires_grad_()  # far below bf16 resolution of 1
    a = torch.randn(h.shape, generator=g)
    b = torch.randn(pre.shape, generator=g)

    packed = pack_stream(h, pre)
    h2, pre2 = unpack_stream(packed.detach().requires_grad_())
    assert packed.shape == (2, 5, 4, 10) and packed.dtype == torch.bfloat16
    assert torch.equal(h2, h) and torch.equal(pre2, pre)

    h3, pre3 = unpack_stream(pack_stream(h, pre))
    ((h3.float() * a).sum() + (pre3 * b).sum()).backward()
    want_h, want_pre = (a.to(torch.bfloat16), b)
    assert torch.equal(h.grad, want_h) and torch.equal(pre.grad, want_pre)


def test_two_stages_compute_what_the_whole_model_does(model):
    cfg, ids = model.cfg, _ids(1)
    first = DSV41Stage(cfg, 0, 4, token_map=model.hasher.token_map)
    last = DSV41Stage(cfg, 4, 8)
    first.load_state_dict({k: v for k, v in model.state_dict().items() if k in first.state_dict()})
    last.load_state_dict({k: v for k, v in model.state_dict().items() if k in last.state_dict()})

    want = model(ids)
    _backward(want, 2)
    stream = first(ids)
    received = stream.detach().requires_grad_()
    got = last(ids, received)
    _backward(got, 2)
    stream.backward(received.grad)

    assert torch.equal(got, want)
    whole, split = _grads(model), {**_grads(first), **_grads(last)}
    assert whole.keys() == split.keys()
    for name, grad in whole.items():
        assert torch.equal(split[name], grad), name


def test_recomputing_each_layer_gives_the_same_gradients(model):
    ids = _ids(3)
    again = DSV41Model(model.cfg, token_map=model.hasher.token_map, recompute=True)
    again.load_state_dict(model.state_dict())

    _backward(model(ids), 4)
    _backward(again(ids), 4)

    plain, recomputed = _grads(model), _grads(again)
    assert plain.keys() == recomputed.keys()
    for name, grad in plain.items():
        assert torch.equal(recomputed[name], grad), name


def test_chunking_the_queries_changes_no_logit(model, monkeypatch):
    """Gradients only in fp32 summation order: the kv gradient sums over the chunks one at a
    time (rounded to bf16 per chunk instead, layers.4.hc_attn_base was 19% off here)."""
    ids = _ids(6)
    again = DSV41Model(model.cfg, token_map=model.hasher.token_map, recompute=True)
    again.load_state_dict(model.state_dict())

    want = model(ids)
    _backward(want, 7)
    monkeypatch.setattr(model_module, "INDEXER_QUERY_CHUNK", 5)
    monkeypatch.setattr(ops, "SPARSE_ATTENTION_QUERY_CHUNK", 5)
    got = again(ids)
    _backward(got, 7)

    assert SEQLEN > 5 and torch.equal(got, want)
    whole, chunked = _grads(model), _grads(again)
    assert whole.keys() == chunked.keys()
    for name, grad in whole.items():
        err = (chunked[name].float() - grad.float()).norm() / grad.float().norm().clamp_min(1e-12)
        assert err < 1e-3, (name, float(err))


def test_autograd_keeps_nothing_of_the_indexer_scores(model):
    """Its top-k is discrete, so nothing it computes can get a gradient; at a ratio-1 layer the
    score of a whole sequence is [s, heads, s]."""
    shapes = []
    cfg = model.cfg
    # the indexer's q [b, s, heads, head_dim] and score [b, s, heads, positions]
    last = {cfg.index_head_dim, SEQLEN, SEQLEN // 2}
    with torch.autograd.graph.saved_tensors_hooks(lambda t: shapes.append(tuple(t.shape)) or t, lambda t: t):
        model(_ids(8))

    indexer = [s for s in shapes if len(s) == 4 and s[1] == SEQLEN and s[2] == cfg.index_n_heads and s[3] in last]
    assert shapes and not indexer


def test_a_stage_holds_only_its_layers_under_their_model_numbers(model):
    last = DSV41Stage(model.cfg, 4, 8)
    names = set(last.state_dict())

    assert {n.split(".")[1] for n in names if n.startswith("layers.")} == {"4", "5", "6", "7"}
    assert "head.weight" in names and "embed.weight" not in names and last.hasher is None


def test_a_split_inside_a_sharing_group_is_refused(model):
    with pytest.raises(ValueError, match="layer 3 takes its kv_source from layer 2"):
        DSV41Stage(model.cfg, 3, 8)


def test_a_later_stage_needs_the_stream(model):
    with pytest.raises(ValueError, match="needs the previous stage's stream"):
        DSV41Stage(model.cfg, 4, 8)(_ids(5))


def test_release_quantization_takes_fp8_inputs_exactly_where_the_release_stores_fp8_or_fp4(model):
    # what the release stores in FP8 / FP4, but wo_a, whose input attn_out_fp8 quantizes
    quantized = ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "indexer.wq_b", "experts.", "engram.wkv")
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    act = {n for n, m in linears.items() if getattr(m, "act_fp8", False)}

    assert act == {n for n in linears if any(k in n for k in quantized) and "compressor" not in n}
    assert any("shared_experts" in n for n in act) and any(".experts.0.w2" in n for n in act)
