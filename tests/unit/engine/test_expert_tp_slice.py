"""``weight_sync.expert_tp_slice``: routed experts sent as the rollout TP rank's slice."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lumenrl.engine.inference.vllm_moe_weight_sync import TpSlicedExpertLoader
from lumenrl.engine.training.bridges.dsv4 import mark_tp_sliced_experts, split_tp_slice_suffix
from lumenrl.engine.training.megatron_native_engine import slice_expert_for_tp, tp_sliced_expert_shape

INTER, HIDDEN, TP = 8, 6, 4


def test_an_fc1_slice_is_what_vllm_cuts_from_gate_and_up():
    fc1 = torch.arange(2 * INTER * HIDDEN, dtype=torch.float32).view(2 * INTER, HIDDEN)
    gate, up = fc1.chunk(2, dim=0)
    per = INTER // TP

    gate_slice, up_slice = slice_expert_for_tp(fc1, "1", 2, TP).chunk(2, dim=0)

    assert torch.equal(gate_slice, gate.narrow(0, 2 * per, per))
    assert torch.equal(up_slice, up.narrow(0, 2 * per, per))


def test_an_fc2_slice_is_the_ranks_columns():
    fc2 = torch.arange(HIDDEN * INTER, dtype=torch.float32).view(HIDDEN, INTER)
    per = INTER // TP

    assert torch.equal(slice_expert_for_tp(fc2, "2", 3, TP), fc2.narrow(1, 3 * per, per))


def test_the_planned_shape_matches_the_slice():
    assert tp_sliced_expert_shape((2 * INTER, HIDDEN), "1", TP) == (2 * INTER // TP, HIDDEN)
    assert tp_sliced_expert_shape((HIDDEN, INTER), "2", TP) == (HIDDEN, INTER // TP)
    with pytest.raises(ValueError):
        tp_sliced_expert_shape((2 * 6, HIDDEN), "1", TP)


def test_only_routed_experts_are_marked_and_only_once_the_slice_applies():
    named = [
        ("layers.0.attn.wq_a.weight", 1),
        ("layers.0.ffn.experts.3.w1.weight", 2),
        ("layers.0.ffn.shared_experts.w1.weight", 3),
        ("layers.1.ffn.experts.0.w2.weight", 4),
    ]
    applied = iter([None, None, (1, 4), (1, 4)])

    out = [n for n, _ in mark_tp_sliced_experts(named, lambda: next(applied))]

    assert out == [
        "layers.0.attn.wq_a.weight",
        "layers.0.ffn.experts.3.w1.weight",
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.1.ffn.experts.0.w2.weight@tp1of4",
    ]
    assert split_tp_slice_suffix(out[-1]) == ("layers.1.ffn.experts.0.w2.weight", 1, 4)
    assert split_tp_slice_suffix(out[0]) is None


class _Experts(torch.nn.Module):
    def __init__(self, tp_rank: int, tp_size: int, use_ep: bool = False) -> None:
        super().__init__()
        per = INTER // tp_size
        self.w13_weight = torch.nn.Parameter(torch.zeros(4, 2 * per, HIDDEN), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(torch.zeros(4, HIDDEN, per), requires_grad=False)
        self.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(
            tp_rank=tp_rank, tp_size=tp_size, use_ep=use_ep,
        ))


def _model(**kw) -> torch.nn.Module:
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.ffn = torch.nn.Module()
    layer.ffn.experts = _Experts(**kw)
    model.model.layers = torch.nn.ModuleList([layer])
    return model


def test_marked_slices_land_in_the_experts_and_count_as_loaded():
    model = _model(tp_rank=1, tp_size=TP)
    per = INTER // TP
    gate, up, down = torch.full((per, HIDDEN), 1.0), torch.full((per, HIDDEN), 2.0), torch.full((HIDDEN, per), 3.0)
    loaded: set[str] = set()

    rest = TpSlicedExpertLoader(model).take([
        ("layers.0.ffn.experts.2.w1.weight@tp1of4", gate),
        ("layers.0.ffn.experts.2.w3.weight@tp1of4", up),
        ("layers.0.ffn.experts.2.w2.weight@tp1of4", down),
        ("layers.0.attn.wq_a.weight", torch.ones(1)),
    ], loaded)

    experts = model.model.layers[0].ffn.experts
    assert torch.equal(experts.w13_weight[2, :per], gate)
    assert torch.equal(experts.w13_weight[2, per:], up)
    assert torch.equal(experts.w2_weight[2], down)
    assert not experts.w13_weight[1].any()
    assert [n for n, _ in rest] == ["layers.0.attn.wq_a.weight"]
    assert loaded == {"model.layers.0.ffn.experts.w13_weight", "model.layers.0.ffn.experts.w2_weight"}
    assert loaded <= dict(model.named_parameters()).keys()


@pytest.mark.parametrize("kw, name", [
    ({"tp_rank": 0, "tp_size": TP}, "layers.0.ffn.experts.2.w1.weight@tp1of4"),
    ({"tp_rank": 1, "tp_size": TP, "use_ep": True}, "layers.0.ffn.experts.2.w1.weight@tp1of4"),
    ({"tp_rank": 1, "tp_size": TP}, "layers.5.ffn.experts.2.w1.weight@tp1of4"),
])
def test_a_slice_this_rank_cannot_place_is_an_error(kw, name):
    with pytest.raises((KeyError, ValueError)):
        TpSlicedExpertLoader(_model(**kw)).take([(name, torch.ones(INTER // TP, HIDDEN))], set())
