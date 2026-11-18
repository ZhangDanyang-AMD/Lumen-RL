import pytest
import torch

from lumenrl.models.deepseek_v41.megatron_moe import (
    moe_state_from_reference,
    reference_names,
)
from lumenrl.models.deepseek_v41.model import MoE
from tests.unit.models.deepseek_v41.test_model_vs_reference import _config


@pytest.fixture
def ffn():
    torch.manual_seed(0)
    moe = MoE(_config((600,)))
    with torch.no_grad():
        for p in moe.parameters():
            p.copy_(torch.randn_like(p))
    return moe.state_dict()


def test_each_ep_rank_takes_its_own_experts_with_gate_before_up(ffn):
    n = 8
    state = moe_state_from_reference(ffn, n_experts=n, ep_rank=1, ep_size=4)

    assert sorted(k for k in state if k.startswith("experts.")) == [
        "experts.linear_fc1.weight0", "experts.linear_fc1.weight1",
        "experts.linear_fc2.weight0", "experts.linear_fc2.weight1",
    ]
    for local, expert in ((0, 2), (1, 3)):
        fc1 = state[f"experts.linear_fc1.weight{local}"]
        gate, up = fc1.chunk(2)
        assert torch.equal(gate, ffn[f"experts.{expert}.w1.weight"])
        assert torch.equal(up, ffn[f"experts.{expert}.w3.weight"])
        assert torch.equal(state[f"experts.linear_fc2.weight{local}"], ffn[f"experts.{expert}.w2.weight"])


def test_router_and_shared_expert_are_on_every_rank(ffn):
    state = moe_state_from_reference(ffn, n_experts=8, ep_rank=3, ep_size=4)

    assert torch.equal(state["router.weight"], ffn["gate.weight"])
    assert torch.equal(state["router.expert_bias"], ffn["gate.bias"])
    assert torch.equal(state["shared_experts.linear_fc1.weight"],
                       torch.cat([ffn["shared_experts.w1.weight"], ffn["shared_experts.w3.weight"]]))
    assert torch.equal(state["shared_experts.linear_fc2.weight"], ffn["shared_experts.w2.weight"])


def test_experts_must_split_evenly(ffn):
    with pytest.raises(ValueError, match="do not split over EP 3"):
        moe_state_from_reference(ffn, n_experts=8, ep_rank=0, ep_size=3)


def test_te_extra_state_has_no_source_and_another_ranks_expert_is_an_error():
    assert reference_names("experts.linear_fc1._extra_state", ep_rank=0, n_local=2) == []
    assert reference_names("experts.linear_fc2.weight1", ep_rank=3, n_local=2) == ["experts.7.w2.weight"]
    with pytest.raises(KeyError, match="no reference counterpart"):
        reference_names("experts.linear_fc1.weight2", ep_rank=0, n_local=2)
