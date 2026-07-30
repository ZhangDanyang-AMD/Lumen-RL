"""Fused-MoE weight sync: routing math and the silent-drop guard.

CPU-only, no vLLM and no model needed. The fake FusedMoE mirrors the parts of
``vllm.model_executor.layers.fused_moe.layer.FusedMoE.weight_loader`` this code
relies on: a 3D ``loaded_weight`` selects the full-load branch, ``w1``/``w3``
address the first/second half of ``w13_weight`` along the intermediate dim, and
a non-local expert id makes the loader return False instead of writing.

Run: python -m lumenrl.tests.test_moe_weight_sync
"""

import os

import torch
from torch import nn

from lumenrl.engine.inference.vllm_moe_weight_sync import (
    FusedMoEWeightRouter,
    assert_weight_sync_coverage,
)

E, I, H = 4, 6, 8
N_LAYERS = 2


class _ParallelConfig:
    def __init__(self, ep_size: int):
        self.ep_size = ep_size
        self.tp_size = 1


class _MoEConfig:
    def __init__(self, ep_size: int, is_act_and_mul: bool):
        self.moe_parallel_config = _ParallelConfig(ep_size)
        self.is_act_and_mul = is_act_and_mul


class FakeFusedMoE(nn.Module):
    """Stand-in for vLLM's FusedMoE with its real weight_loader semantics."""

    def __init__(self, ep_size: int = 1, is_act_and_mul: bool = True, local_experts=None):
        super().__init__()
        out13 = 2 * I if is_act_and_mul else I
        self.w13_weight = nn.Parameter(torch.zeros(E, out13, H), requires_grad=False)
        self.w2_weight = nn.Parameter(torch.zeros(E, H, I), requires_grad=False)
        self.moe_config = _MoEConfig(ep_size, is_act_and_mul)
        self.local_experts = local_experts
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def weight_loader(
        self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
    ):
        self.calls.append((shard_id, tuple(loaded_weight.shape)))
        if self.local_experts is not None and expert_id not in self.local_experts:
            return False if return_success else None

        full_load = loaded_weight.ndim == 3
        expert_data = param.data if full_load else param.data[expert_id]
        shard_dim = (1 if shard_id == "w2" else 0) + (1 if full_load else 0)

        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        else:
            size = expert_data.shape[shard_dim] // (2 if self.moe_config.is_act_and_mul else 1)
            start = 0 if shard_id == "w1" else size
            expert_data.narrow(shard_dim, start, size).copy_(loaded_weight)
        return True if return_success else None


class FakeLayer(nn.Module):
    def __init__(self, **moe_kwargs):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(H, E, bias=False)
        self.mlp.experts = FakeFusedMoE(**moe_kwargs)
        self.mlp.add_module("gate", self.mlp.gate)
        self.mlp.add_module("experts", self.mlp.experts)


class FakeModel(nn.Module):
    """Mimics vLLM's load_weights: unknown names are dropped without raising."""

    def __init__(self, **moe_kwargs):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer(**moe_kwargs) for _ in range(N_LAYERS)])
        self.model.add_module("layers", self.model.layers)
        self.lm_head = nn.Linear(H, 16, bias=False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name not in params:
                continue
            params[name].data.copy_(tensor)
            loaded.add(name)
        return loaded


def _fused_payload(model, scale=1.0):
    """What a transformers-5.x state_dict() sends for this model."""
    weights = []
    for idx in range(N_LAYERS):
        p = f"model.layers.{idx}.mlp"
        weights.append((f"{p}.experts.gate_up_proj", torch.randn(E, 2 * I, H) * scale))
        weights.append((f"{p}.experts.down_proj", torch.randn(E, H, I) * scale))
        weights.append((f"{p}.gate.weight", torch.randn(E, H) * scale))
    weights.append(("lm_head.weight", torch.randn(16, H) * scale))
    return weights


def test_gate_up_splits_into_w13_halves():
    torch.manual_seed(0)
    model = FakeModel()
    gate_up = torch.randn(E, 2 * I, H)
    router = FusedMoEWeightRouter(model)
    passthrough, loaded = router.route([("model.layers.0.mlp.experts.gate_up_proj", gate_up)])

    assert passthrough == []
    assert loaded == {"model.layers.0.mlp.experts.w13_weight"}
    w13 = model.model.layers[0].mlp.experts.w13_weight.data
    assert torch.equal(w13, gate_up), "fused gate_up must land verbatim in w13"
    assert torch.equal(w13[:, :I], gate_up[:, :I])  # w1 = gate
    assert torch.equal(w13[:, I:], gate_up[:, I:])  # w3 = up
    assert [c[0] for c in model.model.layers[0].mlp.experts.calls] == ["w1", "w3"]


def test_down_proj_lands_in_w2_whole():
    model = FakeModel()
    down = torch.randn(E, H, I)
    _, loaded = FusedMoEWeightRouter(model).route(
        [("model.layers.1.mlp.experts.down_proj", down)]
    )
    assert loaded == {"model.layers.1.mlp.experts.w2_weight"}
    assert torch.equal(model.model.layers[1].mlp.experts.w2_weight.data, down)


def test_full_round_trip_touches_every_parameter():
    """The regression this module exists for: nothing may be silently dropped."""
    torch.manual_seed(1)
    model = FakeModel()
    router = FusedMoEWeightRouter(model)
    passthrough, loaded = router.route(_fused_payload(model))
    loaded |= model.load_weights(passthrough)
    assert_weight_sync_coverage(model, loaded, context="test")

    # Without the router the fused names match nothing and 2/3 of the model
    # keeps its old values -- exactly the bug that hid for 54 steps.
    bare = FakeModel()
    bare_loaded = bare.load_weights(_fused_payload(bare))
    try:
        assert_weight_sync_coverage(bare, bare_loaded, context="test")
    except RuntimeError as exc:
        assert "w13_weight" in str(exc) and "untouched" in str(exc)
    else:
        raise AssertionError("coverage check missed the dropped expert tensors")


def test_non_expert_weights_pass_through_untouched():
    model = FakeModel()
    payload = [("model.layers.0.mlp.gate.weight", torch.randn(E, H)),
               ("lm_head.weight", torch.randn(16, H))]
    passthrough, loaded = FusedMoEWeightRouter(model).route(payload)
    assert loaded == set()
    assert [n for n, _ in passthrough] == [n for n, _ in payload]


def test_router_is_inactive_without_fused_moe():
    dense = nn.Sequential(nn.Linear(H, H))
    router = FusedMoEWeightRouter(dense)
    assert not router.active
    payload = [("0.weight", torch.randn(H, H))]
    passthrough, loaded = router.route(payload)
    assert passthrough == payload and loaded == set()


def test_non_3d_fused_tensor_is_loud():
    model = FakeModel()
    router = FusedMoEWeightRouter(model)
    try:
        router.route([("model.layers.0.mlp.experts.gate_up_proj", torch.randn(2 * I, H))])
    except RuntimeError as exc:
        assert "must be 3D" in str(exc)
    else:
        raise AssertionError("a 2D fused tensor must not be loaded blindly")


def test_expert_parallel_falls_back_to_per_expert():
    model = FakeModel(ep_size=2, local_experts={0, 1})
    gate_up = torch.randn(E, 2 * I, H)
    _, loaded = FusedMoEWeightRouter(model).route(
        [("model.layers.0.mlp.experts.gate_up_proj", gate_up)]
    )
    assert loaded == {"model.layers.0.mlp.experts.w13_weight"}
    experts = model.model.layers[0].mlp.experts
    assert len(experts.calls) == 2 * E, "one call per (expert, shard) under EP"
    assert all(len(shape) == 2 for _, shape in experts.calls)
    assert torch.equal(experts.w13_weight.data[:2], gate_up[:2])
    assert torch.equal(experts.w13_weight.data[2:], torch.zeros(E - 2, 2 * I, H))


def test_non_gated_experts_load_as_one_shard():
    model = FakeModel(is_act_and_mul=False)
    gate_up = torch.randn(E, I, H)
    FusedMoEWeightRouter(model).route(
        [("model.layers.0.mlp.experts.gate_up_proj", gate_up)]
    )
    assert torch.equal(model.model.layers[0].mlp.experts.w13_weight.data, gate_up)


def test_coverage_ignores_quantization_artifacts():
    model = FakeModel()
    model.register_parameter("lm_head_weight_scale", nn.Parameter(torch.ones(1)))
    router = FusedMoEWeightRouter(model)
    passthrough, loaded = router.route(_fused_payload(model))
    loaded |= model.load_weights(passthrough)
    assert_weight_sync_coverage(model, loaded, context="test")


def test_verify_accepts_a_correct_load_and_rejects_a_corrupt_one():
    torch.manual_seed(2)
    os.environ["LUMENRL_WEIGHT_SYNC_VERIFY"] = "1"
    try:
        payload = lambda: [  # noqa: E731 - one-liner fixture
            ("model.layers.0.mlp.experts.gate_up_proj", torch.randn(E, 2 * I, H)),
            ("model.layers.0.mlp.experts.down_proj", torch.randn(E, H, I)),
        ]
        FusedMoEWeightRouter(FakeModel()).route(payload())

        class DroppingMoE(FakeFusedMoE):
            def weight_loader(self, param, loaded_weight, weight_name, shard_id,
                              expert_id, return_success=False):
                if shard_id == "w3":  # simulate a shard that never lands
                    return True if return_success else None
                return super().weight_loader(
                    param, loaded_weight, weight_name, shard_id, expert_id, return_success
                )

        model = FakeModel()
        model.model.layers[0].mlp.experts.__class__ = DroppingMoE
        try:
            FusedMoEWeightRouter(model).route(payload())
        except RuntimeError as exc:
            assert "verify failed" in str(exc) and "w3" in str(exc)
        else:
            raise AssertionError("verify missed a shard that was never written")
    finally:
        os.environ.pop("LUMENRL_WEIGHT_SYNC_VERIFY", None)


def test_coverage_modes_are_configurable():
    model = FakeModel()
    previous = os.environ.get("LUMENRL_WEIGHT_SYNC_CHECK")
    try:
        for mode in ("warn", "off", "OFF"):
            os.environ["LUMENRL_WEIGHT_SYNC_CHECK"] = mode
            assert_weight_sync_coverage(model, set(), context="test")
    finally:
        if previous is None:
            os.environ.pop("LUMENRL_WEIGHT_SYNC_CHECK", None)
        else:
            os.environ["LUMENRL_WEIGHT_SYNC_CHECK"] = previous


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("all fused-MoE weight sync tests passed")
