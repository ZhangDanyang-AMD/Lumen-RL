"""DeepSeek-V4.1 on the native engine: how it is recognised, built and exported.

The model itself is checked against the release in ``tests/unit/models/deepseek_v41``;
these pin the glue the engine relies on.
"""

import ast
import inspect
from pathlib import Path

import pytest
import torch

from lumenrl.engine.training import model_specs
from lumenrl.engine.training.bridges import dsv41
from lumenrl.engine.training.model_registry import MODEL_REGISTRY

V41_RELEASE = {
    "model_type": "deepseek_v41",
    "architectures": ["DeepseekV41ForCausalLM"],
    "text_config": {"model_type": "deepseek_v41_text", "hidden_size": 5120, "num_attention_heads": 64,
                    "num_hidden_layers": 40, "n_routed_experts": 384},
    "vision_config": {},
}
DSV4 = {"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"], "n_routed_experts": 256}


def test_v41_resolves_to_its_own_spec_although_dsv4_detection_matches_it_too():
    assert MODEL_REGISTRY.resolve(V41_RELEASE, {}).name == "deepseek_v41"
    assert MODEL_REGISTRY.resolve(V41_RELEASE["text_config"], {}).name == "deepseek_v41"
    assert MODEL_REGISTRY.resolve(DSV4, {}).name == "deepseek_v4"


def test_the_engine_sees_the_language_model_config():
    spec = MODEL_REGISTRY.resolve(V41_RELEASE, {})
    text = spec.language_config(V41_RELEASE)
    assert text is V41_RELEASE["text_config"]
    assert spec.resolve_has_experts(text, {}) is True


def test_v41_builds_its_own_model_and_reads_a_microbatch_as_one_sequence():
    spec = MODEL_REGISTRY.resolve(V41_RELEASE, {})
    assert spec.build_model is dsv41.build_dsv41_model
    assert spec.caps.supports_dynamic_batch is False
    assert spec.caps.packed_stream_is_single_sequence is True
    assert spec.caps.requires_pipeline_forward is True
    assert spec.build_layer_spec is None and spec.sequence_alignment is None


def test_the_engine_applies_language_config_and_build_model():
    from lumenrl.engine.training import megatron_native_engine

    src = inspect.getsource(megatron_native_engine.MegatronNativeEngine.initialize)
    assert "self._spec.language_config(hf)" in src
    assert "self._spec.build_model(self, hf, tfcfg)" in src


def test_the_config_builder_takes_every_keyword_the_engine_passes():
    from lumenrl.engine.training import megatron_native_engine

    tree = ast.parse(Path(megatron_native_engine.__file__).read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "build_config"
    ]
    params = inspect.signature(dsv41.build_dsv41_config).parameters
    assert calls
    for call in calls:
        assert [kw.arg for kw in call.keywords if kw.arg not in params] == []


@pytest.mark.parametrize("parallel", [{"tp": 2}, {"cp": 2}, {"etp": 2}])
def test_only_expert_and_pipeline_parallelism(parallel):
    sizes = {"tp": 1, "pp": 1, "cp": 1, "ep": 8, "etp": 1, "sp": False, **parallel}
    with pytest.raises(NotImplementedError, match="replicates its non-expert weights"):
        dsv41.build_dsv41_config(V41_RELEASE["text_config"], {}, **sizes)


def test_stages_split_the_layers_evenly():
    assert [dsv41.stage_layers(40, 2, r) for r in range(2)] == [(0, 20), (20, 40)]


def test_parameters_go_out_under_their_checkpoint_names():
    t = torch.arange(12.0).view(6, 2)
    assert dsv41.checkpoint_names("stage.layers.3.attn.wq_a.weight", t) == [("layers.3.attn.wq_a.weight", t)]
    assert dsv41.checkpoint_names("module.stage.head.weight", t) == [("head.weight", t)]
    (name, got), = dsv41.checkpoint_names("stage.layers.3.ffn.layer.router.weight", t)
    assert name == "layers.3.ffn.gate.weight" and got is t
    (n1, w1), (n3, w3) = dsv41.checkpoint_names("stage.layers.3.ffn.layer.shared_experts.linear_fc1.weight", t)
    assert (n1, n3) == ("layers.3.ffn.shared_experts.w1.weight", "layers.3.ffn.shared_experts.w3.weight")
    assert torch.equal(w1, t[:3]) and torch.equal(w3, t[3:])
    assert dsv41.checkpoint_names("stage.layers.3.ffn.layer.shared_experts.linear_fc2.weight", t)[0][0] == (
        "layers.3.ffn.shared_experts.w2.weight"
    )
    with pytest.raises(KeyError, match="no checkpoint name"):
        dsv41.checkpoint_names("stage.layers.3.ffn.layer.experts.linear_fc1.weight0", t)


def test_only_trained_parameters_are_exported(monkeypatch):
    from megatron.core import parallel_state as mpu

    module = torch.nn.Module()
    module.stage = torch.nn.Module()
    module.stage.norm = torch.nn.Linear(2, 2, bias=False)
    module.stage.frozen = torch.nn.Linear(2, 2, bias=False)
    module.stage.frozen.weight.requires_grad_(False)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: 1)

    class Engine:
        pass

    engine = Engine()
    engine.module = module
    assert [n for n, _ in dsv41.export_dsv41(engine)] == ["norm.weight"]


def test_the_spec_is_registered_from_the_bridge():
    assert model_specs.DSV41.export_weights is dsv41.export_dsv41


def test_two_wrapped_stages_hand_the_stream_over_as_the_schedule_does():
    """What Megatron's schedule does between stages: the first stage's output
    becomes the second's input through ``set_input_tensor``, and the forward is
    called with the engine's keywords (whose packing V4.1 ignores)."""
    from types import SimpleNamespace

    from lumenrl.models.deepseek_v41.model import DSV41Model, DSV41Stage
    from tests.unit.models.deepseek_v41.test_model_vs_reference import (
        SEQLEN,
        TINY,
        _config,
        _init,
        _token_map,
    )
    from tests.unit.models.deepseek_v41.test_stage import _rows

    whole = DSV41Model(_config(_rows()), token_map=torch.tensor(_token_map()))
    with torch.no_grad():
        _init(whole, 0)
    cls, config = dsv41.megatron_model_class(), SimpleNamespace()
    first = cls(config, DSV41Stage(whole.cfg, 0, 4, token_map=whole.hasher.token_map))
    last = cls(config, DSV41Stage(whole.cfg, 4, 8))
    for m in (first, last):
        m.stage.load_state_dict({k: v for k, v in whole.state_dict().items() if k in m.stage.state_dict()})
    ids = torch.randint(0, TINY["vocab_size"], (1, SEQLEN), generator=torch.Generator().manual_seed(1))
    kwargs = {"position_ids": None, "attention_mask": None, "packed_seq_params": object()}

    with torch.no_grad():
        stream = first(input_ids=ids, **kwargs)
        last.set_input_tensor([stream])
        hidden = last(input_ids=ids, **kwargs)  # the engine applies the head (output_head)
        got = torch.nn.functional.linear(hidden, dsv41.output_head(SimpleNamespace(module=last)))
        want = whole(ids)

    assert (first.pre_process, first.post_process, last.pre_process, last.post_process) == (True, False, False, True)
    assert torch.equal(got, want)
