import pytest
import torch

import lumenrl.engine.training.megatron_lumen_dsv4_engine as dsv4_engine
from lumenrl.engine.training.dsv4_megatron_bridge import DSV4Dims
from lumenrl.engine.training.megatron_lumen_dsv4_engine import (
    MegatronLumenDSV4Engine,
    _StreamingGatheredParamMapping,
    _dsv4_router_kwargs,
    _named_export_tensors,
)


def _parameter(values, *, tensor_parallel=False, partition_dim=0):
    param = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
    param.tensor_model_parallel = tensor_parallel
    param.partition_dim = partition_dim
    param.partition_stride = 1
    return param


def test_noaux_tc_router_uses_checkpoint_scoring_and_normalization():
    kwargs = _dsv4_router_kwargs(
        {
            "topk_method": "noaux_tc",
            "scoring_func": "sqrtsoftplus",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
        },
        {},
    )

    assert kwargs == {
        "moe_router_load_balancing_type": "none",
        "moe_router_score_function": "sqrtsoftplus",
        "moe_router_dtype": "fp32",
        "moe_router_topk_scaling_factor": 1.5,
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 0.0,
    }


def test_noaux_tc_router_rejects_conflicting_scaling_override():
    with pytest.raises(ValueError, match="moe_router_topk_scaling_factor"):
        _dsv4_router_kwargs(
            {
                "topk_method": "noaux_tc",
                "scoring_func": "sqrtsoftplus",
                "norm_topk_prob": True,
                "routed_scaling_factor": 1.5,
            },
            {"moe_router_topk_scaling_factor": 2.0},
        )


def test_precision_aware_optimizer_keeps_fp32_master_on_cpu():
    precision_kwargs = getattr(
        dsv4_engine, "_optimizer_precision_kwargs", lambda _config: {}
    )
    assert precision_kwargs(
        {"use_precision_aware_optimizer": True}
    ) == {
        "use_precision_aware_optimizer": True,
        "main_grads_dtype": torch.float32,
        "main_params_dtype": torch.float32,
        "exp_avg_dtype": torch.float32,
        "exp_avg_sq_dtype": torch.float32,
    }
    assert precision_kwargs({}) == {
        "use_precision_aware_optimizer": False,
    }


def test_hash_tables_are_registered_as_checkpoint_static_buffers():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.router = torch.nn.Module()

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])
    state = {
        f"decoder.layers.{layer}.mlp.router.tid2eid": torch.arange(
            16, dtype=torch.int32
        ).view(8, 2)
        for layer in range(3)
    }
    register_buffers = getattr(
        dsv4_engine, "_register_checkpoint_static_buffers", lambda *_args: None
    )

    register_buffers(model, state)

    incompat = model.load_state_dict(state, strict=False)
    assert incompat.unexpected_keys == []
    assert dict(model.named_buffers()).keys() >= state.keys()


@pytest.mark.parametrize(
    ("checkpoint_config", "error"),
    [
        (
            {
                "topk_method": "noaux_tc",
                "scoring_func": "sigmoid",
                "norm_topk_prob": True,
            },
            "scoring_func",
        ),
        (
            {
                "topk_method": "noaux_tc",
                "scoring_func": "sqrtsoftplus",
                "norm_topk_prob": False,
            },
            "norm_topk_prob",
        ),
        (
            {
                "topk_method": "group_limited_greedy",
                "scoring_func": "sqrtsoftplus",
                "norm_topk_prob": True,
            },
            "topk_method",
        ),
        (
            {
                "scoring_func": "sqrtsoftplus",
                "norm_topk_prob": True,
            },
            "topk_method",
        ),
    ],
)
def test_noaux_tc_router_rejects_unsupported_checkpoint_semantics(
    checkpoint_config,
    error,
):
    with pytest.raises(ValueError, match=error):
        _dsv4_router_kwargs(checkpoint_config, {})


def test_export_tensors_exclude_frozen_router_buffers():
    module = torch.nn.Module()
    module.register_parameter("weight", _parameter([1, 2]))
    router = torch.nn.Module()
    router.register_buffer("expert_bias", torch.tensor([0.25, -0.5]))
    router.register_buffer("local_tokens_per_expert", torch.ones(2), persistent=False)
    module.add_module("router", router)

    exported = dict(_named_export_tensors(module))

    assert set(exported) == {"weight"}


def test_streaming_metadata_mapping_expands_tp_shape_without_allocating(monkeypatch):
    param = _parameter([[1, 2, 3], [4, 5, 6]], tensor_parallel=True)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("metadata lookup must not communicate")
        ),
    )

    mapping = _StreamingGatheredParamMapping(
        [("module.decoder.layers.0.self_attention.wq_a.weight", param)],
        tp_size=2,
        tp_group=object(),
        ep_size=1,
        ep_group=None,
        etp_size=1,
        etp_group=None,
        num_experts=0,
        metadata=True,
    )

    tensor = mapping["decoder.layers.0.self_attention.wq_a.weight"]

    assert tensor.device.type == "meta"
    assert tensor.shape == (4, 3)


def test_streaming_mapping_prefers_recorded_physical_partition_metadata():
    sharded = _parameter([[1, 2, 3], [4, 5, 6]], tensor_parallel=False)
    sharded._lumen_weight_partition_dim = 0
    replicated = _parameter([[1, 2, 3], [4, 5, 6]], tensor_parallel=True)
    replicated._lumen_weight_partition_dim = None
    mapping = _StreamingGatheredParamMapping(
        [("sharded.weight", sharded), ("replicated.weight", replicated)],
        tp_size=4,
        tp_group=object(),
        ep_size=1,
        ep_group=None,
        etp_size=1,
        etp_group=None,
        num_experts=0,
        metadata=True,
    )

    assert mapping["sharded.weight"].shape == (8, 3)
    assert mapping["replicated.weight"].shape == (2, 3)


def test_streaming_mapping_gathers_only_requested_global_expert(monkeypatch):
    local_zero = _parameter([[1, 2], [3, 4]])
    local_one = _parameter([[10, 20], [30, 40]])
    calls = []

    def fake_all_gather(outputs, tensor, group):
        calls.append((tensor.clone(), group))
        for rank, output in enumerate(outputs):
            output.copy_(tensor + rank * 100)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    ep_group = object()
    mapping = _StreamingGatheredParamMapping(
        [
            ("decoder.layers.0.mlp.experts.linear_fc1.weight0", local_zero),
            ("decoder.layers.0.mlp.experts.linear_fc1.weight1", local_one),
        ],
        tp_size=1,
        tp_group=None,
        ep_size=2,
        ep_group=ep_group,
        etp_size=1,
        etp_group=None,
        num_experts=4,
    )

    result = mapping["decoder.layers.0.mlp.experts.linear_fc1.weight3"]

    torch.testing.assert_close(result, local_one + 100)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], local_one)
    assert calls[0][1] is ep_group


def test_streaming_mapping_makes_expert_gather_buffers_contiguous(monkeypatch):
    local = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1)
    )

    def fake_all_gather(outputs, tensor, group):
        assert tensor.is_contiguous()
        assert all(output.is_contiguous() for output in outputs)
        for rank, output in enumerate(outputs):
            output.copy_(tensor + rank * 100)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    mapping = _StreamingGatheredParamMapping(
        [("decoder.layers.0.mlp.experts.linear_fc1.weight0", local)],
        tp_size=1,
        tp_group=None,
        ep_size=2,
        ep_group=object(),
        etp_size=1,
        etp_group=None,
        num_experts=2,
    )

    result = mapping["decoder.layers.0.mlp.experts.linear_fc1.weight1"]

    torch.testing.assert_close(result, local + 100)


def test_get_per_tensor_param_defers_tp_gather_until_iteration(monkeypatch):
    pytest.importorskip("megatron")
    from megatron.core import parallel_state as mpu

    param = _parameter([[1, 2, 3], [4, 5, 6]], tensor_parallel=True)
    calls = []

    def fake_all_gather(outputs, tensor, group):
        calls.append(group)
        for rank, output in enumerate(outputs):
            output.copy_(tensor + rank * 10)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: "tp")
    monkeypatch.setattr(mpu, "get_expert_tensor_parallel_world_size", lambda: 1)

    engine = object.__new__(MegatronLumenDSV4Engine)
    engine.module = type(
        "Module",
        (),
        {
            "named_parameters": lambda self: iter(
                [
                    ("module.embedding.word_embeddings.weight", param),
                    ("module.decoder.final_layernorm.weight", _parameter([1, 1, 1])),
                    ("module.output_layer.weight", _parameter([[1, 1, 1]])),
                ]
                ),
                "named_buffers": lambda self: iter(()),
        },
    )()
    engine._dims = DSV4Dims(num_layers=0, num_experts=0, compress_ratios=[])
    engine._ep_size = 1
    engine._pp_size = 1
    engine._pp_rank = 0

    params, _ = engine.get_per_tensor_param()

    assert calls == []
    converted = dict(params)
    assert calls == ["tp"]
    assert converted["embed.weight"].shape == (4, 3)
