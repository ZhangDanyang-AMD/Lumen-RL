"""Tests for the in-place ROCm Megatron compatibility patches."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


_PATCH_SCRIPT = (
    Path(__file__).parents[2]
    / "examples"
    / "GRPO"
    / "dsv4"
    / "patch_rocm_megatron_dsv4.py"
)
_SPEC = importlib.util.spec_from_file_location("patch_rocm_megatron_dsv4", _PATCH_SCRIPT)
assert _SPEC and _SPEC.loader
patcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(patcher)


def test_patch_tp_copy_supports_fp32_gradient_reduction(tmp_path) -> None:
    mappings = (
        tmp_path / "megatron" / "core" / "tensor_parallel" / "mappings.py"
    )
    mappings.parent.mkdir(parents=True)
    mappings.write_text(
        """
def copy_to_tensor_model_parallel_region(input_, group=None):
    group = get_tensor_model_parallel_group_if_none(group)
    return _CopyToModelParallelRegion.apply(input_, group)

class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None
"""
    )

    assert hasattr(patcher, "patch_tp_copy_fp32_gradient_reduce")
    assert patcher.patch_tp_copy_fp32_gradient_reduce(str(tmp_path))

    patched = mappings.read_text()
    assert "all_reduce_grad_fp32=False" in patched
    assert "grad_output.float()" in patched
    assert "to(grad_output.dtype)" in patched
    assert "return grad_input, None, None" in patched
    assert not patcher.patch_tp_copy_fp32_gradient_reduce(str(tmp_path))


def test_patch_transformer_layer_wraps_tensor_attention_output(tmp_path) -> None:
    layer = tmp_path / "megatron" / "core" / "transformer" / "transformer_layer.py"
    layer.parent.mkdir(parents=True)
    layer.write_text(
        """
        attention_output_with_bias = self.self_attention()
        nvtx_range_pop(suffix="self_attention")

        if self.recompute_input_layernorm:
            pass
"""
    )

    assert patcher.patch_transformer_layer(str(tmp_path))

    patched = layer.read_text()
    assert "isinstance(attention_output_with_bias, torch.Tensor)" in patched
    assert "attention_output_with_bias = (attention_output_with_bias, None)" in patched
    assert not patcher.patch_transformer_layer(str(tmp_path))


def test_patch_hybrid_optimizer_disables_foreach_temporaries(tmp_path) -> None:
    optimizer = (
        tmp_path
        / "megatron"
        / "core"
        / "optimizer"
        / "cpu_offloading"
        / "hybrid_optimizer.py"
    )
    optimizer.parent.mkdir(parents=True)
    optimizer.write_text(
        """
self.cpu_optimizers = [self.cpu_optimizer_cls(self.cpu_param_groups)]
self.gpu_optimizer = self.gpu_optimizer_cls(self.gpu_param_groups)
cpu_optimizers.append(cpu_optimizer_cls([_cpu_param_group]))
"""
    )

    assert hasattr(patcher, "patch_hybrid_optimizer_disable_foreach")
    assert patcher.patch_hybrid_optimizer_disable_foreach(str(tmp_path))

    patched = optimizer.read_text()
    assert patched.count("foreach=False") == 3
    assert not patcher.patch_hybrid_optimizer_disable_foreach(str(tmp_path))


def test_patch_distrib_optimizer_routes_offloaded_grads_without_gpu_fp32_copy(
    tmp_path,
) -> None:
    optimizer = (
        tmp_path / "megatron" / "core" / "optimizer" / "distrib_optimizer.py"
    )
    optimizer.parent.mkdir(parents=True)
    optimizer.write_text(
        "shard_main_param.grad = shard_model_grad.float()\n"
    )

    assert hasattr(patcher, "patch_distrib_optimizer_grad_copy")
    assert patcher.patch_distrib_optimizer_grad_copy(str(tmp_path))

    patched = optimizer.read_text()
    assert "gpu_params_map_cpu_copy" in patched
    assert "shard_main_param.decoupled_grad = shard_model_grad" in patched
    assert "shard_model_grad.to(shard_main_param)" not in patched
    assert not patcher.patch_distrib_optimizer_grad_copy(str(tmp_path))


def test_patch_hybrid_optimizer_streams_full_offload_sgd(tmp_path) -> None:
    optimizer = (
        tmp_path
        / "megatron"
        / "core"
        / "optimizer"
        / "cpu_offloading"
        / "hybrid_optimizer.py"
    )
    optimizer.parent.mkdir(parents=True)
    optimizer.write_text(
        """
class HybridDeviceOptimizer:
    def step(self, closure=None):
        self._sync_hdo_param_groups_to_sub_optimizers()

        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            self._set_sub_optimizer_grads()
"""
    )

    assert hasattr(patcher, "patch_hybrid_optimizer_streaming_sgd")
    assert patcher.patch_hybrid_optimizer_streaming_sgd(str(tmp_path))

    patched = optimizer.read_text()
    assert "def _can_stream_full_offload_sgd(self):" in patched
    assert "def _stream_full_offload_sgd_step(self, closure=None):" in patched
    assert "if self._can_stream_full_offload_sgd():" in patched
    assert "torch.optim.SGD" in patched
    assert "staging = torch.empty(" in patched
    assert "(2, max_numel)," in patched
    assert "finish_slot(slot)" in patched
    assert "cpu_copy_map_grad" not in patched
    assert 'getattr(orig_param, "decoupled_grad", orig_param.grad)' in patched
    assert "cpu_param.mul_(1.0 - lr * weight_decay)" in patched
    assert "cpu_param.add_(staging_view, alpha=-lr)" in patched
    assert "orig_param.copy_(cpu_param" in patched
    assert not patcher.patch_hybrid_optimizer_streaming_sgd(str(tmp_path))


def test_patch_transformer_config_admits_sqrtsoftplus_idempotently(tmp_path) -> None:
    config = (
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "transformer_config.py"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        'moe_router_score_function: Literal["softmax", "sigmoid"] = "softmax"\n'
    )

    assert patcher.patch_transformer_config(str(tmp_path))
    assert (
        'Literal["softmax", "sigmoid", "sqrtsoftplus"]'
        in config.read_text()
    )
    assert not patcher.patch_transformer_config(str(tmp_path))


def test_patch_transformer_config_allows_sqrtsoftplus_expert_bias(tmp_path) -> None:
    config = (
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "transformer_config.py"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        '''
moe_router_score_function: Literal["softmax", "sigmoid"] = "softmax"

if (
    self.moe_router_enable_expert_bias
    and self.moe_router_score_function != "sigmoid"
):
    raise ValueError("Expert bias only supports sigmoid")
'''
    )

    assert patcher.patch_transformer_config(str(tmp_path))
    patched = config.read_text()
    assert (
        'self.moe_router_score_function not in ("sigmoid", "sqrtsoftplus")'
        in patched
    )
    assert not patcher.patch_transformer_config(str(tmp_path))


def test_patch_transformer_config_dsa_section_is_fully_idempotent(tmp_path) -> None:
    config = (
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "transformer_config.py"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        '''
@dataclass
class TransformerConfig:
    experimental_attention_variant: Optional[
        Literal['gated_delta_net', 'dsa']
    ] = None

    ####################
    # DSA
    ####################
    dsa_indexer_n_heads: int = 64
    """Number of DSA indexer heads."""
    dsa_indexer_head_dim: int = 128
    """Dimension of each DSA indexer head."""
    dsa_indexer_topk: int = 2048
    """Number of selected DSA tokens."""

    moe_router_score_function: Literal["softmax", "sigmoid"] = "softmax"

    def __post_init__(self):
        if self.experimental_attention_variant == "dsa":
            self.dsa_mode = True
'''
    )

    assert patcher.patch_transformer_config(str(tmp_path))
    first_patch = config.read_text()
    assert first_patch.count("# DSV4") == 1
    assert first_patch.count("dsv4_mode: bool = False") == 1
    assert first_patch.count(
        'if self.experimental_attention_variant == "dsv4":'
    ) == 1
    assert "sqrtsoftplus" in first_patch

    assert not patcher.patch_transformer_config(str(tmp_path))
    assert config.read_text() == first_patch


def _write_unpatched_moe_routing(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
import torch


def compute_topk(scores, topk, num_groups=None, group_topk=None):
    return torch.topk(scores, k=topk, dim=1)


def topk_routing_with_score_function(
    logits, topk, num_groups=None, group_topk=None, scaling_factor=None,
    score_function="softmax", expert_bias=None, router_replay=None,
):
    if score_function == "softmax":
        scores = torch.softmax(
            logits, dim=-1, dtype=torch.float32
        ).type_as(logits)
        probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(
                scores_for_routing, topk, num_groups, group_topk
            )
            scores = torch.gather(
                scores, dim=1, index=top_indices
            ).type_as(logits)
        else:
            scores, top_indices = compute_topk(
                scores, topk, num_groups, group_topk
            )
        probs = (
            scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
            if topk > 1
            else scores
        )
    else:
        raise ValueError(f"Invalid score_function: {score_function}")
    if scaling_factor:
        probs = probs * scaling_factor
    routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
    routing_map = torch.zeros_like(logits).int().scatter(
        1, top_indices, 1
    ).bool()
    return routing_probs, routing_map
"""
    )


def test_patch_moe_utils_adds_unbiased_sqrtsoftplus_routing(tmp_path) -> None:
    moe_utils = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    )
    _write_unpatched_moe_routing(moe_utils)
    (moe_utils.parent / "router.py").write_text("class TopKRouter:\n    pass\n")

    assert patcher.patch_moe_router_score_function(str(tmp_path))

    patched = moe_utils.read_text()
    assert 'elif score_function in ("sigmoid", "sqrtsoftplus"):' in patched
    assert "torch.nn.functional.softplus(logits.float()).sqrt()" in patched
    assert "torch.sigmoid(logits.float()).type_as(logits)" in patched
    assert "scores_for_routing = scores + expert_bias" in patched
    assert "compute_topk(scores_for_routing, topk, num_groups, group_topk)" in patched
    assert (
        "scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)"
        in patched
    )
    assert (
        "probs = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-20) "
        "if topk > 1 else scores"
        in patched
    )
    assert "probs = probs * scaling_factor" in patched
    assert 'if score_function == "softmax":' in patched
    assert 'if score_function == "sigmoid":' in patched
    assert not patcher.patch_moe_router_score_function(str(tmp_path))


def test_patched_sigmoid_is_bf16_and_scatter_compatible(tmp_path) -> None:
    moe_utils = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    )
    _write_unpatched_moe_routing(moe_utils)
    assert patcher.patch_moe_router_score_function(str(tmp_path))

    namespace: dict[str, object] = {}
    exec(compile(moe_utils.read_text(), str(moe_utils), "exec"), namespace)
    route = namespace["topk_routing_with_score_function"]

    logits = torch.tensor([[0.0, 4.0]], dtype=torch.bfloat16)
    expert_bias = torch.tensor([10.0, 0.0], dtype=torch.float32)
    routing_probs, routing_map = route(
        logits,
        1,
        score_function="sigmoid",
        expert_bias=expert_bias,
    )

    assert routing_map.tolist() == [[True, False]]
    assert routing_probs.dtype == torch.bfloat16
    expected = torch.sigmoid(logits[:, :1].float()).to(logits.dtype)
    torch.testing.assert_close(routing_probs[:, :1], expected)
    torch.testing.assert_close(routing_probs[:, 1:], torch.zeros_like(expected))

    fp32_logits = logits.float()
    sqrt_routing_probs, _ = route(
        fp32_logits,
        1,
        score_function="sqrtsoftplus",
        expert_bias=expert_bias,
    )
    assert sqrt_routing_probs.dtype == fp32_logits.dtype


def test_patch_moe_router_falls_back_to_compatible_router_module(tmp_path) -> None:
    router = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "router.py"
    )
    _write_unpatched_moe_routing(router)

    assert patcher.patch_moe_router_score_function(str(tmp_path))
    assert "sqrtsoftplus" in router.read_text()
    assert not patcher.patch_moe_router_score_function(str(tmp_path))


def test_patch_moe_router_prefers_moe_utils_definition(tmp_path) -> None:
    moe_dir = tmp_path / "megatron" / "core" / "transformer" / "moe"
    moe_utils = moe_dir / "moe_utils.py"
    router = moe_dir / "router.py"
    _write_unpatched_moe_routing(moe_utils)
    _write_unpatched_moe_routing(router)

    assert patcher.patch_moe_router_score_function(str(tmp_path))
    assert "sqrtsoftplus" in moe_utils.read_text()
    assert "sqrtsoftplus" not in router.read_text()


def test_patch_moe_router_returns_false_when_definition_is_absent(tmp_path) -> None:
    moe_dir = tmp_path / "megatron" / "core" / "transformer" / "moe"
    moe_dir.mkdir(parents=True)
    (moe_dir / "moe_utils.py").write_text("def unrelated_utility():\n    pass\n")
    (moe_dir / "router.py").write_text("class TopKRouter:\n    pass\n")

    assert not patcher.patch_moe_router_score_function(str(tmp_path))


def test_patch_checkpoint_recompute_scopes_backward_replay_and_restores(tmp_path) -> None:
    random_py = (
        tmp_path / "megatron" / "core" / "tensor_parallel" / "random.py"
    )
    random_py.parent.mkdir(parents=True)
    random_py.write_text(
        """
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def backward(ctx, *args):
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)
        return outputs
"""
    )

    assert patcher.patch_checkpoint_router_replay(str(tmp_path))

    patched = random_py.read_text()
    assert "RouterReplay.global_router_replay_instances" in patched
    assert "instance.router_replay_action" in patched
    assert "instance.set_router_replay_action(" in patched
    assert "RouterReplayAction.REPLAY_BACKWARD" in patched
    assert "outputs = ctx.run_function(*detached_inputs)" in patched
    assert "finally:" in patched
    assert "instance.clear_router_replay_action()" in patched
    assert "instance.set_router_replay_action(prior_action)" in patched
    assert "set_global_router_replay_action" not in patched
    assert not patcher.patch_checkpoint_router_replay(str(tmp_path))


def test_checkpoint_patch_stamps_importable_runtime_capability(tmp_path) -> None:
    random_py = (
        tmp_path / "megatron" / "core" / "tensor_parallel" / "random.py"
    )
    _write_checkpoint_runtime_fixture(random_py)

    assert patcher.patch_checkpoint_router_replay(str(tmp_path))
    first_patch = random_py.read_text()
    namespace: dict[str, object] = {}
    exec(compile(first_patch, str(random_py), "exec"), namespace)

    assert namespace[
        "LUMENRL_R3_CAPABILITY_CHECKPOINT_REPLAY_BACKWARD"
    ] is True
    assert not patcher.patch_checkpoint_router_replay(str(tmp_path))
    assert random_py.read_text() == first_patch


def test_router_patch_does_not_cross_into_next_function(tmp_path) -> None:
    moe_utils = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    )
    moe_utils.parent.mkdir(parents=True)
    original = """
def topk_routing_with_score_function(logits, topk, score_function="softmax"):
    if score_function == "softmax":
        return logits
    return logits

def unrelated(logits, topk, score_function="softmax"):
    if score_function == "softmax":
        return logits
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        probs, top_indices = compute_topk(scores, topk)
    else:
        raise ValueError(score_function)
    return probs, top_indices
"""
    moe_utils.write_text(original)

    assert not patcher.patch_moe_router_score_function(str(tmp_path))
    assert moe_utils.read_text() == original


def test_router_patch_rejects_malformed_target_without_cross_function_write(
    tmp_path,
) -> None:
    moe_utils = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    )
    moe_utils.parent.mkdir(parents=True)
    original = """
def topk_routing_with_score_function(logits, topk, score_function="softmax"):
    if score_function == "softmax":
        return logits
    elif score_function == "sigmoid":
        return torch.sigmoid(logits)
    return logits

def unrelated(score_function):
    if score_function:
        return True
    else:
        raise ValueError(score_function)
"""
    moe_utils.write_text(original)

    assert not patcher.patch_moe_router_score_function(str(tmp_path))
    assert moe_utils.read_text() == original


def test_sqrtsoftplus_exact_golden_bias_normalization_and_scaling(tmp_path) -> None:
    moe_utils = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    )
    _write_unpatched_moe_routing(moe_utils)
    assert patcher.patch_moe_router_score_function(str(tmp_path))
    namespace: dict[str, object] = {}
    exec(compile(moe_utils.read_text(), str(moe_utils), "exec"), namespace)
    route = namespace["topk_routing_with_score_function"]

    logits = torch.tensor([[-2.0, 1.0, 3.0]], dtype=torch.float32)
    unbiased_scores = torch.sqrt(torch.nn.functional.softplus(logits))
    unbiased_ids = torch.topk(unbiased_scores, 2, dim=1).indices
    expert_bias = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float32)
    routing_probs, routing_map = route(
        logits,
        2,
        score_function="sqrtsoftplus",
        expert_bias=expert_bias,
        scaling_factor=1.5,
    )

    selected_ids = routing_map.nonzero(as_tuple=False)[:, 1].sort().values
    assert selected_ids.tolist() == [0, 2]
    assert set(selected_ids.tolist()) != set(unbiased_ids[0].tolist())
    expected_selected = unbiased_scores[:, [0, 2]]
    expected_selected = (
        expected_selected
        / expected_selected.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        * 1.5
    )
    torch.testing.assert_close(routing_probs[:, [0, 2]], expected_selected)
    assert routing_probs[0, 1].item() == 0.0

    underflow_logits = torch.full((1, 2), -1000.0, dtype=torch.float32)
    clamped_probs, _ = route(
        underflow_logits,
        2,
        score_function="sqrtsoftplus",
        scaling_factor=1.5,
    )
    assert torch.isfinite(clamped_probs).all()
    torch.testing.assert_close(clamped_probs, torch.zeros_like(clamped_probs))


def _write_checkpoint_runtime_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
import torch

class CheckpointFunction(object):
    @staticmethod
    def backward(ctx, *args):
        detached_inputs = args
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)
        return outputs
"""
    )


def _load_patched_checkpoint(tmp_path, monkeypatch):
    random_py = (
        tmp_path / "megatron" / "core" / "tensor_parallel" / "random.py"
    )
    _write_checkpoint_runtime_fixture(random_py)
    assert patcher.patch_checkpoint_router_replay(str(tmp_path))

    replay_backward = object()

    class ReplayAction:
        REPLAY_BACKWARD = replay_backward

    class Replay:
        global_router_replay_instances = []

    module_names = (
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
    )
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    replay_module_name = "megatron.core.transformer.moe.router_replay"
    replay_module = ModuleType(replay_module_name)
    replay_module.RouterReplay = Replay
    replay_module.RouterReplayAction = ReplayAction
    monkeypatch.setitem(sys.modules, replay_module_name, replay_module)

    namespace: dict[str, object] = {}
    exec(compile(random_py.read_text(), str(random_py), "exec"), namespace)
    return namespace["CheckpointFunction"], Replay, replay_backward


class _ReplayInstance:
    def __init__(self, action):
        self.router_replay_action = action

    def set_router_replay_action(self, action):
        self.router_replay_action = action

    def clear_router_replay_action(self):
        self.router_replay_action = None


def _write_router_replay_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
from enum import Enum
import torch

class RouterReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"

class RouterReplay:
    global_router_replay_instances = []

    def __init__(self):
        self.target_topk_idx = None
        self.recorded_topk_idx = None
        self.router_replay_action = None
        self.replay_backward_list = []
        RouterReplay.global_router_replay_instances.append(self)

    def set_target_indices(self, topk_indices):
        self.target_topk_idx = topk_indices
        self.replay_backward_list.append(topk_indices)

    def clear_indices(self):
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

    def set_router_replay_action(self, action):
        self.router_replay_action = action

    def get_replay_topk(self, scores, topk, num_groups=None, group_topk=None,
                        default_compute_topk=None):
        if self.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
            top_indices = self.target_topk_idx
            top_indices = top_indices.to(scores.device)
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            top_indices = self.replay_backward_list.pop(0)
            top_indices = top_indices.to(scores.device)
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        return default_compute_topk(scores, topk, num_groups, group_topk)
"""
    )


def _load_patched_router_replay(tmp_path):
    router_replay = (
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "router_replay.py"
    )
    _write_router_replay_fixture(router_replay)
    assert patcher.patch_router_replay_diagnostics(str(tmp_path))
    first_patch = router_replay.read_text()
    assert not patcher.patch_router_replay_diagnostics(str(tmp_path))
    assert router_replay.read_text() == first_patch
    namespace: dict[str, object] = {}
    exec(compile(first_patch, str(router_replay), "exec"), namespace)
    return namespace["RouterReplay"], namespace["RouterReplayAction"]


def test_router_replay_diagnostics_execute_success_without_cloning(tmp_path) -> None:
    replay_type, action = _load_patched_router_replay(tmp_path)
    replay = replay_type()
    first = torch.tensor([[0, 1], [1, 0]])
    second = torch.tensor([[2, 1], [0, 2]])
    scores = torch.arange(6, dtype=torch.float32).view(2, 3)

    replay.reset_recompute_diagnostics()
    for indices in (first, second):
        replay.set_target_indices(indices)
        replay.set_router_replay_action(action.REPLAY_FORWARD)
        _, selected = replay.get_replay_topk(scores, 2)
        assert replay.recompute_forward_indices[-1] is selected
    replay.set_router_replay_action(action.REPLAY_BACKWARD)
    replay.get_replay_topk(scores, 2)
    replay.get_replay_topk(scores, 2)

    assert replay.get_recompute_diagnostics() == (8, 0)
    assert replay.recompute_forward_indices == []


def test_router_replay_patch_stamps_fifo_and_diagnostic_capabilities(
    tmp_path,
) -> None:
    router_replay = (
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "router_replay.py"
    )
    _write_router_replay_fixture(router_replay)

    assert patcher.patch_router_replay_diagnostics(str(tmp_path))
    first_patch = router_replay.read_text()
    namespace: dict[str, object] = {}
    exec(compile(first_patch, str(router_replay), "exec"), namespace)

    assert namespace["LUMENRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO"] is True
    assert namespace["LUMENRL_R3_CAPABILITY_REPLAY_DIAGNOSTICS"] is True
    assert not patcher.patch_router_replay_diagnostics(str(tmp_path))
    assert router_replay.read_text() == first_patch


def test_runtime_capability_stamps_replace_false_values(tmp_path) -> None:
    router_replay = (
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "router_replay.py"
    )
    _write_router_replay_fixture(router_replay)
    assert patcher.patch_router_replay_diagnostics(str(tmp_path))
    router_replay.write_text(
        router_replay.read_text().replace(
            "LUMENRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO = True",
            "LUMENRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO = False",
        )
    )

    assert patcher.patch_router_replay_diagnostics(str(tmp_path))
    patched = router_replay.read_text()
    assert "LUMENRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO = True" in patched
    assert "LUMENRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO = False" not in patched
    assert not patcher.patch_router_replay_diagnostics(str(tmp_path))


def test_router_replay_diagnostics_detect_mismatched_fifo_order(tmp_path) -> None:
    replay_type, action = _load_patched_router_replay(tmp_path)
    replay = replay_type()
    first = torch.tensor([[0, 1]])
    second = torch.tensor([[2, 1]])
    scores = torch.arange(3, dtype=torch.float32).view(1, 3)

    replay.reset_recompute_diagnostics()
    for indices in (first, second):
        replay.set_target_indices(indices)
        replay.set_router_replay_action(action.REPLAY_FORWARD)
        replay.get_replay_topk(scores, 2)
    replay.replay_backward_list[:] = list(reversed(replay.replay_backward_list))
    replay.set_router_replay_action(action.REPLAY_BACKWARD)
    replay.get_replay_topk(scores, 2)
    replay.get_replay_topk(scores, 2)

    assert replay.get_recompute_diagnostics() == (4, 2)


def test_router_replay_marker_only_partial_patch_fails_without_writing(
    tmp_path,
) -> None:
    router_replay = (
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "router_replay.py"
    )
    _write_router_replay_fixture(router_replay)
    partial = router_replay.read_text().replace(
        "        self.target_topk_idx = None\n",
        "        self.target_topk_idx = None\n"
        "        self.recompute_forward_indices = []\n",
    )
    router_replay.write_text(partial)

    with pytest.raises(RuntimeError, match="partial.*diagnostic"):
        patcher.patch_router_replay_diagnostics(str(tmp_path))

    assert router_replay.read_text() == partial


def test_router_replay_malformed_full_patch_fails_without_writing(tmp_path) -> None:
    router_replay = (
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "router_replay.py"
    )
    _write_router_replay_fixture(router_replay)
    assert patcher.patch_router_replay_diagnostics(str(tmp_path))
    malformed = router_replay.read_text().replace(
        "self.recompute_forward_indices.pop(0)",
        "self.recompute_forward_indices[-1]",
    )
    router_replay.write_text(malformed)

    with pytest.raises(RuntimeError, match="partial.*diagnostic"):
        patcher.patch_router_replay_diagnostics(str(tmp_path))

    assert router_replay.read_text() == malformed


def test_checkpoint_runtime_restores_prior_actions_after_success(
    tmp_path, monkeypatch
) -> None:
    checkpoint, replay, replay_backward = _load_patched_checkpoint(
        tmp_path, monkeypatch
    )
    instances = [_ReplayInstance(None), _ReplayInstance("forward")]
    replay.global_router_replay_instances = instances

    def run_function(value):
        assert [item.router_replay_action for item in instances] == [
            replay_backward,
            replay_backward,
        ]
        return value + 1

    result = checkpoint.backward(
        SimpleNamespace(run_function=run_function), torch.tensor(2.0)
    )
    assert result.item() == 3.0
    assert [item.router_replay_action for item in instances] == [None, "forward"]


def test_checkpoint_runtime_restores_prior_actions_after_exception(
    tmp_path, monkeypatch
) -> None:
    checkpoint, replay, replay_backward = _load_patched_checkpoint(
        tmp_path, monkeypatch
    )
    instances = [_ReplayInstance("record"), _ReplayInstance(None)]
    replay.global_router_replay_instances = instances

    def run_function(_value):
        assert all(
            item.router_replay_action is replay_backward for item in instances
        )
        raise RuntimeError("recompute failed")

    with pytest.raises(RuntimeError, match="recompute failed"):
        checkpoint.backward(
            SimpleNamespace(run_function=run_function), torch.tensor(2.0)
        )
    assert [item.router_replay_action for item in instances] == ["record", None]


_OPTIONAL_PATCHERS = (
    "patch_transformer_block",
    "patch_transformer_layer",
    "patch_eav_specs",
    "patch_tp_layers",
    "patch_tp_copy_fp32_gradient_reduce",
    "patch_hybrid_optimizer_disable_foreach",
    "patch_hybrid_optimizer_streaming_sgd",
    "patch_distrib_optimizer_fp32_detach",
    "patch_distrib_optimizer_grad_copy",
)


def _disable_optional_patchers(monkeypatch) -> None:
    for name in _OPTIONAL_PATCHERS:
        monkeypatch.setattr(patcher, name, lambda _root: False)


def _write_required_main_fixtures(tmp_path) -> tuple[Path, Path, Path]:
    config = (
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "transformer_config.py"
    )
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        'moe_router_score_function: Literal["softmax", "sigmoid"] = "softmax"\n'
    )
    moe_utils = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    )
    _write_unpatched_moe_routing(moe_utils)
    random_py = (
        tmp_path / "megatron" / "core" / "tensor_parallel" / "random.py"
    )
    _write_checkpoint_runtime_fixture(random_py)
    _write_router_replay_fixture(
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "router_replay.py"
    )
    return config, moe_utils, random_py


def test_main_validates_required_patches_and_reports_actual_router_source(
    tmp_path, monkeypatch, capsys
) -> None:
    _disable_optional_patchers(monkeypatch)
    _write_required_main_fixtures(tmp_path)

    patcher.main(str(tmp_path))
    first_output = capsys.readouterr().out
    assert "transformer/moe/moe_utils.py" in first_output
    assert "transformer/moe/router.py" not in first_output

    patcher.main(str(tmp_path))
    second_output = capsys.readouterr().out
    assert "transformer/moe/moe_utils.py" in second_output


def test_required_validation_rejects_config_substring_false_positive(
    tmp_path, monkeypatch
) -> None:
    _disable_optional_patchers(monkeypatch)
    config, _, _ = _write_required_main_fixtures(tmp_path)
    assert patcher.patch_moe_router_score_function(str(tmp_path))
    assert patcher.patch_checkpoint_router_replay(str(tmp_path))
    config.write_text(
        "moe_router_score_function: "
        'Literal["softmax", "sigmoid", "sqrtsoftplus_extra"] = "softmax"\n'
    )

    with pytest.raises(RuntimeError, match="sqrtsoftplus"):
        patcher._validate_required_patches(str(tmp_path))


def test_main_rejects_absent_router_score_config_field(
    tmp_path, monkeypatch
) -> None:
    _disable_optional_patchers(monkeypatch)
    config, _, _ = _write_required_main_fixtures(tmp_path)
    config.write_text("hidden_size: int = 4096\n")

    with pytest.raises(RuntimeError, match="moe_router_score_function"):
        patcher.main(str(tmp_path))


def test_main_reports_missing_required_files_as_runtime_error(
    tmp_path, monkeypatch
) -> None:
    _disable_optional_patchers(monkeypatch)

    with pytest.raises(RuntimeError, match="Required ROCm Megatron patch inputs"):
        patcher.main(str(tmp_path))


def test_main_rejects_malformed_router_anchor_without_writing(
    tmp_path, monkeypatch
) -> None:
    _disable_optional_patchers(monkeypatch)
    _, moe_utils, _ = _write_required_main_fixtures(tmp_path)
    original = """
def topk_routing_with_score_function(logits, topk, score_function="softmax"):
    return logits

def unrelated(score_function):
    if score_function == "softmax":
        return 1
    elif score_function == "sigmoid":
        return 2
    else:
        raise ValueError(score_function)
"""
    moe_utils.write_text(original)

    with pytest.raises(RuntimeError, match="topk_routing_with_score_function"):
        patcher.main(str(tmp_path))
    assert moe_utils.read_text() == original


def test_main_rejects_missing_checkpoint_marker(
    tmp_path, monkeypatch
) -> None:
    _disable_optional_patchers(monkeypatch)
    _, _, random_py = _write_required_main_fixtures(tmp_path)
    random_py.write_text(
        """
class CheckpointFunction(object):
    @staticmethod
    def backward(ctx, *args):
        return ctx.run_function(*args)
"""
    )

    with pytest.raises(RuntimeError, match="CheckpointFunction.backward"):
        patcher.main(str(tmp_path))


def test_main_rejects_partial_router_diagnostics_without_rewriting(
    tmp_path, monkeypatch
) -> None:
    _disable_optional_patchers(monkeypatch)
    config, moe_utils, random_py = _write_required_main_fixtures(tmp_path)
    router_replay = (
        tmp_path
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "router_replay.py"
    )
    partial = router_replay.read_text().replace(
        "        self.target_topk_idx = None\n",
        "        self.target_topk_idx = None\n"
        "        self.recompute_forward_indices = []\n",
    )
    router_replay.write_text(partial)
    originals = {
        path: path.read_text()
        for path in (config, moe_utils, random_py, router_replay)
    }

    with pytest.raises(RuntimeError, match="partial.*diagnostic"):
        patcher.main(str(tmp_path))

    assert {
        path: path.read_text()
        for path in (config, moe_utils, random_py, router_replay)
    } == originals
