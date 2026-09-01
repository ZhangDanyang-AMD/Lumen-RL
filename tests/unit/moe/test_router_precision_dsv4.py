import pytest
import torch

from lumenrl.moe.router_precision import _is_router, enable_fp32_moe_router


@pytest.mark.parametrize("container_name", ["ffn", "mlp"])
def test_dsv4_gate_names_are_detected_and_return_fp32(
    container_name,
    monkeypatch,
):
    monkeypatch.setenv("LUMENRL_FP32_MOE_ROUTER", "1")
    model = torch.nn.Module()
    block = torch.nn.Module()
    container = torch.nn.Module()
    gate = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    container.add_module("gate", gate)
    block.add_module(container_name, container)
    model.add_module("model", block)

    assert enable_fp32_moe_router(model) == 1
    assert gate.weight.dtype == torch.bfloat16

    output = gate(torch.ones(2, 3, dtype=torch.bfloat16))
    assert output.dtype == torch.float32


@pytest.mark.parametrize(
    "name",
    [
        "model.ffn.gatekeeper",
        "model.ffn.gate.projection",
        "model.mlp.gated",
        "model.attention.gate",
    ],
)
def test_similarly_named_non_router_modules_are_rejected(name):
    assert not _is_router(name, torch.nn.Linear(3, 2, bias=False))


@pytest.mark.parametrize("name", ["model.ffn.gate", "model.mlp.gate"])
def test_router_names_with_non_matrix_weights_are_rejected(name):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))

    assert not _is_router(name, module)
