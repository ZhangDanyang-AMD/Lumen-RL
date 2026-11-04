"""Lumen-RL's thin MXFP4 setup and lifecycle wiring."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch.nn as nn

from lumenrl.quantization.mxfp4_training import setup_mxfp4_training
from lumenrl.trainer.spec_distill_trainer import SpecDistillTrainer


def test_disabled_does_not_mutate_cache_environment(monkeypatch) -> None:
    monkeypatch.setenv("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE", "keep")
    model = nn.Linear(8, 8)

    manager, returned_model, cache_enabled = setup_mxfp4_training(
        model, SimpleNamespace(mxfp4=False)
    )

    assert manager is None
    assert returned_model is model
    assert cache_enabled is False
    assert os.environ["LUMEN_MXFP4_DISABLE_WEIGHT_CACHE"] == "keep"


def test_enabled_constructs_fixed_lumen_config_and_clears_cache_opt_out(
    monkeypatch, fake_lumen,
) -> None:
    calls, _ = fake_lumen
    monkeypatch.setenv("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE", "1")
    model = nn.Linear(8, 8)
    config = SimpleNamespace(
        mxfp4=True,
        mxfp4_disable_weight_cache=False,
        mxfp4_quant_weight_selection={
            "mode": "explicit",
            "include_quant_weight": ["layers.*.mlp.gate_proj.weight"],
            "exclude_quant_weight": ["fc.weight"],
        },
        lumen_norm=True,
        lumen_linear=False,
        hf_attn_patch=True,
        fused_mlp=False,
        fused_rope=False,
    )

    manager, returned_model, cache_enabled = setup_mxfp4_training(model, config)

    assert manager == "manager"
    assert returned_model is model
    assert cache_enabled is True
    assert "LUMEN_MXFP4_DISABLE_WEIGHT_CACHE" not in os.environ
    lumen_kwargs = calls["config"]
    assert lumen_kwargs["format"] == "mxfp4"
    assert lumen_kwargs["scaling"] == "blockwise"
    assert lumen_kwargs["block_size"] == 32
    assert lumen_kwargs["quantize_grad"] is None
    assert lumen_kwargs["lumen_linear"] is False
    assert (
        lumen_kwargs["mxfp4_quant_weight_selection"]
        == config.mxfp4_quant_weight_selection
    )


def test_cache_opt_out_sets_lumen_environment(
    monkeypatch, fake_lumen,
) -> None:
    setup_mxfp4_training(
        nn.Linear(8, 8),
        SimpleNamespace(
            mxfp4=True,
            mxfp4_disable_weight_cache=True,
            mxfp4_quant_weight_selection={"mode": "all_linear"},
            lumen_linear=False,
        ),
    )

    assert os.environ["LUMEN_MXFP4_DISABLE_WEIGHT_CACHE"] == "1"


def test_lumen_linear_is_rejected_with_mxfp4(monkeypatch) -> None:
    monkeypatch.delenv("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE", raising=False)

    try:
        setup_mxfp4_training(
            nn.Linear(8, 8),
            SimpleNamespace(mxfp4=True, lumen_linear=True),
        )
    except ValueError as exc:
        assert "lumen_linear must be false" in str(exc)
    else:
        raise AssertionError("Expected conflicting lumen_linear to be rejected")

    assert "LUMEN_MXFP4_DISABLE_WEIGHT_CACHE" not in os.environ


@pytest.mark.parametrize(
    ("cache_enabled", "expected_calls"),
    [(True, 1), (False, 0)],
)
def test_optimizer_hook_registration(
    fake_lumen, cache_enabled: bool, expected_calls: int,
) -> None:
    _, quantize_module = fake_lumen
    calls: list[tuple[object, object]] = []
    quantize_module.register_mxfp4_weight_optimizer_hooks = (
        lambda model, optimizer: calls.append((model, optimizer))
    )

    trainer = SpecDistillTrainer.__new__(SpecDistillTrainer)
    trainer._mxfp4_cache_enabled = cache_enabled
    trainer._mxfp4_manager = object()
    trainer._mxfp4_optimizer_hook_registered = False
    trainer._draft_model = object()
    inner_optimizer = object()
    trainer._optimizer = SimpleNamespace(optimizer=inner_optimizer)

    trainer._register_mxfp4_optimizer_hook()
    trainer._register_mxfp4_optimizer_hook()

    assert len(calls) == expected_calls
    if expected_calls:
        assert calls == [(trainer._draft_model, inner_optimizer)]
    else:
        assert trainer._mxfp4_optimizer_hook_registered is False


def test_trainer_delegates_cache_invalidation_to_lumen(fake_lumen) -> None:
    _, quantize_module = fake_lumen
    model = nn.Linear(8, 8)
    calls: list[object] = []

    def invalidate(arg):
        calls.append(arg)
        return 2

    quantize_module.invalidate_mxfp4_weight_caches = invalidate
    trainer = SpecDistillTrainer.__new__(SpecDistillTrainer)
    trainer._mxfp4_cache_enabled = True
    trainer._mxfp4_manager = object()
    trainer._draft_model = model
    trainer._rank = 0

    assert trainer._invalidate_mxfp4_weight_caches() == 2
    assert calls == [model]


def test_device_move_invalidates_cache_before_moving() -> None:
    trainer = SpecDistillTrainer.__new__(SpecDistillTrainer)
    trainer._draft_model = nn.Linear(8, 8)
    events: list[str] = []
    trainer._invalidate_mxfp4_weight_caches = lambda: events.append("invalidate")

    trainer._move_draft_storage("cpu")

    assert events == ["invalidate"]
