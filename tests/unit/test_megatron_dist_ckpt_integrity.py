"""Megatron dist-checkpoint optimizer state: gloo gather on save, integrity check on load.

RCCL 2.27.7 (torch 2.10 ROCm) torch.distributed.gather at world size 7 delivers
only the first half of each non-root rank's tensor once it is >= ~256 MiB.
Megatron's dp_zero_gather_scatter save gathers each rank's fp32 main params and
Adam moments that way (use_gloo_comm=False, ~4.4 GiB/rank for Qwen3-8B), so the
checkpoint held zeros and the first step after a resume wrote them into the model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lumenrl.engine.training.megatron_native_engine import (
    MegatronNativeEngine,
    force_gloo_dp_zero_gather,
)


class _Opt:
    def __init__(self):
        self.calls = []

    def get_parameter_state_dp_zero(self, use_gloo_comm=True, return_on_all_ranks=False):
        self.calls.append(use_gloo_comm)
        return {"ok": True}


def test_dp_zero_gather_is_forced_onto_gloo() -> None:
    o = _Opt()
    force_gloo_dp_zero_gather(SimpleNamespace(chained_optimizers=[o]))
    assert o.get_parameter_state_dp_zero(use_gloo_comm=False) == {"ok": True}
    assert o.get_parameter_state_dp_zero() == {"ok": True}
    assert o.calls == [True, True]


def test_forcing_gloo_is_idempotent() -> None:
    o = _Opt()
    force_gloo_dp_zero_gather(o)
    force_gloo_dp_zero_gather(o)
    o.get_parameter_state_dp_zero(use_gloo_comm=False)
    assert o.calls == [True]


def _engine_with(main: torch.Tensor, model: torch.Tensor):
    mp = model.to(torch.bfloat16)
    rng = SimpleNamespace(start=0, end=main.numel())
    o = SimpleNamespace(
        model_float16_groups=[[mp]],
        shard_fp32_from_float16_groups=[[main]],
        _get_model_param_range_map=lambda p: {"param": rng},
    )
    e = object.__new__(MegatronNativeEngine)
    e.optimizer = SimpleNamespace(chained_optimizers=[o])
    e._rank = lambda: 0
    return e


def test_load_check_passes_when_main_params_match_the_model() -> None:
    w = torch.randn(64)
    _engine_with(w.clone(), w).check_main_params_match_model("after load")


def test_load_check_rejects_zeroed_main_params() -> None:
    w = torch.randn(64)
    main = w.clone()
    main[32:] = 0  # the RCCL half-gather signature
    with pytest.raises(RuntimeError, match="main params"):
        _engine_with(main, w).check_main_params_match_model("after load")
