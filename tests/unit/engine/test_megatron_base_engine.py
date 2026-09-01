"""Unit tests for the shared Megatron-Native engine helpers."""

from __future__ import annotations

import pytest
import torch

import lumenrl.engine.training  # noqa: F401 - populate EngineRegistry
from lumenrl.engine.training.base_engine import EngineRegistry
from lumenrl.engine.training.megatron_base_engine import (
    MegatronBaseEngine,
    _FusedTokenLogProb,
    moe_dispatcher_kwargs,
)
from lumenrl.engine.training.megatron_native_engine import MegatronNativeEngine
from lumenrl.workers.actor_worker import LumenActorWorker
from lumenrl.workers.critic_worker import CriticWorker


def test_legacy_megatron_backend_is_not_registered() -> None:
    language_backends = EngineRegistry._engines["language_model"]
    value_backends = EngineRegistry._engines["value_model"]

    assert "megatron" not in language_backends
    assert "megatron" not in value_backends
    assert "megatron_native" in language_backends
    assert "megatron_native" in value_backends


def test_native_engine_uses_shared_base() -> None:
    assert issubclass(MegatronNativeEngine, MegatronBaseEngine)


def test_workers_reject_removed_legacy_backend() -> None:
    actor = LumenActorWorker(
        rank=0,
        world_size=1,
        config={"policy": {"training_backend": "megatron"}},
    )
    with pytest.raises(ValueError, match="Unknown policy.training_backend"):
        actor.init_model()

    critic = CriticWorker(
        rank=0,
        world_size=1,
        config={"critic": {"training_backend": "megatron"}},
    )
    with pytest.raises(ValueError, match="Unknown critic training_backend"):
        critic.init_model()


def test_fused_token_log_prob_matches_reference_value_and_gradient() -> None:
    torch.manual_seed(7)
    target = torch.tensor([0, 3, 1, 4, 2], dtype=torch.long)
    weight = torch.randn(target.numel())

    fused_logits = torch.randn(target.numel(), 7, requires_grad=True)
    reference_logits = fused_logits.detach().clone().requires_grad_(True)

    fused = _FusedTokenLogProb.apply(fused_logits, target)
    reference = torch.log_softmax(reference_logits, dim=-1).gather(
        -1, target.unsqueeze(-1)
    ).squeeze(-1)

    torch.testing.assert_close(fused, reference)

    (fused * weight).sum().backward()
    (reference * weight).sum().backward()
    torch.testing.assert_close(fused_logits.grad, reference_logits.grad)


def test_shared_packing_helpers() -> None:
    engine = MegatronBaseEngine({}, {}, {}, "")

    bins = engine._build_bins([7, 5, 4, 2], budget=9)
    assert sorted(index for group in bins for index in group) == [0, 1, 2, 3]
    assert all(sum([7, 5, 4, 2][index] for index in group) <= 9 for group in bins)

    start, length = engine._real_block(torch.tensor([0, 0, 1, 1, 1, 0]))
    assert (start, length) == (2, 3)
    assert engine._real_block(torch.zeros(4, dtype=torch.long)) == (0, 0)


def test_moe_dispatcher_kwargs_alltoall_default() -> None:
    assert moe_dispatcher_kwargs({}, tp=1, cp=1, sp=False) == {
        "moe_token_dispatcher_type": "alltoall",
    }


def test_moe_dispatcher_kwargs_mori_auto_derives_heap() -> None:
    kwargs = moe_dispatcher_kwargs(
        {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "mori",
        },
        tp=1,
        cp=1,
        sp=False,
        max_tokens_per_gpu=2048,
    )
    assert kwargs["moe_token_dispatcher_type"] == "flex"
    assert kwargs["moe_flex_dispatcher_backend"] == "mori"
    assert kwargs["moe_mori_max_tokens_per_rank"] == 2048


def test_moe_dispatcher_kwargs_mori_scales_for_cp_and_sp() -> None:
    kwargs = moe_dispatcher_kwargs(
        {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "mori",
        },
        tp=2,
        cp=2,
        sp=True,
        max_tokens_per_gpu=2048,
    )
    # ceil(2048/2)=1024 for CP, then ceil(1024/2)=512 for SP.
    assert kwargs["moe_mori_max_tokens_per_rank"] == 512


def test_moe_dispatcher_kwargs_mori_explicit_heap_and_kernel() -> None:
    kwargs = moe_dispatcher_kwargs(
        {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "mori",
            "moe_mori_max_tokens_per_rank": 4096,
            "moe_mori_kernel_type": "intranode",
        },
        tp=2,
        cp=2,
        sp=True,
        max_tokens_per_gpu=2048,
    )
    assert kwargs["moe_mori_max_tokens_per_rank"] == 4096
    assert kwargs["moe_mori_kernel_type"] == "intranode"
