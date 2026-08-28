"""Unit tests for the shared Megatron engine base and backend registry."""

from __future__ import annotations

import pytest
import torch

import lumenrl.engine.training  # noqa: F401 - populate EngineRegistry
from lumenrl.engine.training.base_engine import EngineRegistry
from lumenrl.engine.training.megatron_base_engine import (
    MegatronBaseEngine,
    _FusedTokenLogProb,
)
from lumenrl.engine.training.megatron_engine import MegatronEngine
from lumenrl.engine.training.megatron_native_engine import MegatronNativeEngine
from lumenrl.workers.actor_worker import LumenActorWorker
from lumenrl.workers.critic_worker import CriticWorker


def test_registered_training_backends() -> None:
    """Both Megatron engines are registered alongside FSDP2.

    The legacy ``megatron`` backend was removed in ac37fb0 and later reinstated
    (colocated/DSV4 merges) as a supported, documented backend next to
    ``megatron_native``; the registry must expose all three.
    """
    language_backends = EngineRegistry._engines["language_model"]
    value_backends = EngineRegistry._engines["value_model"]

    for backend in ("fsdp2", "megatron", "megatron_native"):
        assert backend in language_backends, backend
    for backend in ("megatron", "megatron_native"):
        assert backend in value_backends, backend

    assert issubclass(
        EngineRegistry.get_engine_cls(model_type="language_model", backend="megatron"),
        MegatronEngine,
    )
    assert issubclass(
        EngineRegistry.get_engine_cls(model_type="language_model", backend="megatron_native"),
        MegatronNativeEngine,
    )


def test_native_engine_uses_shared_base() -> None:
    assert issubclass(MegatronNativeEngine, MegatronBaseEngine)


def test_workers_reject_unknown_backend() -> None:
    """Unknown backend strings fail fast with a clear ValueError.

    The actor accepts fsdp/fsdp2/megatron/megatron_native (plus the
    ``megatron-native`` spelling); the critic is FSDP2-only today.
    """
    actor = LumenActorWorker(
        rank=0,
        world_size=1,
        config={"policy": {"training_backend": "no_such_backend"}},
    )
    with pytest.raises(ValueError, match="Unknown policy.training_backend"):
        actor.init_model()

    for backend in ("megatron", "megatron_native", "no_such_backend"):
        critic = CriticWorker(
            rank=0,
            world_size=1,
            config={"critic": {"training_backend": backend}},
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
