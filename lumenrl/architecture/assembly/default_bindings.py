"""Default component registrations for phase-1 assembler rollout."""

from __future__ import annotations

from lumenrl.architecture.registry.component_registries import (
    inference_backend_registry,
    trainer_registry,
    training_backend_registry,
    worker_role_registry,
)
from lumenrl.engine.inference.atom_engine import AtomEngine
from lumenrl.engine.training.fsdp_backend import FSDP2Backend
from lumenrl.engine.training.megatron_backend import MegatronBackend
from lumenrl.trainer.async_trainer import AsyncRLTrainer
from lumenrl.trainer.opd_trainer import OPDTrainer
from lumenrl.trainer.rl_trainer import RLTrainer
from lumenrl.trainer.spec_distill_trainer import SpecDistillTrainer
from lumenrl.workers.actor_worker import LumenActorWorker
from lumenrl.workers.critic_worker import CriticWorker
from lumenrl.workers.ref_worker import RefPolicyWorker
from lumenrl.workers.reward_worker import RewardWorker
from lumenrl.workers.rollout_worker import AtomRolloutWorker
from lumenrl.workers.teacher_worker import TeacherWorker


def _ensure(registry, key: str, value) -> None:
    if key not in registry:
        registry.register(key, value)


def register_default_bindings() -> None:
    """Register built-in components for assembler usage."""
    _ensure(worker_role_registry, "actor.default", LumenActorWorker)
    _ensure(worker_role_registry, "critic.default", CriticWorker)
    _ensure(worker_role_registry, "ref.default", RefPolicyWorker)
    _ensure(worker_role_registry, "reward.default", RewardWorker)
    _ensure(worker_role_registry, "rollout.atom", AtomRolloutWorker)
    _ensure(worker_role_registry, "teacher.default", TeacherWorker)

    _ensure(training_backend_registry, "fsdp", FSDP2Backend)
    _ensure(training_backend_registry, "fsdp2", FSDP2Backend)
    _ensure(training_backend_registry, "megatron", MegatronBackend)
    _ensure(inference_backend_registry, "atom", AtomEngine)

    _ensure(trainer_registry, "trainer.rl", RLTrainer)
    _ensure(trainer_registry, "trainer.async", AsyncRLTrainer)
    _ensure(trainer_registry, "trainer.opd", OPDTrainer)
    _ensure(trainer_registry, "trainer.spec_distill", SpecDistillTrainer)
