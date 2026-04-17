"""High-level RL training orchestration on the controller process."""

from __future__ import annotations

import logging
from typing import Any, Type

import torch

import lumenrl.algorithms  # noqa: F401  # side-effect: populate ``ALGORITHM_REGISTRY``
from lumenrl.controller.ray_cluster import RayCluster
from lumenrl.controller.ray_worker_group import RayWorkerGroup
from lumenrl.core.config import LumenRLConfig
from lumenrl.core.protocol import DataProto
from lumenrl.core.registry import ALGORITHM_REGISTRY
from lumenrl.quantization.rollout_correction import apply_rollout_correction
from lumenrl.trainer.callbacks import Callback, LoggingCallback
from lumenrl.utils.metrics import MetricsTracker, compute_kl_divergence

logger = logging.getLogger(__name__)


def _algo_num_generations(config: LumenRLConfig) -> int:
    name = config.algorithm.name.lower()
    if name == "ppo":
        return 1
    if name == "dapo":
        return int(config.algorithm.dapo.num_generations)
    return int(config.algorithm.grpo.num_generations)


class StubRolloutWorker:
    """Minimal rollout actor used when no engine-specific worker is wired."""

    def __init__(self, rank: int, world_size: int, trainer_cfg: LumenRLConfig | None = None, **_: Any) -> None:
        self.rank = rank
        self.world_size = world_size
        self._trainer_cfg = trainer_cfg
        self._record_r3 = False

    def prepare_r3_recording(self) -> None:
        """Enable MoE R3 router capture for the next rollout."""
        self._record_r3 = True

    def clear_r3_recording(self) -> None:
        self._record_r3 = False

    def rollout(self, batch: DataProto) -> DataProto:
        """Attach rollout-time tensors (log-probs, optional FP8, optional R3)."""
        if not batch.tensors:
            raise ValueError("StubRolloutWorker.rollout requires a non-empty DataProto.")

        ref = next(iter(batch.tensors.values()))
        b, t = ref.shape[0], ref.shape[1]
        device, dtype = ref.device, torch.float32

        if "input_ids" not in batch.tensors:
            batch.tensors["input_ids"] = torch.zeros((b, t), dtype=torch.long, device=device)
        if "attention_mask" not in batch.tensors:
            batch.tensors["attention_mask"] = torch.ones((b, t), dtype=torch.long, device=device)
        if "values" not in batch.tensors:
            batch.tensors["values"] = torch.randn((b, t), device=device, dtype=dtype) * 0.02

        old_lp = torch.randn((b, t), device=device, dtype=dtype) * 0.05 - 1.5
        batch.tensors["old_log_probs"] = old_lp

        if self._trainer_cfg is not None:
            if self._trainer_cfg.quantization.rollout.precision.lower() == "fp8":
                batch.tensors["fp8_log_probs"] = old_lp + torch.randn_like(old_lp) * 0.02

        if self._record_r3 and self._trainer_cfg is not None and self._trainer_cfg.moe.r3.enabled:
            # Router logits placeholder: [batch, layers, experts] flattened per layer via DataProto helper
            num_experts = 8
            batch.add_router_distributions(0, torch.randn(b, num_experts, device=device, dtype=dtype))
            logger.debug("R3: recorded stub router distribution layer 0 (rank=%d)", self.rank)

        self._record_r3 = False
        return batch

    def sync_weights_from_actor(self) -> None:
        """Receive updated policy weights from training workers (stub)."""
        logger.debug("StubRolloutWorker.sync_weights_from_actor rank=%d", self.rank)


class StubRefWorker:
    """Computes reference-policy log-probabilities (stub)."""

    def __init__(self, rank: int, world_size: int, **_: Any) -> None:
        self.rank = rank
        self.world_size = world_size

    def compute_ref_log_probs(self, batch: DataProto) -> DataProto:
        if "old_log_probs" not in batch.tensors:
            raise KeyError("compute_ref_log_probs expects 'old_log_probs'.")
        noise = 0.03 * torch.randn_like(batch.tensors["old_log_probs"])
        batch.tensors["ref_log_probs"] = batch.tensors["old_log_probs"] + noise
        return batch


class StubRewardWorker:
    """Assigns scalar rewards for GRPO/DAPO/PPO (stub)."""

    def __init__(self, rank: int, world_size: int, **_: Any) -> None:
        self.rank = rank
        self.world_size = world_size

    def compute_rewards(self, batch: DataProto) -> DataProto:
        b = batch.batch_size
        rewards = torch.randn(b, device=batch.tensors["old_log_probs"].device, dtype=torch.float32) * 0.1
        batch.tensors["rewards"] = rewards
        batch.meta.setdefault("response_lengths", [int(batch.tensors["attention_mask"][i].sum().item()) for i in range(b)])
        return batch


class StubActorWorker:
    """Training-side worker: policy forward (stub), R3 replay hooks, weight sync.

    The ``train_step`` accepts a pre-computed scalar loss tensor stored in
    ``batch.meta["loss"]`` and simulates the backward/update cycle. In a real
    deployment, the worker recomputes the RL loss inside its own process using
    the algorithm's ``compute_loss`` and calls ``.backward()`` on the
    resulting graph.
    """

    def __init__(self, rank: int, world_size: int, **_: Any) -> None:
        self.rank = rank
        self.world_size = world_size
        self._r3_mode: str | None = None

    def prepare_r3_replay(self, mode: str) -> None:
        """Install training-side R3 replay behavior for the next forward pass."""
        self._r3_mode = mode
        logger.debug("StubActorWorker.prepare_r3_replay mode=%s rank=%d", mode, self.rank)

    def finish_r3_replay(self) -> None:
        self._r3_mode = None

    def forward_log_probs(self, batch: DataProto) -> DataProto:
        """Populate ``log_probs`` for the current policy (stub forward pass)."""
        old = batch.tensors["old_log_probs"]
        noise = 0.01 * torch.randn_like(old)
        batch.tensors["log_probs"] = old + noise
        if self._r3_mode is not None:
            logger.debug(
                "R3 replay active (mode=%s); router keys=%s",
                self._r3_mode,
                list(batch.get_router_distributions().keys()),
            )
        return batch

    def train_step(self, batch: DataProto) -> DataProto:
        """Simulate backward/update using algorithm-computed loss from controller.

        In real runs, each actor worker recomputes forward + loss + backward
        using the local model and optimizer.  The stub records the loss value
        so the trainer can verify the pipeline is end-to-end connected.
        """
        loss_val = batch.meta.get("loss")
        if loss_val is not None:
            logger.debug(
                "StubActorWorker.train_step rank=%d loss=%.6f (simulated backward)",
                self.rank, float(loss_val),
            )
        else:
            logger.debug(
                "StubActorWorker.train_step rank=%d (no loss in meta; skipping backward)",
                self.rank,
            )
        return batch

    def sync_weights_to_rollout(self) -> None:
        """Push consolidated weights toward rollout workers (stub)."""
        logger.debug("StubActorWorker.sync_weights_to_rollout rank=%d", self.rank)


class RLTrainer:
    """Coordinates Ray workers, algorithms, R3, rollout correction, and callbacks."""

    def __init__(self, config: LumenRLConfig) -> None:
        self.config = config
        self.global_step: int = 0
        self.last_metrics: dict[str, float] = {}
        self.callbacks: list[Callback] = []
        self._cluster: RayCluster | None = None
        self._algorithm: Any = None
        self.rollout_wg: RayWorkerGroup | None = None
        self.ref_wg: RayWorkerGroup | None = None
        self.reward_wg: RayWorkerGroup | None = None
        self.actor_wg: RayWorkerGroup | None = None
        self._metrics = MetricsTracker()

    def setup(self) -> None:
        """Initialize Ray, worker groups, and the selected algorithm."""
        self._cluster = RayCluster(self.config.cluster)
        self._cluster.init()

        total = int(self.config.cluster.num_nodes) * int(self.config.cluster.gpus_per_node)
        if total > 0:
            quarter = max(1, total // 4)
            pool_roll = self._cluster.create_pool("rollout", num_gpus=quarter)
            pool_ref = self._cluster.create_pool("reference", num_gpus=quarter)
            pool_rw = self._cluster.create_pool("reward", num_gpus=quarter)
            pool_act = self._cluster.create_pool("actor", num_gpus=max(1, total - 3 * quarter))
        else:
            logger.warning("Cluster reports 0 GPUs; starting CPU Ray actors (num_gpus=0).")
            pool_roll = self._cluster.create_pool("rollout", num_gpus=0)
            pool_ref = self._cluster.create_pool("reference", num_gpus=0)
            pool_rw = self._cluster.create_pool("reward", num_gpus=0)
            pool_act = self._cluster.create_pool("actor", num_gpus=0)

        nw = max(1, min(self.config.cluster.gpus_per_node, 2))
        kw = {"trainer_cfg": self.config}
        self.rollout_wg = RayWorkerGroup(StubRolloutWorker, pool_roll, nw, worker_kwargs=kw)
        self.ref_wg = RayWorkerGroup(StubRefWorker, pool_ref, nw)
        self.reward_wg = RayWorkerGroup(StubRewardWorker, pool_rw, nw)
        self.actor_wg = RayWorkerGroup(StubActorWorker, pool_act, nw)

        self.rollout_wg.start()
        self.ref_wg.start()
        self.reward_wg.start()
        self.actor_wg.start()

        algo_cls: Type[Any] = ALGORITHM_REGISTRY.get(self.config.algorithm.name)
        self._algorithm = algo_cls(self.config)

        if not self.callbacks:
            self.callbacks.append(LoggingCallback(interval=max(1, self.config.logger.log_interval)))

        logger.info(
            "RLTrainer.setup complete: algorithm=%s workers_per_group=%d",
            self.config.algorithm.name,
            nw,
        )

    def run_validation(self) -> dict[str, float]:
        """Run a lightweight validation pass (stub returns diagnostics only)."""
        out: dict[str, float] = {"val/kl_proxy": 0.0}
        if self.ref_wg is None or self.rollout_wg is None:
            return out
        batch = self._build_rollout_batch()
        batch = self.rollout_wg.dispatch_and_call("rollout", batch)
        batch = self.ref_wg.dispatch_and_call("compute_ref_log_probs", batch)
        if "log_probs" not in batch.tensors:
            batch.tensors["log_probs"] = batch.tensors["old_log_probs"].clone()
        kl = compute_kl_divergence(batch.tensors["log_probs"], batch.tensors["ref_log_probs"])
        out["val/kl_proxy"] = kl
        return out

    def _build_rollout_batch(self) -> DataProto:
        g = _algo_num_generations(self.config)
        b = max(g, int(self.config.policy.train_micro_batch_size))
        b = ((b + g - 1) // g) * g
        t = min(int(self.config.policy.max_total_sequence_length), 64)
        return DataProto(
            tensors={
                "input_ids": torch.zeros(b, t, dtype=torch.long),
                "attention_mask": torch.ones(b, t, dtype=torch.long),
            },
            meta={"num_generations": g},
        )

    def _maybe_fp8_log_probs(self, batch: DataProto) -> None:
        """Populate FP8 log-probs for correction when rollout requests FP8 precision."""
        if self.config.quantization.rollout.precision.lower() != "fp8":
            return
        if "old_log_probs" in batch.tensors and "fp8_log_probs" not in batch.tensors:
            batch.tensors["fp8_log_probs"] = batch.tensors["old_log_probs"] + 0.01 * torch.randn_like(
                batch.tensors["old_log_probs"]
            )

    def _apply_rollout_correction(self, batch: DataProto) -> DataProto:
        """Delegate to the canonical ``apply_rollout_correction`` implementation."""
        return apply_rollout_correction(batch, self.config)

    def train(self) -> None:
        """Main training loop: rollout → ref → reward → advantages → correction → train → sync."""
        if self._algorithm is None or self.rollout_wg is None or self.actor_wg is None:
            raise RuntimeError("Call setup() before train().")

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for step in range(int(self.config.num_training_steps)):
            self.global_step = step
            for cb in self.callbacks:
                cb.on_step_begin(self, step)

            batch = self._build_rollout_batch()

            if self.config.moe.r3.enabled:
                self.rollout_wg.call_all("prepare_r3_recording")

            batch = self.rollout_wg.dispatch_and_call("rollout", batch)
            self._maybe_fp8_log_probs(batch)

            batch = self.ref_wg.dispatch_and_call("compute_ref_log_probs", batch)  # type: ignore[union-attr]
            batch = self.reward_wg.dispatch_and_call("compute_rewards", batch)  # type: ignore[union-attr]

            batch = self._algorithm.compute_advantages(batch)
            batch = self._apply_rollout_correction(batch)

            if self.config.moe.r3.enabled:
                self.actor_wg.call_all("prepare_r3_replay", self.config.moe.r3.replay_mode)

            metrics_accum: dict[str, float] = {}
            algo_name = self.config.algorithm.name.lower()
            if algo_name == "ppo":
                ppo_epochs = max(1, int(self.config.algorithm.ppo.num_ppo_epochs))
                n_mini = max(1, int(self.config.algorithm.ppo.num_mini_batches))
                mini_bs = max(1, batch.batch_size // n_mini)
            else:
                ppo_epochs = 1
                mini_bs = max(1, int(self.config.policy.train_micro_batch_size))

            step_count = 0
            for _ in range(ppo_epochs):
                for mini in batch.mini_batches(mini_bs):
                    updated = self.actor_wg.dispatch_and_call("forward_log_probs", mini)
                    loss, m = self._algorithm.compute_loss(updated)
                    loss_val = float(loss.detach().cpu())
                    metrics_accum.setdefault("loss", 0.0)
                    metrics_accum["loss"] += loss_val
                    for k, v in m.items():
                        metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)

                    updated.meta["loss"] = loss_val
                    updated.meta["algorithm"] = self.config.algorithm.name
                    self.actor_wg.dispatch_and_call("train_step", updated)
                    step_count += 1

            denom = max(1, step_count)
            metrics = {k: v / denom for k, v in metrics_accum.items()}
            self.last_metrics = metrics
            for k, v in metrics.items():
                self._metrics.update(k, v)

            if self.config.moe.r3.enabled:
                self.actor_wg.call_all("finish_r3_replay")

            self.actor_wg.call_all("sync_weights_to_rollout")
            self.rollout_wg.call_all("sync_weights_from_actor")

            for cb in self.callbacks:
                cb.on_step_end(self, step, metrics)

        for cb in self.callbacks:
            cb.on_train_end(self)

        logger.info("RLTrainer.train finished after %d steps.", self.config.num_training_steps)


__all__ = ["RLTrainer"]
