"""CLI entry point for LumenRL RL training.

Usage:
    python -m lumenrl.trainer.main --config path/to/config.yaml [overrides...]

    torchrun --nproc_per_node=8 -m lumenrl.trainer.main --config config.yaml
"""

from __future__ import annotations

import logging
import os
import sys

import torch


class _FlushHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit (unbuffered logging)."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _setup_logging() -> None:
    level = os.environ.get("LUMENRL_LOG_LEVEL", "INFO").upper()
    handler = _FlushHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


def _setup_distributed(use_ray_controller: bool) -> None:
    """Initialize torch.distributed when launched via torchrun."""
    if use_ray_controller:
        logging.getLogger(__name__).info("Skipping torch.distributed init (Ray controller mode).")
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        from datetime import timedelta
        timeout = timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT", 7200)))
        torch.distributed.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            timeout=timeout,
        )
        rank = torch.distributed.get_rank()
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)


def main() -> None:
    _setup_logging()

    from lumenrl.core.config import LumenRLConfig
    from lumenrl.architecture.assembly.runtime_assembler import RuntimeAssembler
    from lumenrl.core.types import AlgorithmName
    from lumenrl.trainer.callbacks import (
        LoggingCallback,
        CheckpointCallback,
        EvalCallback,
        WandbCallback,
    )

    config = LumenRLConfig.from_cli()
    use_ray_controller = bool(config.controller.ray.enabled) or os.environ.get("LUMENRL_USE_RAY_CONTROLLER", "0") == "1"
    algo_name = config.algorithm.name.lower()
    # Only pure RLTrainer ray-controller path replaces torch.distributed init.
    # Other trainers may enable RayCluster lifecycle but still rely on torch.distributed.
    skip_torch_distributed = use_ray_controller and (not config.async_training.enabled) and (
        algo_name in {"grpo", "dapo", "ppo"}
    )
    _setup_distributed(use_ray_controller=skip_torch_distributed)
    logger = logging.getLogger("lumenrl.main")
    logger.info("LumenRL starting: algo=%s, model=%s, steps=%d, async=%s",
                config.algorithm.name, config.policy.model_name,
                config.num_training_steps, config.async_training.enabled)

    algo_name = config.algorithm.name.lower()
    if config.assembly.use_new_assembler:
        assembler = RuntimeAssembler(config)
        graph = assembler.build_graph()
        trainer = graph["trainer_cls"](config)
    else:
        from lumenrl.trainer.rl_trainer import RLTrainer
        from lumenrl.trainer.async_trainer import AsyncRLTrainer

        if algo_name == AlgorithmName.OPD.value:
            from lumenrl.trainer.opd_trainer import OPDTrainer

            trainer = OPDTrainer(config)
        elif algo_name == AlgorithmName.SPEC_DISTILL.value:
            from lumenrl.trainer.spec_distill_trainer import SpecDistillTrainer

            trainer = SpecDistillTrainer(config)
        elif config.async_training.enabled:
            trainer = AsyncRLTrainer(config)
        else:
            trainer = RLTrainer(config)

    cbs: list = [LoggingCallback(interval=max(1, config.logger.log_interval))]

    if config.checkpointing.checkpoint_dir:
        cbs.append(CheckpointCallback(
            checkpoint_dir=config.checkpointing.checkpoint_dir,
            save_interval=config.checkpointing.save_steps,
            save_total_limit=config.checkpointing.save_total_limit,
        ))

    if config.eval.enabled:
        cbs.append(EvalCallback(interval=config.eval.interval))

    if config.logger.wandb_enabled:
        cbs.append(WandbCallback(
            project=config.logger.wandb.project,
            name=config.logger.wandb.name or None,
            entity=config.logger.wandb.entity or None,
        ))

    trainer.callbacks = cbs

    try:
        trainer.setup()
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception:
        logger.exception("Training failed.")
        sys.exit(1)
    finally:
        trainer.cleanup()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    logger.info("LumenRL finished.")


if __name__ == "__main__":
    main()
