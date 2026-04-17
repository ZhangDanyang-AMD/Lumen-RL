from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from lumenrl.core.config import LumenRLConfig
from lumenrl.core.types import AlgorithmName, GenerationBackend, TrainingBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
GRPO_YAML = REPO_ROOT / "configs" / "grpo_dense_bf16.yaml"


def test_default_config() -> None:
    cfg = LumenRLConfig()
    assert cfg.num_training_steps == 1000
    assert cfg.seed == 42
    assert cfg.algorithm.name == AlgorithmName.GRPO.value
    assert cfg.policy.training_backend == TrainingBackend.FSDP2.value
    assert cfg.policy.generation_backend == GenerationBackend.ATOM.value
    assert cfg.policy.train_global_batch_size == 64
    assert cfg.policy.train_micro_batch_size == 8
    assert cfg.checkpointing.save_steps == 50
    assert cfg.logger.wandb_enabled is False


def test_from_yaml() -> None:
    assert GRPO_YAML.is_file(), f"Missing fixture config: {GRPO_YAML}"
    cfg = LumenRLConfig.from_yaml(GRPO_YAML)
    assert cfg.algorithm.name == "grpo"
    assert cfg.policy.model_name == "Qwen/Qwen3-0.6B"
    assert cfg.policy.max_total_sequence_length == 2048
    assert cfg.policy.train_global_batch_size == 32
    assert cfg.num_training_steps == 200
    assert cfg.reward.dataset == "nvidia/OpenMathInstruct-2"


def test_cli_overrides() -> None:
    cfg = LumenRLConfig.from_yaml(
        GRPO_YAML,
        overrides=["seed=999", "policy.train_micro_batch_size=4", "logger.log_interval=10"],
    )
    assert cfg.seed == 999
    assert cfg.policy.train_micro_batch_size == 4
    assert cfg.logger.log_interval == 10


def test_fp8_config_values() -> None:
    cfg = LumenRLConfig.from_yaml(GRPO_YAML)
    assert cfg.quantization.rollout.precision == "bf16"
    assert cfg.quantization.training.fp8 is None
    assert cfg.quantization.training.fp8_recipe == "blockwise"
    assert cfg.quantization.training.fp8_weight_cache is False

    merged = LumenRLConfig.from_yaml(
        GRPO_YAML,
        overrides=[
            "quantization.training.fp8=e4m3",
            "quantization.training.fp8_recipe=tensorwise",
            "quantization.training.fp8_weight_cache=true",
        ],
    )
    assert merged.quantization.training.fp8 == "e4m3"
    assert merged.quantization.training.fp8_recipe == "tensorwise"
    assert merged.quantization.training.fp8_weight_cache is True


def test_moe_config_values() -> None:
    cfg = LumenRLConfig.from_yaml(GRPO_YAML)
    assert cfg.moe.r3.enabled is False
    assert cfg.moe.r3.record_router_logits is True
    assert cfg.moe.r3.replay_mode == "distribution"

    schema = OmegaConf.structured(LumenRLConfig)
    assert OmegaConf.select(schema, "moe.r3.enabled") is not None
