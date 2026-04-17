"""Unified YAML + OmegaConf configuration system for LumenRL."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from lumenrl.core.types import (
    AlgorithmName,
    GenerationBackend,
    TrainingBackend,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    num_nodes: int = 1
    gpus_per_node: int = 1
    ray_address: Optional[str] = None


@dataclass
class MegatronConfig:
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    num_experts: Optional[int] = None
    moe_grouped_gemm: bool = False
    moe_use_legacy_grouped_gemm: bool = False


@dataclass
class AtomConfig:
    tensor_parallel_size: int = 1
    kv_cache_dtype: str = "auto"
    max_model_len: Optional[int] = None


@dataclass
class TrainingConfig:
    megatron_cfg: MegatronConfig = field(default_factory=MegatronConfig)
    fsdp_cfg: Optional[dict] = None


@dataclass
class GenerationConfig:
    atom_cfg: AtomConfig = field(default_factory=AtomConfig)


@dataclass
class PolicyConfig:
    model_name: str = ""
    training_backend: str = TrainingBackend.FSDP2.value
    generation_backend: str = GenerationBackend.ATOM.value
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    max_total_sequence_length: int = 4096
    train_global_batch_size: int = 64
    train_micro_batch_size: int = 8


@dataclass
class GRPOConfig:
    num_generations: int = 8
    kl_coeff: float = 0.0
    clip_ratio: float = 0.2
    num_ppo_epochs: int = 1
    num_mini_batches: int = 1


@dataclass
class DAPOConfig:
    num_generations: int = 8
    kl_coeff: float = 0.0
    clip_ratio_low: float = 0.8
    clip_ratio_high: float = 1.2
    dynamic_sampling: bool = True
    token_level_pg: bool = True
    overlong_reward_shaping: bool = True


@dataclass
class PPOConfig:
    kl_coeff: float = 0.02
    clip_ratio: float = 0.2
    num_ppo_epochs: int = 4
    num_mini_batches: int = 4
    gae_lambda: float = 0.95
    discount: float = 1.0


@dataclass
class AlgorithmConfig:
    name: str = AlgorithmName.GRPO.value
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


@dataclass
class RolloutQuantConfig:
    precision: str = "bf16"
    use_deep_gemm: bool = True
    num_first_layers_in_bf16: int = 0
    num_last_layers_in_bf16: int = 0


@dataclass
class TrainingQuantConfig:
    fp8: Optional[str] = None
    fp8_recipe: str = "blockwise"
    fp8_weight_cache: bool = False


@dataclass
class RolloutCorrectionConfig:
    enabled: bool = False
    method: str = "tis"
    clip: float = 1.5


@dataclass
class QuantizationConfig:
    rollout: RolloutQuantConfig = field(default_factory=RolloutQuantConfig)
    training: TrainingQuantConfig = field(default_factory=TrainingQuantConfig)
    rollout_correction: RolloutCorrectionConfig = field(
        default_factory=RolloutCorrectionConfig
    )


@dataclass
class R3Config:
    enabled: bool = False
    record_router_logits: bool = True
    replay_mode: str = "distribution"


@dataclass
class MoEConfig:
    r3: R3Config = field(default_factory=R3Config)


@dataclass
class RewardConfig:
    type: str = "function"
    function: str = "math_reward"
    dataset: str = ""
    model_name: Optional[str] = None


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "results/default"
    save_steps: int = 50
    save_total_limit: int = 3


@dataclass
class WandbConfig:
    project: str = "lumenrl"
    name: str = ""
    entity: Optional[str] = None


@dataclass
class LoggerConfig:
    wandb_enabled: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_interval: int = 1
    num_val_samples_to_print: int = 5


@dataclass
class LumenRLConfig:
    """Top-level configuration for LumenRL."""

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    num_training_steps: int = 1000
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> "LumenRLConfig":
        """Load config from YAML file with optional CLI overrides."""
        schema = OmegaConf.structured(cls)
        file_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, file_cfg)
        if overrides:
            cli_cfg = OmegaConf.from_dotlist(overrides)
            merged = OmegaConf.merge(merged, cli_cfg)
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    @classmethod
    def from_cli(cls) -> "LumenRLConfig":
        """Parse config from command-line arguments."""
        parser = argparse.ArgumentParser(description="LumenRL")
        parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
        args, unknown = parser.parse_known_args()
        return cls.from_yaml(args.config, overrides=unknown)


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> LumenRLConfig:
    """Convenience function to load a LumenRLConfig."""
    return LumenRLConfig.from_yaml(config_path, overrides=overrides)
