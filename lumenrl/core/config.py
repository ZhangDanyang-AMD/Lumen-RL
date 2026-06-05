"""Unified YAML + OmegaConf configuration system for LumenRL."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from lumenrl.architecture.config.assembly_config import RuntimeAssemblyConfig
from lumenrl.core.types import (
    AlgorithmName,
    GenerationBackend,
    TrainingBackend,
)

logger = logging.getLogger(__name__)


@dataclass
class FSDPEngineConfig:
    """Configuration for the FSDP2 training engine."""
    strategy: str = "fsdp2"
    fsdp_size: int = -1
    param_offload: bool = False
    optimizer_offload: bool = False
    grad_offload: bool = False
    reshard_after_forward: bool = True
    forward_only: bool = False
    seed: int = 42
    model_dtype: str = "bf16"
    mixed_precision: Optional[dict] = None
    use_remove_padding: bool = True
    ulysses_sequence_parallel_size: int = 1
    forward_prefetch: bool = False
    use_orig_params: bool = True
    use_torch_compile: bool = False


@dataclass
class McoreEngineConfig:
    """Configuration for the Megatron-Core training engine."""
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    sequence_parallel: bool = False
    param_offload: bool = False
    optimizer_offload: bool = False
    grad_offload: bool = False
    forward_only: bool = False
    seed: int = 42
    dtype: str = "bf16"
    use_distributed_optimizer: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer and LR scheduler configuration."""
    optimizer_type: str = "adamw"
    lr: float = 1e-6
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    lr_scheduler_type: str = "cosine"
    lr_warmup_steps: int = 10
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = 1000
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5


@dataclass
class LoRAConfig:
    """LoRA / PEFT configuration."""
    enabled: bool = False
    rank: int = 0
    alpha: int = 16
    target_modules: Optional[list] = None
    exclude_modules: Optional[list] = None
    merge: bool = False
    adapter_path: Optional[str] = None


@dataclass
class HFModelConfig:
    """HuggingFace model loading configuration."""
    local_path: str = ""
    model_type: str = "language_model"
    trust_remote_code: bool = True
    use_remove_padding: bool = True
    enable_gradient_checkpointing: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    use_liger: bool = False
    use_fused_kernels: bool = False
    calculate_entropy: bool = False
    calculate_sum_pi_squared: bool = False


@dataclass
class ClusterConfig:
    num_nodes: int = 1
    gpus_per_node: int = 1
    ray_address: Optional[str] = None


@dataclass
class RayWorkerRoleConfig:
    """Per-role worker-group orchestration knobs for Ray controller path."""

    # 0 means auto-infer from pool world size.
    num_workers: int = 0
    # Supported dispatch modes:
    # - dp_compute_proto (default)
    # - dp_compute
    # - dp_compute_proto_with_func
    # - dp_compute_metric
    # - one_to_all
    # - all_to_all
    # - rank_zero
    # - direct_rollout_method (forbidden in controller dispatch path)
    # Legacy alias accepted at runtime: broadcast -> one_to_all
    dispatch_mode: str = "dp_compute_proto"
    mesh_mapping: Optional[list[int]] = None
    lazy_dispatch_key: Optional[str] = None
    detached: bool = False
    process_on_nodes: Optional[list[int]] = None
    max_colocate_count: int = 1
    topology_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class RayControllerConfig:
    """Ray-controller runtime options for the trainer main path."""

    enabled: bool = False
    fuse_actor_ref: bool = False
    actor: RayWorkerRoleConfig = field(default_factory=RayWorkerRoleConfig)
    ref: RayWorkerRoleConfig = field(default_factory=RayWorkerRoleConfig)
    # Optional role->pool name mapping for complex topology routing.
    topology_map: dict[str, str] = field(default_factory=dict)


@dataclass
class ControllerConfig:
    ray: RayControllerConfig = field(default_factory=RayControllerConfig)


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
    gpu_memory_utilization: float = 0.6
    gpu_id: Optional[int] = None


@dataclass
class TrainingConfig:
    megatron_cfg: MegatronConfig = field(default_factory=MegatronConfig)
    fsdp_cfg: Optional[dict] = None
    # Training dtype controls both model param storage and FSDP2 MixedPrecision
    # param_dtype. The optimizer (AdamW) keeps master weights and momentum/variance
    # in this dtype. Use "fp32" for full FP32 training, "bf16" for mixed precision.
    # Reduce dtype is always FP32 for numerical stability.
    optimizer_dtype: str = "bf16"


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
    max_response_length: int = 20480
    train_global_batch_size: int = 64
    train_micro_batch_size: int = 8
    max_token_len_per_gpu: int = 0
    ppo_mini_batch_size: int = 0
    learning_rate: float = 1e-6
    lr_warmup_steps: int = 10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0


@dataclass
class GRPOConfig:
    num_generations: int = 8
    kl_coeff: float = 0.0
    clip_ratio: float = 0.2
    num_ppo_epochs: int = 1
    num_mini_batches: int = 1
    discount: float = 1.0


@dataclass
class DAPOConfig:
    num_generations: int = 8
    kl_coeff: float = 0.0
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    clip_ratio_c: float = 3.0
    dynamic_sampling: bool = True
    token_level_pg: bool = True
    overlong_reward_shaping: bool = True
    loss_mode: str = "token_level"  # "token_level" (standard DAPO) or "gmpo" (geometric mean PO)
    discount: float = 1.0


@dataclass
class PPOConfig:
    kl_coeff: float = 0.02
    clip_ratio: float = 0.2
    num_ppo_epochs: int = 4
    num_mini_batches: int = 4
    gae_lambda: float = 0.95
    discount: float = 1.0


@dataclass
class OPDConfig:
    """On-Policy Distillation (DeepSeek-V4 style)."""
    kl_direction: str = "reverse"
    temperature: float = 1.0
    position_weighting: bool = False
    position_decay: float = 0.8
    opd_coeff: float = 1.0
    lazy_logits: bool = True
    teacher_micro_batch_size: int = 4


@dataclass
class SpecDistillConfig:
    """Speculative Decoding draft model distillation."""
    draft_type: str = "eagle3"
    loss_type: str = "forward_kl"
    position_decay: float = 0.8
    loss_decay_gamma: float = 7.0
    num_target_layers: int = 1
    aux_hidden_state_layer_ids: Optional[list[int]] = None
    anchor_num: int = 512
    spec_length: int = 5


@dataclass
class TeacherConfig:
    """Teacher / target model configuration."""
    model_name: str = ""
    key: str = ""                               # routing key for multi-teacher
    lm_head_key: str = "lm_head.weight"
    norm_key: str = "model.norm.weight"
    load_norm: bool = False
    inference_backend: str = "hf"           # "hf" | "atom" | "sglang" | "vllm"
    quantization: str = ""                  # "" | "fp8" | "fp4" | "mxfp4"
    tensor_parallel_size: int = 1           # ATOM tensor parallelism
    gpu_ids: Optional[list[int]] = None     # GPUs for ATOM inference
    # MORI-IO P2P RDMA for GPU-direct hidden state transfer
    mori_io_host: str = "127.0.0.1"         # OOB communication address
    mori_io_port: int = 0                   # 0 = auto-assign
    mori_io_qp_per_transfer: int = 2        # RDMA queue pairs per transfer
    atom_plugin: bool = False               # Use ATOM as SGLang model plugin


@dataclass
class DraftModelConfig:
    """Draft model (student) configuration for speculative distillation."""
    model_name: str = ""
    from_scratch: bool = False
    head_dim: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    ffn_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling_type: Optional[str] = None
    rope_scaling_factor: float = 64.0
    rope_original_max_pos: int = 4096
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    dtype: str = "float16"
    resume_from: Optional[str] = None


@dataclass
class DistillationConfig:
    """Multi-teacher distillation configuration."""
    enabled: bool = False
    teacher_key: str = "data_source"           # field name in dataset used to route samples
    teachers: dict = field(default_factory=dict)  # key -> teacher config overrides
    loss_mode: str = "reverse_kl"               # k1, k3, forward_kl, reverse_kl
    topk: Optional[int] = None                  # for top-k distillation losses
    use_task_rewards: bool = False               # combine with task rewards
    distillation_loss_coef: float = 1.0          # coefficient for distillation loss


@dataclass
class AlgorithmConfig:
    name: str = AlgorithmName.GRPO.value
    adv_estimator: str = ""  # empty = auto-infer from algorithm.name
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    opd: OPDConfig = field(default_factory=OPDConfig)
    spec_distill: SpecDistillConfig = field(default_factory=SpecDistillConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    draft: DraftModelConfig = field(default_factory=DraftModelConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)


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
    lumen_norm: bool = False
    fused_mlp: bool = False
    fused_rope: bool = False
    lumen_linear: bool = False
    hf_attn_patch: bool = False


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
class CriticConfig:
    """Configuration for the critic (value) network used by PPO/GAE."""
    enabled: bool = False
    model_name: str = ""
    training_backend: str = "fsdp2"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    value_clip_ratio: float = 0.2
    num_critic_epochs: int = 1


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "results/default"
    save_steps: int = 50
    save_total_limit: int = 3
    resume: bool = True


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
class MooncakeTransferConfig:
    """Mooncake distributed KV store for hidden state transfer."""
    master_server_address: Optional[str] = None
    metadata_server: Optional[str] = None
    local_hostname: str = ""
    protocol: str = "rdma"
    device_name: str = ""
    global_segment_size: str = "16GB"
    local_buffer_size: str = "4GB"
    host_buffer_size: int = 536870912   # 512 MB
    gpu_buffer_size: int = 536870912
    async_put_pool_size: int = 4
    enable_gpu_direct: bool = False
    enable_hard_pin: bool = False
    kv_lease_ttl_s: float = 5.0


@dataclass
class AsyncTrainingConfig:
    """Configuration for fully-async separated rollout + training."""
    enabled: bool = False
    require_batches: int = 4
    trigger_parameter_sync_step: int = 4
    staleness_threshold: float = 0.0
    partial_rollout: bool = False
    use_rollout_log_probs: bool = True
    rollout_n_gpus: int = 0
    trainer_n_gpus: int = 0
    queue_maxsize: int = 64
    weight_sync_dir: str = "/tmp/lumenrl_weight_sync"


@dataclass
class TorchProfilerToolConfig:
    """Configuration for torch.profiler backend."""

    # Supported values: "cpu", "cuda", "memory", "shapes", "stack"
    contents: list[str] = field(default_factory=lambda: ["cpu", "cuda"])


@dataclass
class RocprofToolConfig:
    """Configuration for ROCm `rocprof` command-line profiling."""

    # Trace toggles.
    hip_trace: bool = True
    hsa_trace: bool = True
    kernel_trace: bool = False
    memory_copy_trace: bool = False
    sys_trace: bool = False
    timestamp_on: bool = True

    # Optional summary/statistics dump.
    stats: bool = False

    # Output control.
    output_file: str = "rocprof_trace"
    output_format: str = "csv"  # csv | json

    # Optional kernel filter (regex), only when kernel tracing is enabled.
    kernel_regex: Optional[str] = None

    # Extra raw CLI arguments appended at the end.
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ProfilerConfig:
    """Global profiler configuration for trainer/controller process.

    Example (rocprof):

    ```yaml
    profiler:
      tool: rocprof
      enable: true
      all_ranks: false
      ranks: [0]
      save_path: outputs/profile
      steps: [10, 20, 30]
      profile_continuous_steps: false
      tool_config:
        hip_trace: true
        hsa_trace: true
        kernel_trace: true
        memory_copy_trace: true
        sys_trace: false
        timestamp_on: true
        stats: true
        output_file: rocprof_trace
        output_format: csv
        kernel_regex: null
        extra_args: []
    ```
    """

    tool: str = "torch"
    enable: bool = False
    all_ranks: bool = False
    ranks: list[int] = field(default_factory=list)
    save_path: str = "outputs/profile"
    steps: Optional[list[int]] = None
    profile_continuous_steps: bool = False
    tool_config: TorchProfilerToolConfig | RocprofToolConfig = field(default_factory=TorchProfilerToolConfig)


@dataclass
class LumenRLConfig:
    """Top-level configuration for LumenRL."""

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    mooncake: MooncakeTransferConfig = field(default_factory=MooncakeTransferConfig)
    async_training: AsyncTrainingConfig = field(default_factory=AsyncTrainingConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    assembly: RuntimeAssemblyConfig = field(default_factory=RuntimeAssemblyConfig)
    num_training_steps: int = 1000
    seed: int = 42
    val_dataset: str = ""
    val_steps: int = 0        # validate every N steps; 0 = no validation
    val_batch_size: int = 16

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
