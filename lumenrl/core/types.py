"""Shared type definitions for LumenRL."""

from __future__ import annotations

from enum import Enum
from typing import TypedDict

from typing_extensions import NotRequired


class TrainingBackend(str, Enum):
    FSDP2 = "fsdp2"
    MEGATRON = "megatron"


class GenerationBackend(str, Enum):
    ATOM = "atom"


class AlgorithmName(str, Enum):
    GRPO = "grpo"
    DAPO = "dapo"
    PPO = "ppo"


class FP8Precision(str, Enum):
    FP8 = "fp8"
    BF16 = "bf16"


class FP8Recipe(str, Enum):
    BLOCKWISE = "blockwise"
    TENSORWISE = "tensorwise"


class RolloutCorrectionMethod(str, Enum):
    TIS = "tis"
    MIS = "mis"


class R3ReplayMode(str, Enum):
    DISTRIBUTION = "distribution"
    HARD_ASSIGNMENT = "hard_assignment"


class DispatchMode(str, Enum):
    """How DataProto is dispatched across DP workers."""
    DP_COMPUTE_PROTO = "dp_compute_proto"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"


class ClusterConfigDict(TypedDict):
    num_nodes: int
    gpus_per_node: int
    ray_address: NotRequired[str]


class MegatronConfigDict(TypedDict):
    tensor_parallel_size: int
    expert_parallel_size: NotRequired[int]
    pipeline_parallel_size: NotRequired[int]
    num_experts: NotRequired[int]
    moe_grouped_gemm: NotRequired[bool]


class AtomConfigDict(TypedDict):
    tensor_parallel_size: int
    kv_cache_dtype: NotRequired[str]
    max_model_len: NotRequired[int]


class RolloutQuantConfigDict(TypedDict):
    precision: str
    use_deep_gemm: NotRequired[bool]
    num_first_layers_in_bf16: NotRequired[int]
    num_last_layers_in_bf16: NotRequired[int]


class TrainingQuantConfigDict(TypedDict):
    fp8: NotRequired[str]
    fp8_recipe: NotRequired[str]
    fp8_weight_cache: NotRequired[bool]


class RolloutCorrectionConfigDict(TypedDict):
    enabled: bool
    method: NotRequired[str]
    clip: NotRequired[float]


class R3ConfigDict(TypedDict):
    enabled: bool
    record_router_logits: NotRequired[bool]
    replay_mode: NotRequired[str]
