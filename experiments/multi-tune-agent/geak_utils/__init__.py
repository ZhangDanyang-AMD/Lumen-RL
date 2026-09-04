"""Public API for orchestration-neutral GEAK adapters."""

from .catalog import load_tasks
from .errors import SandboxError
from .evaluation import CommandResult, EvaluationResult
from .sandbox import KernelSandbox
from .task import TaskSpec
from .templates import (
    FORMAT_ALIASES,
    GEMM_TEMPLATES,
    GemmTemplate,
    architecture_for_target,
    get_gemm_template,
    normalize_gemm_format,
    validate_template_target,
)

__all__ = [
    "CommandResult",
    "EvaluationResult",
    "KernelSandbox",
    "SandboxError",
    "TaskSpec",
    "FORMAT_ALIASES",
    "GEMM_TEMPLATES",
    "GemmTemplate",
    "architecture_for_target",
    "get_gemm_template",
    "load_tasks",
    "normalize_gemm_format",
    "validate_template_target",
]
