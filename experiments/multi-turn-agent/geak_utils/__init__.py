"""Public API for orchestration-neutral GEAK adapters."""

from .aiter_discovery import (
    HIGH_CONFIDENCE_SCORE,
    AiterCandidate,
    AiterDiscoveryIndex,
    AiterQuery,
    build_aiter_index,
    discover_aiter,
)
from .catalog import load_tasks
from .errors import SandboxError
from .evaluation import CommandResult, EvaluationResult
from .local_templates import (
    VerifiedTemplateRecord,
    find_verified_template,
    load_verified_templates,
    register_verified_template,
)
from .sandbox import KernelSandbox
from .task import TaskSpec
from .template_validation import (
    ValidationIssue,
    ValidationReport,
    validate_generated_template,
)
from .templates import (
    FORMAT_ALIASES,
    GEMM_TEMPLATES,
    GemmTemplate,
    architecture_for_target,
    canonical_gemm_template_for_contract,
    gemm_template_matches_contract,
    get_gemm_template,
    normalize_gemm_format,
    validate_template_target,
)

__all__ = [
    "AiterCandidate",
    "AiterDiscoveryIndex",
    "AiterQuery",
    "CommandResult",
    "EvaluationResult",
    "FORMAT_ALIASES",
    "GEMM_TEMPLATES",
    "HIGH_CONFIDENCE_SCORE",
    "GemmTemplate",
    "KernelSandbox",
    "SandboxError",
    "TaskSpec",
    "ValidationIssue",
    "ValidationReport",
    "VerifiedTemplateRecord",
    "architecture_for_target",
    "build_aiter_index",
    "canonical_gemm_template_for_contract",
    "discover_aiter",
    "find_verified_template",
    "gemm_template_matches_contract",
    "get_gemm_template",
    "load_tasks",
    "load_verified_templates",
    "normalize_gemm_format",
    "register_verified_template",
    "validate_generated_template",
    "validate_template_target",
]
