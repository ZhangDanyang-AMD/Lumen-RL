"""Offline weight PTQ for ATOM Kimi-K3 DSpark draft checkpoints."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

logger = logging.getLogger(__name__)

_MODEL_FILE = "model.safetensors"
_CONFIG_FILE = "config.json"
_MANIFEST_FILE = "atom_dspark_ptq_manifest.json"
_SUPPORTED_ARCHITECTURE = "K3DSparkModel"
_PHASE1_RUNTIME_EXCLUDES = (
    "context_proj",
    "layers.*.self_attn.fused_qkv_a_proj",
)
_PTPC_FP8 = "ptpc_fp8"
_MXFP4 = "mxfp4"
PTQ_FORMATS = (_PTPC_FP8, _MXFP4)
MIXED_PROJECTION_GROUPS = (
    "mlp_gate_up",
    "mlp_down",
    "attn_q_b",
    "attn_kv_b",
    "attn_o",
)
_MIXED_PROJECTION_LAYOUT = {
    "mlp_gate_up": (
        ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
        "mlp.gate_up_proj",
    ),
    "mlp_down": (("mlp.down_proj.weight",), "mlp.down_proj"),
    "attn_q_b": (("self_attn.q_b_proj.weight",), "self_attn.q_b_proj"),
    "attn_kv_b": (("self_attn.kv_b_proj.weight",), "self_attn.kv_b_proj"),
    "attn_o": (("self_attn.o_proj.weight",), "self_attn.o_proj"),
}


@dataclass(frozen=True)
class DSparkPTQProfile:
    """A fail-closed set of Kimi-K3 DSpark checkpoint weights to quantize."""

    name: str
    include_a_projection: bool
    include_context_projection: bool


PTQ_PROFILES: dict[str, DSparkPTQProfile] = {
    "phase1": DSparkPTQProfile(
        name="phase1",
        include_a_projection=False,
        include_context_projection=False,
    ),
    "phase2": DSparkPTQProfile(
        name="phase2",
        include_a_projection=True,
        include_context_projection=True,
    ),
}


def get_ptq_profile(name: str) -> DSparkPTQProfile:
    """Return a named Kimi-K3 DSpark PTQ profile."""

    try:
        return PTQ_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PTQ_PROFILES))
        raise ValueError(f"Unknown PTQ profile {name!r}; expected one of: {choices}") from exc


def expected_quantized_weight_names(
    num_hidden_layers: int,
    profile: DSparkPTQProfile,
) -> tuple[str, ...]:
    """Return the exact source-checkpoint weight names selected by a profile."""

    names: list[str] = []
    for layer_idx in range(num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        names.extend(
            [
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.self_attn.q_b_proj.weight",
                f"{prefix}.self_attn.kv_b_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
            ]
        )
        if profile.include_a_projection:
            names.extend(
                [
                    f"{prefix}.self_attn.q_a_proj.weight",
                    f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
                ]
            )
    if profile.include_context_projection:
        names.append("context_proj.weight")
    return tuple(sorted(names))


def expected_dspark_tensor_names(num_hidden_layers: int) -> tuple[str, ...]:
    """Return the complete BF16 Inferact/Kimi-K3-DSpark tensor schema."""

    names = [
        "confidence_head.proj.bias",
        "confidence_head.proj.weight",
        "context_norm.weight",
        "context_proj.weight",
        "embed_tokens.weight",
        "final_norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
    ]
    for layer_idx in range(num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.kv_a_layernorm.weight",
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
                f"{prefix}.self_attn.kv_b_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_a_layernorm.weight",
                f"{prefix}.self_attn.q_a_proj.weight",
                f"{prefix}.self_attn.q_b_proj.weight",
            ]
        )
    return tuple(sorted(names))


def build_quark_quant_config(
    profile: DSparkPTQProfile,
    quant_format: str,
) -> dict[str, Any]:
    """Build ATOM static-weight/dynamic-activation Quark metadata."""

    if quant_format == _PTPC_FP8:
        weight_config = {
            "qscheme": "per_channel",
            "dtype": "fp8_e4m3",
        }
    elif quant_format == _MXFP4:
        weight_config = {
            "qscheme": "per_group",
            "dtype": "fp4_e2m1",
        }
    else:
        choices = ", ".join(PTQ_FORMATS)
        raise ValueError(
            f"Unknown PTQ quant format {quant_format!r}; expected one of: {choices}"
        )
    config: dict[str, Any] = {
        "quant_method": "quark",
        "global_quant_config": {
            "weight": weight_config,
            # This is runtime metadata only. Activations are not calibrated or
            # serialized by this converter. For MXFP4, ATOM maps per_group to
            # per_1x32, dynamically quantizes each token to FP4, and dispatches
            # the A4W4 GEMM.
            "input_tensors": {"is_dynamic": True},
        },
    }
    if profile.name == "phase1":
        config["exclude"] = list(_PHASE1_RUNTIME_EXCLUDES)
    return config


def build_quark_ptpc_config(profile: DSparkPTQProfile) -> dict[str, Any]:
    """Build ATOM's static-weight/dynamic-activation Quark PTPC config."""

    return build_quark_quant_config(profile, _PTPC_FP8)


def select_mxfp4_weight_names(
    num_hidden_layers: int,
    projections: Sequence[str],
    layer_indices: Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Select source-checkpoint weights to take from an MXFP4 checkpoint.

    ``mlp_gate_up`` is one selection because ATOM fuses the checkpoint's
    separate gate/up tensors into one runtime ``gate_up_proj`` GEMM.
    """

    if num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
    projection_names = tuple(dict.fromkeys(projections))
    unknown = sorted(set(projection_names) - set(MIXED_PROJECTION_GROUPS))
    if unknown:
        choices = ", ".join(MIXED_PROJECTION_GROUPS)
        raise ValueError(
            f"Unknown mixed projection groups {unknown}; expected members of: {choices}"
        )
    if not projection_names:
        raise ValueError("At least one MXFP4 projection group must be selected")

    if layer_indices is None:
        selected_layers = tuple(range(num_hidden_layers))
    else:
        selected_layers = tuple(sorted(set(layer_indices)))
    if not selected_layers:
        raise ValueError("At least one MXFP4 layer must be selected")
    invalid_layers = [
        layer_idx
        for layer_idx in selected_layers
        if layer_idx < 0 or layer_idx >= num_hidden_layers
    ]
    if invalid_layers:
        raise ValueError(
            f"MXFP4 layer indices {invalid_layers} are outside "
            f"[0, {num_hidden_layers - 1}]"
        )

    selected: list[str] = []
    for layer_idx in selected_layers:
        for projection_name in projection_names:
            checkpoint_suffixes, _ = _MIXED_PROJECTION_LAYOUT[projection_name]
            selected.extend(
                f"layers.{layer_idx}.{suffix}" for suffix in checkpoint_suffixes
            )
    return tuple(sorted(selected))


def build_mixed_quark_quant_config(
    num_hidden_layers: int,
    mxfp4_projections: Sequence[str],
    mxfp4_layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build per-linear Quark metadata for an FP8/MXFP4 draft checkpoint."""

    # Validate and normalize the cross-product selection first.
    select_mxfp4_weight_names(
        num_hidden_layers,
        mxfp4_projections,
        mxfp4_layers,
    )
    projection_names = tuple(dict.fromkeys(mxfp4_projections))
    selected_layers = (
        tuple(range(num_hidden_layers))
        if mxfp4_layers is None
        else tuple(sorted(set(mxfp4_layers)))
    )
    all_layers = selected_layers == tuple(range(num_hidden_layers))
    mxfp4_spec = {
        "weight": {
            "qscheme": "per_group",
            "dtype": "fp4_e2m1",
        },
        "input_tensors": {"is_dynamic": True},
    }
    layer_quant_config: dict[str, dict[str, Any]] = {}
    for projection_name in projection_names:
        _, runtime_suffix = _MIXED_PROJECTION_LAYOUT[projection_name]
        if all_layers:
            layer_quant_config[f"layers.*.{runtime_suffix}"] = mxfp4_spec
        else:
            for layer_idx in selected_layers:
                layer_quant_config[f"layers.{layer_idx}.{runtime_suffix}"] = mxfp4_spec

    return {
        "quant_method": "quark",
        "global_quant_config": {
            "weight": {
                "qscheme": "per_channel",
                "dtype": "fp8_e4m3",
            },
            "input_tensors": {"is_dynamic": True},
        },
        "layer_quant_config": layer_quant_config,
        "exclude": list(_PHASE1_RUNTIME_EXCLUDES),
    }


def validate_dspark_checkpoint(
    config: dict[str, Any],
    tensor_names: set[str],
    profile: DSparkPTQProfile,
) -> tuple[str, ...]:
    """Validate the checkpoint and return its exact quantized weight set."""

    architectures = config.get("architectures")
    if architectures != [_SUPPORTED_ARCHITECTURE]:
        raise ValueError(
            "Expected a Kimi-K3 DSpark checkpoint with architectures="
            f"['{_SUPPORTED_ARCHITECTURE}']; got {architectures!r}"
        )
    num_hidden_layers = config.get("num_hidden_layers")
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError(f"Invalid num_hidden_layers: {num_hidden_layers!r}")

    expected_schema = set(expected_dspark_tensor_names(num_hidden_layers))
    missing_schema = sorted(expected_schema - tensor_names)
    unexpected_schema = sorted(tensor_names - expected_schema)
    if missing_schema or unexpected_schema:
        details = []
        if missing_schema:
            details.append("missing=" + ", ".join(missing_schema))
        if unexpected_schema:
            details.append("unexpected=" + ", ".join(unexpected_schema))
        raise ValueError("Kimi-K3 DSpark tensor schema mismatch: " + "; ".join(details))

    expected = expected_quantized_weight_names(num_hidden_layers, profile)
    existing_scales = sorted(name for name in tensor_names if name.endswith(".weight_scale"))
    if existing_scales:
        raise ValueError(
            "Input checkpoint is already quantized or contains weight scales: "
            + ", ".join(existing_scales)
        )
    return expected


def weight_scale_name(weight_name: str) -> str:
    """Return ATOM's sidecar scale name for a checkpoint weight."""

    suffix = ".weight"
    if not weight_name.endswith(suffix):
        raise ValueError(f"Expected a '.weight' tensor name, got {weight_name!r}")
    return f"{weight_name[: -len(suffix)]}.weight_scale"


@torch.no_grad()
def quantize_ptpc_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize one 2-D weight with ATOM's canonical PTPC implementation.

    The checkpoint stores standard E4M3FN because that dtype is portable in
    safetensors. ATOM performs the platform-specific E4M3FN-to-E4M3FNUZ
    normalization after loading on ROCm.
    """

    if weight.ndim != 2:
        raise ValueError(f"PTPC requires a 2-D weight, got shape={tuple(weight.shape)}")
    from aiter import QuantType
    from atom.quantization.quark.utils import quant_weight_online

    quantized, scale = quant_weight_online(
        weight.contiguous(),
        QuantType.per_Token,
        torch.float8_e4m3fn,
    )
    expected_scale_shape = (weight.shape[0], 1)
    if tuple(scale.shape) != expected_scale_shape:
        raise RuntimeError(
            f"ATOM PTPC scale shape {tuple(scale.shape)} != {expected_scale_shape}"
        )
    return quantized.contiguous(), scale.to(torch.float32).contiguous()


@torch.no_grad()
def quantize_mxfp4_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize one 2-D weight to packed MXFP4 with 1x32 E8M0 scales."""

    if weight.ndim != 2:
        raise ValueError(f"MXFP4 requires a 2-D weight, got shape={tuple(weight.shape)}")
    if weight.shape[1] % 32 != 0:
        raise ValueError(
            "MXFP4 requires the contracted dimension to be divisible by 32; "
            f"got shape={tuple(weight.shape)}"
        )
    from aiter import QuantType, dtypes
    from atom.quantization.quark.utils import quant_weight_online

    quantized, scale = quant_weight_online(
        weight.contiguous(),
        QuantType.per_1x32,
        dtypes.fp4x2,
    )
    expected_weight_shape = (weight.shape[0], weight.shape[1] // 2)
    expected_scale_shape = (weight.shape[0], weight.shape[1] // 32)
    if tuple(quantized.shape) != expected_weight_shape:
        raise RuntimeError(
            f"ATOM MXFP4 packed shape {tuple(quantized.shape)} "
            f"!= {expected_weight_shape}"
        )
    if tuple(scale.shape) != expected_scale_shape:
        raise RuntimeError(
            f"ATOM MXFP4 scale shape {tuple(scale.shape)} != {expected_scale_shape}"
        )
    return quantized.contiguous(), scale.contiguous()


def get_quantize_fn(
    quant_format: str,
) -> Callable[[Tensor], tuple[Tensor, Tensor]]:
    """Return ATOM's canonical weight quantizer for a checkpoint format."""

    if quant_format == _PTPC_FP8:
        return quantize_ptpc_weight
    if quant_format == _MXFP4:
        return quantize_mxfp4_weight
    choices = ", ".join(PTQ_FORMATS)
    raise ValueError(
        f"Unknown PTQ quant format {quant_format!r}; expected one of: {choices}"
    )


def _resolve_device(device: str, *, require_accelerator: bool) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if require_accelerator and resolved.type != "cuda":
        raise ValueError(
            "ATOM's canonical PTPC quantizer requires a ROCm/CUDA device; "
            "run this utility in the ATOM image with --device cuda"
        )
    return resolved


def _copy_checkpoint_assets(source: Path, output: Path) -> None:
    for item in source.iterdir():
        if not item.is_file() or item.name in {_MODEL_FILE, _CONFIG_FILE, _MANIFEST_FILE}:
            continue
        shutil.copy2(item, output / item.name)


def _tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _prepare_output_dir(source: Path, output: Path, overwrite: bool) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("Output directory must differ from the source checkpoint")
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output}; pass --overwrite to replace it"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)


def _checkpoint_quant_dtype(config: dict[str, Any]) -> str | None:
    quantization_config = config.get("quantization_config") or {}
    global_config = quantization_config.get("global_quant_config") or {}
    return (global_config.get("weight") or {}).get("dtype")


def _base_dspark_config(config: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "atom_dspark_ptq_format",
        "atom_dspark_ptq_profile",
        "quantization_config",
    }
    return {key: value for key, value in config.items() if key not in ignored}


def _validate_mixed_sources(
    fp8_source: Path,
    mxfp4_source: Path,
) -> tuple[dict[str, Any], tuple[str, ...], set[str]]:
    fp8_config = json.loads((fp8_source / _CONFIG_FILE).read_text())
    mxfp4_config = json.loads((mxfp4_source / _CONFIG_FILE).read_text())
    if _checkpoint_quant_dtype(fp8_config) != "fp8_e4m3":
        raise ValueError(f"{fp8_source} is not an FP8 E4M3 checkpoint")
    if _checkpoint_quant_dtype(mxfp4_config) != "fp4_e2m1":
        raise ValueError(f"{mxfp4_source} is not an MXFP4 E2M1 checkpoint")
    if _base_dspark_config(fp8_config) != _base_dspark_config(mxfp4_config):
        raise ValueError("FP8 and MXFP4 checkpoints have different model configs")

    num_hidden_layers = fp8_config.get("num_hidden_layers")
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError(f"Invalid num_hidden_layers: {num_hidden_layers!r}")
    phase1_weights = expected_quantized_weight_names(
        num_hidden_layers,
        get_ptq_profile("phase1"),
    )
    expected_tensors = set(expected_dspark_tensor_names(num_hidden_layers))
    expected_tensors.update(weight_scale_name(name) for name in phase1_weights)

    fp8_model = fp8_source / _MODEL_FILE
    mxfp4_model = mxfp4_source / _MODEL_FILE
    if not fp8_model.is_file() or not mxfp4_model.is_file():
        raise FileNotFoundError("Both mixed-quant sources must contain model.safetensors")
    with (
        safe_open(fp8_model, framework="pt", device="cpu") as fp8_checkpoint,
        safe_open(mxfp4_model, framework="pt", device="cpu") as mxfp4_checkpoint,
    ):
        fp8_names = set(fp8_checkpoint.keys())
        mxfp4_names = set(mxfp4_checkpoint.keys())
    if fp8_names != expected_tensors or mxfp4_names != expected_tensors:
        raise ValueError("Mixed-quant source tensor schemas do not match Phase 1")

    fp8_manifest_path = fp8_source / _MANIFEST_FILE
    mxfp4_manifest_path = mxfp4_source / _MANIFEST_FILE
    if fp8_manifest_path.is_file() and mxfp4_manifest_path.is_file():
        fp8_manifest = json.loads(fp8_manifest_path.read_text())
        mxfp4_manifest = json.loads(mxfp4_manifest_path.read_text())
        if fp8_manifest.get("source") != mxfp4_manifest.get("source"):
            raise ValueError("FP8 and MXFP4 checkpoints were converted from different sources")

    return fp8_config, phase1_weights, expected_tensors


def compose_mixed_dspark_checkpoint(
    fp8_source_dir: str | Path,
    mxfp4_source_dir: str | Path,
    output_dir: str | Path,
    *,
    mxfp4_projections: Sequence[str],
    mxfp4_layers: Sequence[int] | None = None,
    selection_name: str = "selective",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compose a mixed FP8/MXFP4 checkpoint without re-quantizing weights."""

    fp8_source = Path(fp8_source_dir)
    mxfp4_source = Path(mxfp4_source_dir)
    output = Path(output_dir)
    if output.resolve() == mxfp4_source.resolve():
        raise ValueError("Output directory must differ from the MXFP4 source checkpoint")
    fp8_config, phase1_weights, tensor_names = _validate_mixed_sources(
        fp8_source,
        mxfp4_source,
    )
    num_hidden_layers = fp8_config["num_hidden_layers"]
    mxfp4_weights = select_mxfp4_weight_names(
        num_hidden_layers,
        mxfp4_projections,
        mxfp4_layers,
    )
    phase1_weight_set = set(phase1_weights)
    unexpected = sorted(set(mxfp4_weights) - phase1_weight_set)
    if unexpected:
        raise ValueError(f"MXFP4 selection is outside the Phase 1 weight set: {unexpected}")

    mxfp4_tensor_names = set(mxfp4_weights)
    mxfp4_tensor_names.update(weight_scale_name(name) for name in mxfp4_weights)
    fp8_weights = tuple(sorted(phase1_weight_set - set(mxfp4_weights)))
    selected_layers = (
        tuple(range(num_hidden_layers))
        if mxfp4_layers is None
        else tuple(sorted(set(mxfp4_layers)))
    )
    projection_names = tuple(dict.fromkeys(mxfp4_projections))
    manifest: dict[str, Any] = {
        "format_version": 1,
        "model_architecture": _SUPPORTED_ARCHITECTURE,
        "profile": selection_name,
        "quant_format": "mixed_fp8_mxfp4",
        "activation_quantization": "dynamic_per_linear_fp8_or_mxfp4_at_atom_runtime",
        "fp8_source": str(fp8_source.resolve()),
        "mxfp4_source": str(mxfp4_source.resolve()),
        "output": str(output.resolve()),
        "mxfp4_projection_groups": list(projection_names),
        "mxfp4_layers": list(selected_layers),
        "mxfp4_weight_count": len(mxfp4_weights),
        "mxfp4_weights": list(mxfp4_weights),
        "fp8_weight_count": len(fp8_weights),
        "fp8_weights": list(fp8_weights),
    }

    _prepare_output_dir(fp8_source, output, overwrite)
    output_tensors: dict[str, Tensor] = {}
    try:
        with (
            safe_open(
                fp8_source / _MODEL_FILE,
                framework="pt",
                device="cpu",
            ) as fp8_checkpoint,
            safe_open(
                mxfp4_source / _MODEL_FILE,
                framework="pt",
                device="cpu",
            ) as mxfp4_checkpoint,
        ):
            source_metadata = fp8_checkpoint.metadata()
            for name in sorted(tensor_names):
                checkpoint = (
                    mxfp4_checkpoint if name in mxfp4_tensor_names else fp8_checkpoint
                )
                output_tensors[name] = checkpoint.get_tensor(name)

        save_file(
            output_tensors,
            output / _MODEL_FILE,
            metadata=source_metadata or {"format": "pt"},
        )
        output_config = dict(fp8_config)
        output_config["quantization_config"] = build_mixed_quark_quant_config(
            num_hidden_layers,
            projection_names,
            selected_layers,
        )
        output_config["atom_dspark_ptq_profile"] = selection_name
        output_config["atom_dspark_ptq_format"] = "mixed_fp8_mxfp4"
        (output / _CONFIG_FILE).write_text(
            json.dumps(output_config, indent=2, sort_keys=True) + "\n"
        )
        _copy_checkpoint_assets(fp8_source, output)
        (output / "README.md").write_text(
            "# Kimi-K3 DSpark Selective MXFP4\n\n"
            "This checkpoint reuses the validated Phase 1 FP8 and Phase 2 "
            "MXFP4 tensors. ATOM selects dynamic FP8 A8W8 or MXFP4 A4W4 "
            "per runtime linear from `config.json`.\n"
        )
        manifest["output_model_bytes"] = (output / _MODEL_FILE).stat().st_size
        (output / _MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return manifest


def convert_dspark_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    profile_name: str = "phase1",
    quant_format: str = _PTPC_FP8,
    device: str = "auto",
    dry_run: bool = False,
    overwrite: bool = False,
    quantize_fn: Callable[[Tensor], tuple[Tensor, Tensor]] | None = None,
) -> dict[str, Any]:
    """Convert a BF16 Kimi-K3 DSpark checkpoint to an ATOM quant format."""

    source = Path(source_dir)
    output = Path(output_dir)
    model_path = source / _MODEL_FILE
    config_path = source / _CONFIG_FILE
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Expected {_MODEL_FILE} and {_CONFIG_FILE} under {source}"
        )

    config = json.loads(config_path.read_text())
    profile = get_ptq_profile(profile_name)
    canonical_quantize_fn = get_quantize_fn(quant_format)
    resolved_quantize_fn = quantize_fn or canonical_quantize_fn
    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        tensor_names = set(checkpoint.keys())
        selected = validate_dspark_checkpoint(config, tensor_names, profile)
        selected_set = set(selected)
        tensor_records = []
        for name in sorted(tensor_names):
            tensor_slice = checkpoint.get_slice(name)
            shape = tuple(tensor_slice.get_shape())
            tensor_records.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "source_dtype": str(tensor_slice.get_dtype()),
                    "action": "quantize" if name in selected_set else "preserve",
                }
            )
        source_metadata = checkpoint.metadata()

    manifest: dict[str, Any] = {
        "format_version": 1,
        "model_architecture": _SUPPORTED_ARCHITECTURE,
        "profile": profile.name,
        "quant_format": quant_format,
        "weight_quantization": (
            "fp8_e4m3_per_output_channel"
            if quant_format == _PTPC_FP8
            else "mxfp4_e2m1_packed_per_1x32"
        ),
        "activation_quantization": (
            "dynamic_per_token_fp8_at_atom_runtime"
            if quant_format == _PTPC_FP8
            else "dynamic_per_token_mxfp4_at_atom_runtime_a4w4"
        ),
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "selected_weight_count": len(selected),
        "selected_weights": list(selected),
        "tensors": tensor_records,
    }
    if dry_run:
        return manifest

    target_device = _resolve_device(
        device,
        require_accelerator=quantize_fn is None,
    )
    _prepare_output_dir(source, output, overwrite)
    output_tensors: dict[str, Tensor] = {}
    selected_stats: dict[str, dict[str, Any]] = {}
    try:
        with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
            for name in sorted(tensor_names):
                tensor = checkpoint.get_tensor(name)
                if name not in selected_set:
                    output_tensors[name] = tensor
                    continue
                quantized, scale = resolved_quantize_fn(tensor.to(target_device))
                quantized_cpu = quantized.cpu()
                scale_cpu = scale.cpu()
                output_tensors[name] = quantized_cpu
                output_tensors[weight_scale_name(name)] = scale_cpu
                selected_stats[name] = {
                    "source_bytes": _tensor_nbytes(tensor),
                    "quantized_bytes": _tensor_nbytes(quantized_cpu),
                    "scale_bytes": _tensor_nbytes(scale_cpu),
                    "quantized_dtype": str(quantized_cpu.dtype),
                    "scale_dtype": str(scale_cpu.dtype),
                    "scale_shape": list(scale_cpu.shape),
                }
                del tensor, quantized, scale, quantized_cpu, scale_cpu
                torch.cuda.empty_cache()

        save_file(
            output_tensors,
            output / _MODEL_FILE,
            metadata=source_metadata or {"format": "pt"},
        )
        output_config = dict(config)
        output_config["quantization_config"] = build_quark_quant_config(
            profile,
            quant_format,
        )
        output_config["atom_dspark_ptq_profile"] = profile.name
        output_config["atom_dspark_ptq_format"] = quant_format
        (output / _CONFIG_FILE).write_text(
            json.dumps(output_config, indent=2, sort_keys=True) + "\n"
        )
        _copy_checkpoint_assets(source, output)
        manifest["selected_weight_stats"] = selected_stats
        manifest["source_model_bytes"] = model_path.stat().st_size
        manifest["output_model_bytes"] = (output / _MODEL_FILE).stat().st_size
        (output / _MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return manifest

