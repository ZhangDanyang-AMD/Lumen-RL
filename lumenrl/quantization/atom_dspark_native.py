"""ATOM-native export for the validated Kimi-K3 DSpark Phase 1 FP8 model."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from lumenrl.quantization.atom_dspark_ptq import weight_scale_name

logger = logging.getLogger(__name__)

MODEL_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
MANIFEST_FILE = "atom_dspark_native_manifest.json"
SOURCE_ARCHITECTURE = "K3DSparkModel"
NATIVE_ARCHITECTURE = "AtomK3DSparkModel"
NATIVE_MODEL_TYPE = "atom_k3_dspark"
NATIVE_FORMAT = "atom_k3_dspark_fp8"
NATIVE_FORMAT_VERSION = 1
NATIVE_WEIGHT_LAYOUT = "logical_global_unshuffled"
NATIVE_FP8_DTYPE = "float8_e4m3fn"
NATIVE_FP8_STORAGE_DTYPE = "float8_e4m3fn"
NATIVE_SCALE_LAYOUT = "global_per_output_channel_fp32"

NATIVE_QUANT_EXCLUDES = (
    "context_proj",
    "layers.*.self_attn.context_kv_proj",
    "layers.*.self_attn.fused_qkv_a_proj",
)

MERGED_PROJECTIONS: dict[str, tuple[str, ...]] = {
    "gate_up_proj": ("gate_proj", "up_proj"),
    "fused_qkv_a_proj": ("q_a_proj", "kv_a_proj_with_mqa"),
}


def validate_e4m3fn_weight(
    weight: Tensor,
    weight_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    """Validate the native gfx950 E4M3FN weight and scale contract."""

    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"Expected an E4M3FN source weight, got dtype={weight.dtype}."
        )
    if weight_scale.dtype != torch.float32:
        raise ValueError(
            f"Expected FP32 per-channel scales, got dtype={weight_scale.dtype}."
        )
    expected_scale_shape = (weight.shape[0], 1)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"Scale shape {tuple(weight_scale.shape)} does not match "
            f"{expected_scale_shape} for weight shape {tuple(weight.shape)}."
        )

    return weight, weight_scale


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _validate_source_config(config: Mapping[str, Any]) -> int:
    if config.get("architectures") != [SOURCE_ARCHITECTURE]:
        raise ValueError(
            "ATOM-native export only accepts the portable K3DSparkModel source; "
            f"got architectures={config.get('architectures')!r}."
        )
    if config.get("atom_dspark_ptq_profile") != "phase1":
        raise ValueError(
            "ATOM-native export requires the validated Phase 1 profile; got "
            f"{config.get('atom_dspark_ptq_profile')!r}."
        )
    quant_config = config.get("quantization_config") or {}
    global_config = quant_config.get("global_quant_config") or {}
    weight_config = global_config.get("weight") or {}
    input_config = global_config.get("input_tensors") or {}
    if weight_config != {"qscheme": "per_channel", "dtype": "fp8_e4m3"}:
        raise ValueError(
            "ATOM-native export requires Phase 1 per-channel E4M3 weights; "
            f"got {weight_config!r}."
        )
    if input_config.get("is_dynamic") is not True:
        raise ValueError("Phase 1 activation quantization must remain dynamic.")
    num_layers = config.get("num_hidden_layers")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError(f"Invalid num_hidden_layers={num_layers!r}.")
    return num_layers


def _prepare_output_dir(source: Path, output: Path, overwrite: bool) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("Source and output checkpoint directories must differ.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output}; pass overwrite=True."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)


def _copy_assets(source: Path, output: Path) -> None:
    skipped = {
        MODEL_FILE,
        CONFIG_FILE,
        MANIFEST_FILE,
        "atom_dspark_ptq_manifest.json",
    }
    for path in source.iterdir():
        if path.name in skipped or path.suffix == ".safetensors":
            continue
        destination = output / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)


def _write_native_readme(output: Path) -> None:
    source_card = output / "README.md"
    source_card.write_text("""---
library_name: atom
base_model: Inferact/Kimi-K3-DSpark
pipeline_tag: text-generation
tags:
  - dspark
  - speculative-decoding
  - atom
  - fp8
---

# ATOM-native Kimi-K3 DSpark E4M3FN

This checkpoint is the ATOM-only, gfx950-native form of the Phase 1 FP8 PTPC
model. It stores global unshuffled tensors, pre-merges `gate_up_proj` and
`fused_qkv_a_proj`, and adds a BF16 `context_kv_proj`. Tensor-parallel slicing
and AITER preshuffle still happen at load time.

It requires `model_type=atom_k3_dspark` / `AtomK3DSparkModel` support from ATOM
and is intentionally not compatible with stock Transformers, vLLM, or SGLang.
""")


def _native_quant_config(source_config: Mapping[str, Any]) -> dict[str, Any]:
    quant_config = dict(source_config["quantization_config"])
    quant_config["exclude"] = list(NATIVE_QUANT_EXCLUDES)
    return quant_config


def _native_checkpoint_metadata() -> dict[str, Any]:
    return {
        "format": NATIVE_FORMAT,
        "format_version": NATIVE_FORMAT_VERSION,
        "weight_layout": NATIVE_WEIGHT_LAYOUT,
        "fp8_dtype": NATIVE_FP8_DTYPE,
        "fp8_storage_dtype": NATIVE_FP8_STORAGE_DTYPE,
        "scale_layout": NATIVE_SCALE_LAYOUT,
        "runtime_tp_slice": True,
        "runtime_preshuffle": True,
        "context_kv_proj": {
            "source": "kv_a_proj_with_mqa",
            "fused_rows": [1536, 2112],
            "shape": [576, 7168],
            "dtype": "bfloat16",
        },
        "merged_projections": {
            name: {"sources": list(sources), "axis": 0}
            for name, sources in MERGED_PROJECTIONS.items()
        },
    }


def _source_names_for_layer(layer_idx: int) -> dict[str, str]:
    prefix = f"layers.{layer_idx}"
    return {
        "gate": f"{prefix}.mlp.gate_proj.weight",
        "up": f"{prefix}.mlp.up_proj.weight",
        "gate_up": f"{prefix}.mlp.gate_up_proj.weight",
        "q_a": f"{prefix}.self_attn.q_a_proj.weight",
        "kv_a": f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
        "fused_qkv_a": f"{prefix}.self_attn.fused_qkv_a_proj.weight",
        "context_kv": f"{prefix}.self_attn.context_kv_proj.weight",
    }


def export_atom_native_dspark_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export a global, merged, E4M3FN ATOM-only DSpark checkpoint.

    The input must be the validated portable Phase 1 FP8 PTPC checkpoint.
    Tensor-parallel slicing and AITER preshuffle intentionally remain runtime
    operations.
    """

    source = Path(source_dir)
    output = Path(output_dir)
    model_path = source / MODEL_FILE
    config_path = source / CONFIG_FILE
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Expected {MODEL_FILE} and {CONFIG_FILE} under {source}."
        )

    source_config = _read_json(config_path)
    num_layers = _validate_source_config(source_config)
    removed_names: set[str] = set()
    native_names: set[str] = set()
    for layer_idx in range(num_layers):
        names = _source_names_for_layer(layer_idx)
        for key in ("gate", "up", "q_a", "kv_a"):
            removed_names.add(names[key])
        removed_names.update(
            {
                weight_scale_name(names["gate"]),
                weight_scale_name(names["up"]),
            }
        )
        native_names.update(
            {
                names["gate_up"],
                weight_scale_name(names["gate_up"]),
                names["fused_qkv_a"],
                names["context_kv"],
            }
        )

    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        source_tensor_names = set(checkpoint.keys())
        missing = sorted(removed_names - source_tensor_names)
        if missing:
            raise ValueError(
                "Phase 1 checkpoint is missing required merge source tensors: "
                f"{missing}."
            )
        conflicting = sorted(native_names & source_tensor_names)
        if conflicting:
            raise ValueError(
                "Source already contains ATOM-native tensor names: "
                f"{conflicting}."
            )
        source_metadata = checkpoint.metadata()

    output_config = dict(source_config)
    output_config["model_type"] = NATIVE_MODEL_TYPE
    output_config["architectures"] = [NATIVE_ARCHITECTURE]
    output_config["atom_dspark_ptq_format"] = "atom_native_ptpc_fp8"
    output_config["quantization_config"] = _native_quant_config(source_config)
    output_config["atom_native_checkpoint"] = _native_checkpoint_metadata()

    manifest: dict[str, Any] = {
        "format": NATIVE_FORMAT,
        "format_version": NATIVE_FORMAT_VERSION,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_profile": "phase1",
        "source_architecture": SOURCE_ARCHITECTURE,
        "architecture": NATIVE_ARCHITECTURE,
        "weight_layout": NATIVE_WEIGHT_LAYOUT,
        "fp8_dtype": NATIVE_FP8_DTYPE,
        "fp8_storage_dtype": NATIVE_FP8_STORAGE_DTYPE,
        "scale_layout": NATIVE_SCALE_LAYOUT,
        "runtime_tp_slice": True,
        "runtime_preshuffle": True,
        "merged_projections": output_config["atom_native_checkpoint"][
            "merged_projections"
        ],
        "context_kv_rows": [1536, 2112],
        "context_kv_shape": [576, 7168],
        "num_hidden_layers": num_layers,
    }
    if dry_run:
        return manifest

    _prepare_output_dir(source, output, overwrite)
    output_tensors: dict[str, Tensor] = {}
    tensor_records: list[dict[str, Any]] = []
    try:
        with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
            for name in sorted(source_tensor_names - removed_names):
                tensor = checkpoint.get_tensor(name)
                if tensor.dtype == torch.float8_e4m3fn:
                    scale_name = weight_scale_name(name)
                    if scale_name not in source_tensor_names:
                        raise ValueError(f"Missing FP8 scale tensor {scale_name}.")
                    scale = checkpoint.get_tensor(scale_name)
                    tensor, scale = validate_e4m3fn_weight(tensor, scale)
                    output_tensors[name] = tensor
                    output_tensors[scale_name] = scale
                elif name not in output_tensors:
                    output_tensors[name] = tensor

            for layer_idx in range(num_layers):
                names = _source_names_for_layer(layer_idx)
                gate = checkpoint.get_tensor(names["gate"])
                up = checkpoint.get_tensor(names["up"])
                gate_scale = checkpoint.get_tensor(weight_scale_name(names["gate"]))
                up_scale = checkpoint.get_tensor(weight_scale_name(names["up"]))
                gate_up, gate_up_scale = validate_e4m3fn_weight(
                    torch.cat((gate, up), dim=0),
                    torch.cat((gate_scale, up_scale), dim=0),
                )
                output_tensors[names["gate_up"]] = gate_up
                output_tensors[weight_scale_name(names["gate_up"])] = gate_up_scale

                q_a = checkpoint.get_tensor(names["q_a"])
                kv_a = checkpoint.get_tensor(names["kv_a"])
                output_tensors[names["fused_qkv_a"]] = torch.cat((q_a, kv_a), dim=0)
                output_tensors[names["context_kv"]] = kv_a.clone()

        for name, tensor in sorted(output_tensors.items()):
            tensor_records.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            )
        save_file(
            output_tensors,
            output / MODEL_FILE,
            metadata=source_metadata or {"format": "pt"},
        )
        (output / CONFIG_FILE).write_text(
            json.dumps(output_config, indent=2, sort_keys=True) + "\n"
        )
        _copy_assets(source, output)
        _write_native_readme(output)
        manifest["tensor_count"] = len(output_tensors)
        manifest["tensors"] = tensor_records
        manifest["source_model_bytes"] = model_path.stat().st_size
        manifest["output_model_bytes"] = (output / MODEL_FILE).stat().st_size
        (output / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    logger.info(
        "Exported ATOM-native K3 DSpark checkpoint with %d tensors to %s",
        len(output_tensors),
        output,
    )
    return manifest
