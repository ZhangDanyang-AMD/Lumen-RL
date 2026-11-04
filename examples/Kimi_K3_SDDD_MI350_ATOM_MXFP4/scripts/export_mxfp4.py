#!/usr/bin/env python3
"""Export a Lumen-RL DSpark checkpoint as ATOM-loadable MXFP4 weights.

The exported GEMM weights use packed E2M1 bytes with one uint8 E8M0 scale for
each 1x32 group, matching ATOM's ``per_1x32`` A4W4 path. Model/master weights in
the training checkpoint remain BF16/FP32; this export is inference-only and is
not resumable.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file


BLOCK_SIZE = 32
FORMAT_NAME = "atom_k3_dspark_mxfp4_v1"
SCALE_SUFFIX = ".weight_scale"
ATOM_EXCLUDE = [
    "context_proj",
    "layers.*.self_attn.fused_qkv_a_proj",
]


def atom_quantization_config() -> dict[str, object]:
    return {
        "quant_method": "quark",
        "global_quant_config": {
            "weight": {
                "qscheme": "per_group",
                "dtype": "fp4_e2m1",
            },
            "input_tensors": {"is_dynamic": True},
        },
        "exclude": ATOM_EXCLUDE,
    }


def scale_key_for(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"expected a .weight key, got {weight_key!r}")
    return weight_key.removesuffix(".weight") + SCALE_SUFFIX


def should_quantize_weight(
    name: str,
    tensor: torch.Tensor,
    include_patterns: tuple[str, ...],
) -> bool:
    return (
        name.endswith(".weight")
        and tensor.ndim == 2
        and tensor.is_floating_point()
        and any(fnmatch.fnmatchcase(name, pattern) for pattern in include_patterns)
    )


def quantize_model_state(
    model_state: Mapping[str, torch.Tensor],
    include_patterns: tuple[str, ...],
    quantize_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[OrderedDict[str, torch.Tensor], list[dict[str, object]], dict[str, int]]:
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    quantized: list[dict[str, object]] = []
    hits = {pattern: 0 for pattern in include_patterns}

    for name, value in model_state.items():
        if not should_quantize_weight(name, value, include_patterns):
            output[name] = value.contiguous()
            continue
        if value.shape[-1] % BLOCK_SIZE:
            raise ValueError(
                f"{name} reduction dim {value.shape[-1]} is not divisible by "
                f"MXFP4 block size {BLOCK_SIZE}"
            )

        packed, scale = quantize_fn(value)
        expected_packed_shape = (value.shape[0], value.shape[1] // 2)
        expected_scale_shape = (value.shape[0], value.shape[1] // BLOCK_SIZE)
        if tuple(packed.shape) != expected_packed_shape:
            raise RuntimeError(
                f"{name}: packed shape {tuple(packed.shape)}, expected "
                f"{expected_packed_shape}"
            )
        if tuple(scale.shape) != expected_scale_shape:
            raise RuntimeError(
                f"{name}: scale shape {tuple(scale.shape)}, expected "
                f"{expected_scale_shape}"
            )
        if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
            raise RuntimeError(
                f"{name}: exporter requires uint8 packed weight/E8M0 scale, got "
                f"{packed.dtype}/{scale.dtype}"
            )

        output[name] = packed.cpu().contiguous()
        output[scale_key_for(name)] = scale.cpu().contiguous()
        quantized.append(
            {
                "weight": name,
                "logical_shape": list(value.shape),
                "packed_shape": list(packed.shape),
                "scale": scale_key_for(name),
                "scale_shape": list(scale.shape),
            }
        )
        for pattern in include_patterns:
            if fnmatch.fnmatchcase(name, pattern):
                hits[pattern] += 1

    return output, quantized, hits


def _selection_from_training_yaml(path: Path) -> tuple[str, ...]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    training = document["quantization"]["training"]
    selection = training["mxfp4_quant_weight_selection"]
    if training.get("mxfp4") is not True or selection.get("mode") != "explicit":
        raise ValueError(
            "MXFP4 export requires training.mxfp4=true and explicit weight selection"
        )
    patterns = tuple(selection.get("include_quant_weight") or ())
    if not patterns:
        raise ValueError("include_quant_weight must be non-empty")
    return patterns


def _production_quantizer(device: torch.device):
    if device.type != "cuda":
        raise ValueError(
            "MXFP4 export requires --device cuda in the Lumen training image"
        )
    from lumen.ops.quantize.ops import convert_to_mxfp4

    def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight_device = weight.to(
            device=device, dtype=torch.bfloat16
        ).contiguous()
        packed, scale = convert_to_mxfp4(
            weight_device,
            block_size=BLOCK_SIZE,
            axis=-1,
            use_sr=False,
            swizzle_scale=False,
        )
        result = packed.view(torch.uint8).cpu(), scale.view(torch.uint8).cpu()
        del weight_device, packed, scale
        torch.cuda.empty_cache()
        return result

    return quantize


def _run_bf16_export(
    *,
    repo_root: Path,
    checkpoint: Path,
    base_model: Path,
    train_config: Path,
    output_dir: Path,
) -> None:
    command = [
        sys.executable,
        str(repo_root / "output" / "export_dspark_hf.py"),
        "--ckpt",
        str(checkpoint),
        "--base-model",
        str(base_model),
        "--output",
        str(output_dir),
        "--train-config",
        str(train_config),
    ]
    subprocess.run(command, check=True)


def export_checkpoint(
    *,
    checkpoint: Path,
    base_model: Path,
    train_config: Path,
    output_dir: Path,
    repo_root: Path,
    device: str = "cuda",
    force: bool = False,
) -> dict[str, object]:
    include_patterns = _selection_from_training_yaml(train_config)
    quantize_fn = _production_quantizer(torch.device(device))

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} already exists; pass --force")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f"{output_dir.name}.tmp-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    try:
        # The BF16 export is a same-container intermediate. Keeping it beside
        # the durable output puts thousands of recently closed shard files on
        # shared NFS, where TemporaryDirectory cleanup can race `.nfs*`
        # lifecycle and fail with ENOTEMPTY after an otherwise valid export.
        # Use node-local temporary storage; only the final MXFP4 staging
        # directory belongs on shared storage.
        temporary_root = Path(
            os.environ.get("MXFP4_EXPORT_TMPDIR", tempfile.gettempdir())
        )
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="dspark-bf16-export-", dir=temporary_root
        ) as temporary:
            bf16_dir = Path(temporary)
            _run_bf16_export(
                repo_root=repo_root,
                checkpoint=checkpoint,
                base_model=base_model,
                train_config=train_config,
                output_dir=bf16_dir,
            )

            index_path = bf16_dir / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(index["weight_map"].values()))
            output_weight_map: dict[str, str] = {}
            reports: list[dict[str, object]] = []
            pattern_hits = {pattern: 0 for pattern in include_patterns}
            total_size = 0

            for shard_name in shard_names:
                state = load_file(bf16_dir / shard_name, device="cpu")
                exported, quantized, hits = quantize_model_state(
                    state, include_patterns, quantize_fn
                )
                save_file(exported, staging / shard_name)
                for name, tensor in exported.items():
                    output_weight_map[name] = shard_name
                    total_size += tensor.numel() * tensor.element_size()
                reports.extend(quantized)
                for pattern, count in hits.items():
                    pattern_hits[pattern] += count
                del state, exported

            unmatched = [
                pattern for pattern, count in pattern_hits.items() if count == 0
            ]
            if unmatched:
                raise ValueError(
                    f"include_quant_weight patterns matched no exported weight: "
                    f"{unmatched}"
                )

            output_index = {
                "metadata": {
                    "total_parameters": index["metadata"]["total_parameters"],
                    "total_size": total_size,
                },
                "weight_map": dict(sorted(output_weight_map.items())),
            }
            (staging / "model.safetensors.index.json").write_text(
                json.dumps(output_index, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            config = json.loads(
                (bf16_dir / "config.json").read_text(encoding="utf-8")
            )
            config["quantization_config"] = atom_quantization_config()
            config["mxfp4_export"] = {
                "format": FORMAT_NAME,
                "weight_block_size": [1, BLOCK_SIZE],
                "weight_storage_dtype": "uint8_packed_e2m1",
                "weight_scale_dtype": "uint8_e8m0",
                "weight_scale_suffix": SCALE_SUFFIX,
            }
            (staging / "config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            shutil.copy2(bf16_dir / ".gitattributes", staging / ".gitattributes")

        manifest: dict[str, object] = {
            "format": FORMAT_NAME,
            "source_checkpoint": str(checkpoint),
            "train_config": str(train_config),
            "block_size": [1, BLOCK_SIZE],
            "weight_storage_dtype": "uint8_packed_e2m1",
            "weight_scale_dtype": "uint8_e8m0",
            "weight_scale_suffix": SCALE_SUFFIX,
            "activation_scheme": "dynamic",
            "resumable_training_checkpoint": False,
            "quantized_weights": reports,
            "include_pattern_hits": pattern_hits,
            "atom_exclude": ATOM_EXCLUDE,
        }
        (staging / "mxfp4_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    manifest = export_checkpoint(
        checkpoint=args.ckpt,
        base_model=args.base_model,
        train_config=args.train_config,
        output_dir=args.output,
        repo_root=repo_root,
        device=args.device,
        force=args.force,
    )
    print(
        f"Exported {len(manifest['quantized_weights'])} MXFP4 weights "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
