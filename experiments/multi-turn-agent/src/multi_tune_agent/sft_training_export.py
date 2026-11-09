"""Deterministic GEAK Kernel SFT export for Qwen-compatible training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .sft_dataset import (
    SAMPLE_SCHEMA,
    DatasetError,
    _dataset_files,
    compact_hash,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    validate_dataset,
    write_json,
    write_jsonl,
)


EXPORT_TOOL_VERSION = "geak_qwen_sft_export_v1"
EXPORT_SCHEMA = "geak_qwen_messages_v1"
EXPORT_MANIFEST_SCHEMA = "geak_qwen_sft_export_manifest_v1"
OVERFLOW_SCHEMA = "geak_qwen_sft_overflow_v1"
PROMPT_VERSION = "geak_kernel_qwen_prompt_v1"

SYSTEM_PROMPT = (
    "You optimize AMD GPU kernels from a frozen, validated GEAK input. "
    "Preserve the contract and change only implementation source. "
    "Return only the unified diff patch; do not add prose or code fences."
)

TASK_INSTRUCTIONS = {
    "cold_start": "Optimize independently from the frozen contract and baseline.",
    "profile_guided": "Optimize using only the frozen parent profile.",
    "direction_conditioned": "Implement the frozen optimization direction.",
    "error_recovery": "Repair the source using the exact frozen error feedback.",
    "regression_balance": (
        "Remove the frozen per-case regressions while preserving correctness."
    ),
}


def canonical_messages(sample: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the stable two-message prompt and verbatim patch completion."""
    task_type = str(sample.get("task_type") or "")
    try:
        instruction = TASK_INSTRUCTIONS[task_type]
    except KeyError as exc:
        raise DatasetError(f"unsupported task_type for training export: {task_type}") from exc
    input_value = sample.get("input")
    output = sample.get("output")
    if not isinstance(input_value, Mapping):
        raise DatasetError(f"{sample.get('sample_id')}: input must be an object")
    if not isinstance(output, Mapping) or not isinstance(output.get("patch"), str):
        raise DatasetError(f"{sample.get('sample_id')}: output.patch must be a string")
    patch = output["patch"]
    if not patch:
        raise DatasetError(f"{sample.get('sample_id')}: output.patch is empty")
    user = (
        f"PROMPT_VERSION={PROMPT_VERSION}\n"
        f"TASK_TYPE={task_type}\n"
        f"INSTRUCTION={instruction}\n"
        "FROZEN_INPUT=\n"
        + json.dumps(input_value, indent=2, sort_keys=True, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": patch},
    ]


def _normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(
        value[0], (list, tuple)
    ):
        value = list(value[0])
    if not isinstance(value, list):
        raise DatasetError("tokenizer returned unsupported token IDs")
    return [int(item.item() if hasattr(item, "item") else item) for item in value]


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    try:
        value = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as exc:
        raise DatasetError(f"tokenizer chat template failed: {exc}") from exc
    return _normalize_token_ids(value)


def _content_loss_mask(full_ids: list[int], empty_ids: list[int]) -> list[int]:
    """Mask exactly the token span introduced by assistant content.

    Comparing the full rendering with an empty assistant rendering excludes the
    static assistant header and end-of-turn tokens while preserving any
    context-sensitive token at the content boundary.
    """
    prefix = 0
    limit = min(len(full_ids), len(empty_ids))
    while prefix < limit and full_ids[prefix] == empty_ids[prefix]:
        prefix += 1
    suffix = 0
    suffix_limit = min(len(full_ids) - prefix, len(empty_ids) - prefix)
    while (
        suffix < suffix_limit
        and full_ids[len(full_ids) - suffix - 1] == empty_ids[len(empty_ids) - suffix - 1]
    ):
        suffix += 1
    end = len(full_ids) - suffix
    if end <= prefix:
        raise DatasetError("assistant patch produced no distinct chat-template tokens")
    return [0] * prefix + [1] * (end - prefix) + [0] * suffix


def tokenize_training_record(
    sample: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[str, Any]:
    """Create one messages record with an explicit patch-only loss mask."""
    if max_length < 1:
        raise DatasetError("max_length must be positive")
    messages = canonical_messages(sample)
    prompt_ids = _apply_chat_template(
        tokenizer, messages[:-1], add_generation_prompt=True
    )
    full_ids = _apply_chat_template(
        tokenizer, messages, add_generation_prompt=False
    )
    empty_messages = [*messages[:-1], {"role": "assistant", "content": ""}]
    empty_ids = _apply_chat_template(
        tokenizer, empty_messages, add_generation_prompt=False
    )
    loss_mask = _content_loss_mask(full_ids, empty_ids)
    loss_tokens = sum(loss_mask)
    stats = {
        "prompt_tokens": len(prompt_ids),
        "assistant_tokens": loss_tokens,
        "loss_mask_tokens": loss_tokens,
        "non_loss_tokens": len(full_ids) - loss_tokens,
        "total_tokens": len(full_ids),
        "max_length": max_length,
        "overflow_tokens": max(0, len(full_ids) - max_length),
    }
    return {
        "schema_version": EXPORT_SCHEMA,
        "sample_id": sample["sample_id"],
        "messages": messages,
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "loss_mask": loss_mask,
        "token_stats": stats,
        "source": {
            "schema_version": sample.get("schema_version"),
            "dataset_split": sample.get("split"),
            "task_type": sample.get("task_type"),
            "provenance": sample.get("provenance"),
        },
    }


def _chat_template(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        getter = getattr(tokenizer, "get_chat_template", None)
        if callable(getter):
            template = getter()
    if not isinstance(template, str) or not template:
        raise DatasetError("tokenizer has no pinned chat template")
    return template


def _tokenizer_manifest(
    tokenizer: Any,
    model: str,
    revision: str,
    expected_chat_template_sha256: str,
) -> dict[str, Any]:
    if not model or not revision:
        raise DatasetError("tokenizer model and revision must be non-empty")
    template_hash = sha256_bytes(_chat_template(tokenizer).encode("utf-8"))
    if template_hash != expected_chat_template_sha256:
        raise DatasetError(
            "chat template SHA256 mismatch: "
            f"expected {expected_chat_template_sha256}, got {template_hash}"
        )
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    resolved = (
        init_kwargs.get("_commit_hash")
        if isinstance(init_kwargs, Mapping)
        else None
    ) or getattr(tokenizer, "_commit_hash", None) or revision
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    return {
        "model": model,
        "requested_revision": revision,
        "resolved_revision": str(resolved),
        "class": type(tokenizer).__name__,
        "transformers_version": transformers_version,
        "chat_template_sha256": template_hash,
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "prompt_tokens",
        "assistant_tokens",
        "loss_mask_tokens",
        "non_loss_tokens",
        "total_tokens",
    )
    result: dict[str, Any] = {"samples": len(records)}
    for field in fields:
        values = [int(record["token_stats"][field]) for record in records]
        result[field] = {
            "total": sum(values),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": (sum(values) / len(values)) if values else 0.0,
        }
    return result


def _provenance_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    keys = (
        "lumen_git_sha",
        "lumen_working_state_sha256",
        "geak_git_sha",
        "geak_working_state_sha256",
    )
    return {
        key: sorted(
            {
                str(sample.get("provenance", {}).get(key))
                for sample in samples
                if isinstance(sample.get("provenance"), Mapping)
                and sample["provenance"].get(key) not in (None, "")
            }
        )
        for key in keys
    }


def _load_split_samples(
    manifest: Mapping[str, Any], root: Path, split: str
) -> list[Mapping[str, Any]]:
    samples: list[Mapping[str, Any]] = []
    for relative, metadata in sorted(manifest["files"].items()):
        if (
            isinstance(metadata, Mapping)
            and metadata.get("schema_version") == SAMPLE_SCHEMA
            and Path(str(relative)).stem == split
        ):
            records, errors = read_jsonl(root / str(relative))
            if errors:
                raise DatasetError(f"{relative}: source JSONL could not be parsed")
            samples.extend(records)
    sample_ids = [str(sample.get("sample_id") or "") for sample in samples]
    if any(not sample_id for sample_id in sample_ids):
        raise DatasetError(f"{split}: source sample is missing sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise DatasetError(f"{split}: duplicate source sample_id")
    return sorted(samples, key=lambda sample: str(sample["sample_id"]))


def _write_parquet(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetError(
            "Parquet requested but pyarrow is not installed; JSONL export is available"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    table = pa.Table.from_pylist(list(records))
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)
    return sha256_file(path)


def export_qwen_sft(
    source_manifest_path: Path,
    output_root: Path,
    tokenizer: Any,
    *,
    tokenizer_model: str,
    tokenizer_revision: str,
    expected_chat_template_sha256: str,
    max_length: int,
    split: str = "train",
    overflow_policy: str = "error",
    parquet: bool = False,
) -> dict[str, Any]:
    """Validate canonical data, then export deterministic training artifacts."""
    if overflow_policy not in {"error", "quarantine"}:
        raise DatasetError("overflow_policy must be error or quarantine")
    source_manifest_path = source_manifest_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    validation_path = output_root / "source_quality_report.json"
    quality = validate_dataset(source_manifest_path, validation_path)
    if quality["status"] != "pass":
        raise DatasetError(
            f"source dataset validation failed with {quality['error_count']} errors"
        )
    source_manifest, source_root = _dataset_files(source_manifest_path)
    samples = _load_split_samples(source_manifest, source_root, split)
    if not samples:
        raise DatasetError(f"source dataset has no samples for split {split!r}")
    tokenizer_info = _tokenizer_manifest(
        tokenizer,
        tokenizer_model,
        tokenizer_revision,
        expected_chat_template_sha256,
    )

    accepted: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    for sample in samples:
        record = tokenize_training_record(sample, tokenizer, max_length=max_length)
        if record["token_stats"]["overflow_tokens"]:
            overflow.append(
                {
                    "schema_version": OVERFLOW_SCHEMA,
                    "sample_id": record["sample_id"],
                    "reason": "context_overflow",
                    "token_stats": record["token_stats"],
                }
            )
        else:
            accepted.append(record)
    if overflow and overflow_policy == "error":
        raise DatasetError(
            f"{len(overflow)} sample(s) exceed max_length={max_length}; "
            "rerun with --overflow-policy quarantine to isolate them"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    jsonl_path = output_root / f"{split}.qwen.jsonl"
    digest = write_jsonl(jsonl_path, accepted)
    files[jsonl_path.name] = {
        "sha256": digest,
        "bytes": jsonl_path.stat().st_size,
        "records": len(accepted),
        "schema_version": EXPORT_SCHEMA,
    }
    if overflow_policy == "quarantine":
        quarantine_path = output_root / f"{split}.overflow.jsonl"
        digest = write_jsonl(quarantine_path, overflow)
        files[quarantine_path.name] = {
            "sha256": digest,
            "bytes": quarantine_path.stat().st_size,
            "records": len(overflow),
            "schema_version": OVERFLOW_SCHEMA,
        }
    if parquet:
        parquet_path = output_root / f"{split}.qwen.parquet"
        digest = _write_parquet(parquet_path, accepted)
        files[parquet_path.name] = {
            "sha256": digest,
            "bytes": parquet_path.stat().st_size,
            "records": len(accepted),
            "schema_version": EXPORT_SCHEMA,
        }
    files[validation_path.name] = {
        "sha256": sha256_file(validation_path),
        "bytes": validation_path.stat().st_size,
        "schema_version": quality["schema_version"],
    }

    manifest = {
        "schema_version": EXPORT_MANIFEST_SCHEMA,
        "tool_version": EXPORT_TOOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "source": {
            "manifest": str(source_manifest_path),
            "manifest_sha256": sha256_file(source_manifest_path),
            "dataset_version": source_manifest["dataset_version"],
            "schema_version": source_manifest["schema_version"],
            "split": split,
            "samples": len(samples),
        },
        "tokenizer": tokenizer_info,
        "max_length_policy": {
            "max_length": max_length,
            "overflow_policy": overflow_policy,
            "truncation": "forbidden",
        },
        "counts": {
            "source": len(samples),
            "exported": len(accepted),
            "quarantined": len(overflow),
        },
        "token_stats": {
            "exported": _aggregate(accepted),
            "overflow": _aggregate(
                [
                    {"token_stats": item["token_stats"]}
                    for item in overflow
                ]
            ),
        },
        "provenance": _provenance_summary(samples),
        "files": dict(sorted(files.items())),
    }
    manifest["content_sha256"] = compact_hash(manifest)
    manifest_path = output_root / "export_manifest.json"
    manifest_sha = write_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "sha256": manifest_sha,
        "exported": len(accepted),
        "quarantined": len(overflow),
        "files": files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export validated GEAK Kernel SFT data as Qwen messages JSONL"
    )
    parser.add_argument("--source-manifest", "--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--expected-chat-template-sha256", required=True)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--overflow-policy", choices=("error", "quarantine"), default="error"
    )
    parser.add_argument("--parquet", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        parser.error(
            "transformers is required to load the tokenizer; install it in the "
            "training environment"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_model,
            revision=args.tokenizer_revision,
            trust_remote_code=args.trust_remote_code,
        )
        result = export_qwen_sft(
            args.source_manifest,
            args.output_root,
            tokenizer,
            tokenizer_model=args.tokenizer_model,
            tokenizer_revision=args.tokenizer_revision,
            expected_chat_template_sha256=args.expected_chat_template_sha256,
            max_length=args.max_length,
            split=args.split,
            overflow_policy=args.overflow_policy,
            parquet=args.parquet,
        )
    except DatasetError as exc:
        parser.error(str(exc))
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
