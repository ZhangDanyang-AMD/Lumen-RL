"""Turn specialist RL records into answer-masked SFT rows.

Only assistant turns whose advantage is positive are trained. Earlier context
stays in the sequence with a zero loss mask, so the update follows the turns
that beat the role's leave-one-out baseline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


MANIFEST_SCHEMA = "geak_training_data_manifest_v2"


class ChatTokenizer(Protocol):
    """Enough of a Hugging Face tokenizer to render a tool-calling chat."""

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
    ) -> Any:
        """Return token ids for ``messages``."""


def positive_assistant_indices(sample: Mapping[str, Any], min_advantage: float) -> list[int]:
    """Assistant-message indices whose step advantage clears ``min_advantage``."""
    chosen: list[int] = []
    for step in sample["steps"]:
        if float(step["advantage"]) > min_advantage:
            index = int(step["assistant_message_index"])
            if index not in chosen:
                chosen.append(index)
    return chosen


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict) and "input_ids" in value:
        value = value["input_ids"]
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


SCHEMA_KEY_ORDER = ("type", "function", "name", "description", "parameters", "properties", "enum", "items",
                    "required", "additionalProperties")


def _ordered(keys: Sequence[str], first: Sequence[str]) -> list[str]:
    rank = {key: i for i, key in enumerate(first)}
    return sorted(keys, key=lambda k: (rank.get(k, len(rank)), list(keys).index(k)))


def canonical_schema(value: Any) -> Any:
    """Restore the live key order of a tool schema after a ``sort_keys`` round trip.

    The chat template renders schema keys in dict order, so training prompts
    must match what rollouts send: ``properties`` follow ``required`` order and
    the other keys follow ``SCHEMA_KEY_ORDER``.
    """
    if isinstance(value, list):
        return [canonical_schema(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    out: dict[str, Any] = {}
    for key in _ordered(list(value), SCHEMA_KEY_ORDER):
        item = value[key]
        if key == "properties" and isinstance(item, Mapping):
            required = [k for k in value.get("required") or () if k in item]
            item = {k: canonical_schema(item[k]) for k in _ordered(list(item), required)}
        else:
            item = canonical_schema(item)
        out[key] = item
    return out


def _argument_order(tools: Sequence[Mapping[str, Any]] | None) -> dict[str, list[str]]:
    order: dict[str, list[str]] = {}
    for tool in tools or ():
        function = tool.get("function") or {}
        parameters = function.get("parameters") or {}
        order[str(function.get("name"))] = list(parameters.get("required") or ())
    return order


def _template_messages(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Copy messages into the mapping form Qwen's chat template renders.

    Stored trajectories keep OpenAI tool-call arguments as JSON strings. The
    Qwen3 template iterates those arguments as a mapping, in the tool's
    ``required`` order (the order live rollouts emit).
    """
    argument_order = _argument_order(tools)
    rendered: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        calls = copied.get("tool_calls")
        if calls:
            normalized = []
            for call in calls:
                call = dict(call)
                function = dict(call.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments) if arguments else {}
                if isinstance(arguments, Mapping):
                    first = argument_order.get(str(function.get("name")), [])
                    arguments = {k: arguments[k] for k in _ordered(list(arguments), first)}
                function["arguments"] = arguments
                call["function"] = function
                normalized.append(call)
            copied["tool_calls"] = normalized
        rendered.append(copied)
    return rendered


def _render(
    tokenizer: ChatTokenizer,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
) -> list[int]:
    tools = canonical_schema(list(tools)) if tools else None
    rendered = tokenizer.apply_chat_template(
        _template_messages(messages, tools), tools=tools, add_generation_prompt=False, tokenize=True,
    )
    return _token_ids(rendered)


def masked_turn(
    tokenizer: ChatTokenizer,
    sample: Mapping[str, Any],
    assistant_index: int,
) -> dict[str, list[int]]:
    """One sequence whose loss is restricted to ``assistant_index``.

    Raises:
        ValueError: When the chat template is not prefix-stable or the turn has no target tokens.
    """
    messages = sample["messages"][: assistant_index + 1]
    tools = sample.get("tools") or None
    ids: list[int] = []
    mask: list[int] = []
    for index, message in enumerate(messages):
        prefix = _render(tokenizer, messages[:index], tools) if index else []
        full = _render(tokenizer, messages[: index + 1], tools)
        if full[: len(prefix)] != prefix:
            raise ValueError(f"chat template is not prefix-stable at message {index}")
        piece = full[len(prefix) :]
        trainable = message.get("role") == "assistant" and index == assistant_index
        ids.extend(piece)
        mask.extend([1 if trainable else 0] * len(piece))
    if not any(mask[1:]):
        raise ValueError(f"assistant message {assistant_index} produced no trainable tokens")
    return {"input_ids": ids, "loss_mask": mask}


def write_role_sft(
    samples: Sequence[Mapping[str, Any]],
    tokenizer: ChatTokenizer,
    out_dir: Path,
    *,
    min_advantage: float,
    max_sequence_length: int,
    tokenizer_sha256: str,
    tokenizer_path: str,
) -> dict[str, Any]:
    """Write the tokenized JSONL and the pinned data manifest.

    Returns:
        Manifest contents, including how many turns were kept and skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, list[int]]] = []
    skipped_long = 0
    for sample in samples:
        for index in positive_assistant_indices(sample, min_advantage):
            row = masked_turn(tokenizer, sample, index)
            if len(row["input_ids"]) > max_sequence_length:
                skipped_long += 1
                continue
            rows.append(row)
    tokenized = out_dir / "train.tokenized.jsonl"
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    tokenized.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": {
            "train": {
                "path": tokenized.name,
                "rows": len(rows),
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            },
        },
        "source": {"selection": f"specialist turns with advantage > {min_advantage}"},
        "max_length": max_sequence_length,
        "observed_max_length": max((len(row["input_ids"]) for row in rows), default=0),
        "skipped_long_rows": skipped_long,
        "tokenizer": {"path": tokenizer_path, "sha256": tokenizer_sha256},
    }
    manifest_path = out_dir / "manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(encoded)
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    manifest["manifest_path"] = str(manifest_path)
    manifest["train_path"] = str(tokenized)
    return manifest
