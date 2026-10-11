import json
from pathlib import Path

import pytest

from multi_tune_agent.sft_dataset import (
    DATASET_MANIFEST_SCHEMA,
    SAMPLE_SCHEMA,
    DatasetError,
    compact_hash,
    sha256_bytes,
    sha256_file,
)
from multi_tune_agent import sft_training_export as exporter
from multi_tune_agent.sft_training_export import (
    EXPORT_MANIFEST_SCHEMA,
    EXPORT_SCHEMA,
    canonical_messages,
    export_qwen_sft,
    tokenize_training_record,
)


class FakeTokenizer:
    """A deterministic character tokenizer with static chat framing."""

    chat_template = "fake-qwen-template-v1"
    init_kwargs = {"_commit_hash": "resolved-commit"}

    role_tokens = {"system": 10, "user": 20, "assistant": 30}
    bos_token = 1
    end_token = 2

    @staticmethod
    def content_ids(content: str) -> list[int]:
        return [1000 + ord(character) for character in content]

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt=False
    ):
        assert tokenize is True
        result = [self.bos_token]
        for message in messages:
            result.append(self.role_tokens[message["role"]])
            result.extend(self.content_ids(message["content"]))
            result.append(self.end_token)
        if add_generation_prompt:
            result.append(self.role_tokens["assistant"])
        return result


def _sample(sample_id: str = "sample-b", patch: str = "--- a/k\n+++ b/k\n") -> dict:
    return {
        "schema_version": SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "task_type": "cold_start",
        "split": "train",
        "input": {
            "contract": {"operator": "gemm", "shape": [16, 16]},
            "parent_source": {"kernel.py": "old\n"},
            "baseline": {"case": 1.0},
            "profile": None,
        },
        "output": {"patch": patch},
        "provenance": {
            "lumen_git_sha": "lumen-sha",
            "lumen_working_state_sha256": "lumen-state",
            "geak_git_sha": "geak-sha",
            "geak_working_state_sha256": "geak-state",
        },
    }


def _write_source(tmp_path: Path, samples: list[dict]) -> Path:
    processed = tmp_path / "canonical" / "processed"
    processed.mkdir(parents=True)
    data_path = processed / "train.jsonl"
    data_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in samples),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "dataset_version": "canonical-test-v1",
        "output_root": str(processed.parent),
        "input_manifest": str(tmp_path / "unused-input-manifest.json"),
        "files": {
            "processed/train.jsonl": {
                "sha256": sha256_file(data_path),
                "records": len(samples),
                "bytes": data_path.stat().st_size,
                "schema_version": SAMPLE_SCHEMA,
            }
        },
    }
    manifest["content_sha256"] = compact_hash(manifest)
    path = processed.parent / "dataset_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _passing_validation(monkeypatch):
    def validate(_manifest, report):
        value = {
            "schema_version": "geak_sft_quality_report_v1",
            "status": "pass",
            "error_count": 0,
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return value

    monkeypatch.setattr(exporter, "validate_dataset", validate)


def _template_hash(tokenizer: FakeTokenizer) -> str:
    return sha256_bytes(tokenizer.chat_template.encode())


def test_messages_are_canonical_and_assistant_is_verbatim_patch():
    patch = "--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-old\n+new\n"
    sample = _sample(patch=patch)

    first = canonical_messages(sample)
    second = canonical_messages(dict(reversed(list(sample.items()))))

    assert first == second
    assert first[-1] == {"role": "assistant", "content": patch}
    assert "FROZEN_INPUT=\n{" in first[1]["content"]
    assert "Return only the unified diff patch" in first[0]["content"]


def test_tokenization_masks_exactly_patch_content_tokens():
    tokenizer = FakeTokenizer()
    patch = "--- a/k\n+++ b/k\n+λ\n"

    record = tokenize_training_record(
        _sample(patch=patch), tokenizer, max_length=10_000
    )

    masked_ids = [
        token_id
        for token_id, mask in zip(record["input_ids"], record["loss_mask"])
        if mask
    ]
    assert masked_ids == tokenizer.content_ids(patch)
    assert record["messages"][-1]["content"] == patch
    assert record["token_stats"]["assistant_tokens"] == len(patch)
    assert record["token_stats"]["loss_mask_tokens"] == sum(record["loss_mask"])
    assert record["token_stats"]["overflow_tokens"] == 0
    assert len(record["attention_mask"]) == len(record["input_ids"])


def test_export_is_sorted_deterministic_and_pins_provenance(tmp_path, monkeypatch):
    _passing_validation(monkeypatch)
    tokenizer = FakeTokenizer()
    source = _write_source(tmp_path, [_sample("sample-b"), _sample("sample-a")])
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    kwargs = {
        "tokenizer_model": "Qwen/fake",
        "tokenizer_revision": "requested-revision",
        "expected_chat_template_sha256": _template_hash(tokenizer),
        "max_length": 10_000,
    }

    first = export_qwen_sft(source, first_root, tokenizer, **kwargs)
    second = export_qwen_sft(source, second_root, tokenizer, **kwargs)

    first_data = (first_root / "train.qwen.jsonl").read_bytes()
    assert first_data == (second_root / "train.qwen.jsonl").read_bytes()
    assert (first_root / "export_manifest.json").read_bytes() == (
        second_root / "export_manifest.json"
    ).read_bytes()
    rows = [json.loads(line) for line in first_data.splitlines()]
    assert [row["sample_id"] for row in rows] == ["sample-a", "sample-b"]
    assert all(row["schema_version"] == EXPORT_SCHEMA for row in rows)
    assert all(sum(row["loss_mask"]) == len(row["messages"][-1]["content"]) for row in rows)

    manifest = json.loads((first_root / "export_manifest.json").read_text())
    assert manifest["schema_version"] == EXPORT_MANIFEST_SCHEMA
    assert manifest["tokenizer"] == {
        "chat_template_sha256": _template_hash(tokenizer),
        "class": "FakeTokenizer",
        "model": "Qwen/fake",
        "requested_revision": "requested-revision",
        "resolved_revision": "resolved-commit",
        "transformers_version": manifest["tokenizer"]["transformers_version"],
    }
    assert manifest["provenance"]["geak_git_sha"] == ["geak-sha"]
    assert manifest["provenance"]["lumen_git_sha"] == ["lumen-sha"]
    assert manifest["token_stats"]["exported"]["samples"] == 2
    assert manifest["files"]["train.qwen.jsonl"]["sha256"] == sha256_bytes(first_data)
    assert first["exported"] == second["exported"] == 2


def test_overflow_errors_without_partial_training_export(tmp_path, monkeypatch):
    _passing_validation(monkeypatch)
    tokenizer = FakeTokenizer()
    source = _write_source(tmp_path, [_sample()])
    output = tmp_path / "error-output"

    with pytest.raises(DatasetError, match="exceed max_length"):
        export_qwen_sft(
            source,
            output,
            tokenizer,
            tokenizer_model="Qwen/fake",
            tokenizer_revision="revision",
            expected_chat_template_sha256=_template_hash(tokenizer),
            max_length=8,
        )

    assert not (output / "train.qwen.jsonl").exists()
    assert (output / "source_quality_report.json").is_file()


def test_overflow_can_be_quarantined_with_stats(tmp_path, monkeypatch):
    _passing_validation(monkeypatch)
    tokenizer = FakeTokenizer()
    source = _write_source(tmp_path, [_sample("overflow")])
    output = tmp_path / "quarantine-output"

    result = export_qwen_sft(
        source,
        output,
        tokenizer,
        tokenizer_model="Qwen/fake",
        tokenizer_revision="revision",
        expected_chat_template_sha256=_template_hash(tokenizer),
        max_length=8,
        overflow_policy="quarantine",
    )

    assert result["exported"] == 0
    assert result["quarantined"] == 1
    assert (output / "train.qwen.jsonl").read_text() == ""
    quarantined = json.loads((output / "train.overflow.jsonl").read_text())
    assert quarantined["reason"] == "context_overflow"
    assert quarantined["token_stats"]["overflow_tokens"] > 0
    manifest = json.loads((output / "export_manifest.json").read_text())
    assert manifest["max_length_policy"]["truncation"] == "forbidden"
    assert manifest["counts"] == {"exported": 0, "quarantined": 1, "source": 1}


def test_source_validation_and_template_hash_are_hard_gates(tmp_path, monkeypatch):
    tokenizer = FakeTokenizer()
    source = _write_source(tmp_path, [_sample()])
    invalid_output = tmp_path / "invalid"

    def fail_validation(_manifest, report):
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}\n")
        return {"status": "fail", "error_count": 3}

    monkeypatch.setattr(exporter, "validate_dataset", fail_validation)
    with pytest.raises(DatasetError, match="validation failed with 3 errors"):
        export_qwen_sft(
            source,
            invalid_output,
            tokenizer,
            tokenizer_model="Qwen/fake",
            tokenizer_revision="revision",
            expected_chat_template_sha256=_template_hash(tokenizer),
            max_length=10_000,
        )
    assert not (invalid_output / "train.qwen.jsonl").exists()

    _passing_validation(monkeypatch)
    with pytest.raises(DatasetError, match="chat template SHA256 mismatch"):
        export_qwen_sft(
            source,
            tmp_path / "bad-template",
            tokenizer,
            tokenizer_model="Qwen/fake",
            tokenizer_revision="revision",
            expected_chat_template_sha256="0" * 64,
            max_length=10_000,
        )
