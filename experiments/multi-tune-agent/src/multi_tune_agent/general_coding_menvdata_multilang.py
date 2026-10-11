"""Freeze and verify pinned MEnvData-SWE non-C++ replay candidates."""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests

from .general_coding_menvdata import (
    DATASET_ID,
    DATASET_REVISION,
    FULL_COMMIT,
    SOURCE_FILE,
    SOURCE_FILE_SHA256,
    SOURCE_ROWS,
    _image_tag,
)
from .general_coding_quota_override import (
    _docker_hub_digest,
    _exclusion_dimensions,
    _normalized_repository,
    overlap_reasons,
)
from .general_coding_rebench import (
    _normalized_patch,
    _supplemental_exclusions,
    _write_frozen_jsonl,
)
from .general_coding_replay import (
    PERMISSIVE_SOURCE_LICENSES,
    _docker_image_identity,
    _patch_paths,
    atomic_write,
    canonical_json,
    derive_repository_task_type,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verify_candidates,
    write_json,
    write_jsonl,
)


SOURCE_ID = "menvdata_swe_multilang"
TARGET_LANGUAGES = {
    "Go": "go",
    "JavaScript": "javascript",
    "Python": "python",
    "Rust": "rust",
    "TypeScript": "typescript",
}
DEFAULT_PER_LANGUAGE = 50
TARGET_ROWS = DEFAULT_PER_LANGUAGE * len(TARGET_LANGUAGES)
WAVE2_TARGET_ROWS = 209
WAVE2_REPOSITORY_CAP = 15
MAX_PREPARE_IMAGES = 25
LICENSE_FILENAMES = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "COPYING",
    "COPYING.md",
    "COPYRIGHT",
)
TEST_TOOL_PATTERNS = {
    "python": re.compile(
        r"(?i)(?:^|[;&|]\s*)(?:python(?:3)?\s+-m\s+(?:pytest|unittest)|pytest)\b"
    ),
    "go": re.compile(r"(?i)(?:^|[;&|]\s*)go\s+test\b"),
    "javascript": re.compile(
        r"(?i)(?:^|[;&|]\s*)(?:(?:npm|yarn|pnpm)\s+(?:run\s+)?test\b|"
        r"(?:npx\s+)?(?:jest|vitest|mocha)\b|node\s+.*(?:test|spec))"
    ),
    "typescript": re.compile(
        r"(?i)(?:^|[;&|]\s*)(?:(?:npm|yarn|pnpm)\s+(?:run\s+)?test\b|"
        r"(?:npx\s+)?(?:jest|vitest|mocha)\b|(?:bun|deno)\s+test\b)"
    ),
    "rust": re.compile(r"(?i)(?:^|[;&|]\s*)cargo\s+(?:nextest\s+run|test)\b"),
}
FORBIDDEN_COMMAND = re.compile(
    r"(?i)(?:\b(?:curl|wget|apt(?:-get)?|pip|git|docker)\b|"
    r"https?://|--network|\|\s*(?:sh|bash)\b)"
)
LICENSE_MARKERS = (
    (
        "Apache-2.0",
        ("apache license", "version 2.0", "limitations under the license"),
    ),
    (
        "MIT",
        ("permission is hereby granted, free of charge", "the software is provided"),
    ),
    (
        "BSD-3-Clause",
        (
            "redistribution and use in source and binary forms",
            "neither the name",
            "this software is provided",
        ),
    ),
    (
        "BSD-2-Clause",
        (
            "redistribution and use in source and binary forms",
            "this software is provided",
        ),
    ),
    (
        "0BSD",
        (
            "permission to use, copy, modify, and/or distribute this software",
            "for any purpose with or without fee",
            "the software is provided",
        ),
    ),
    (
        "ISC",
        (
            "permission to use, copy, modify, and/or distribute this software",
            "the software is provided",
        ),
    ),
    (
        "BSL-1.0",
        ("boost software license", "version 1.0"),
    ),
    (
        "PSF-2.0",
        ("python software foundation license", "psf license agreement"),
    ),
)


def _strip_heredocs(script: str) -> list[str]:
    """Return shell lines excluding embedded patch/heredoc payloads."""
    output: list[str] = []
    terminator: str | None = None
    heredoc = re.compile(r"<<-?['\"]?([A-Za-z0-9_]+)['\"]?")
    for line in script.replace("\r\n", "\n").splitlines():
        if terminator is not None:
            if line.strip() == terminator:
                terminator = None
            continue
        match = heredoc.search(line)
        if match:
            terminator = match.group(1)
            continue
        output.append(line)
    if terminator is not None:
        raise ValueError("unterminated heredoc in eval script")
    return output


def _logical_shell_lines(script: str) -> list[str]:
    logical: list[str] = []
    pending = ""
    for raw in _strip_heredocs(script):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical.append(pending)
        pending = ""
    if pending:
        logical.append(pending)
    return logical


def derive_targeted_command(eval_script: str, language: str) -> str:
    """Derive the narrowest fail-closed test command from a pinned eval script."""
    pattern = TEST_TOOL_PATTERNS[language]
    candidates = []
    for line in _logical_shell_lines(eval_script):
        if not pattern.search(line):
            continue
        if line.startswith(("echo ", "printf ")):
            continue
        if FORBIDDEN_COMMAND.search(line):
            continue
        if re.search(r"(?:\|\||&&)\s*true(?:\s|$)", line):
            continue
        line = re.sub(r"\s*\|\s*tee(?:\s+-[A-Za-z]+)*\s+\S+\s*$", "", line)
        line = re.sub(r";?\s*(?:rc|exit_code|test_rc|final_rc)\w*\s*=\s*\$\?\s*$", "", line)
        if line:
            candidates.append(line)
    if not candidates:
        raise ValueError("no offline language-specific test command in eval script")

    def score(command: str) -> tuple[int, int, str]:
        broad = bool(
            re.search(
                r"(?:\./\.\.\.|--all(?:-targets|-features)?\b|cargo\s+test\s*$)",
                command,
            )
        )
        targeted = bool(
            re.search(
                r"(?:[/\\](?:test|tests|spec)[/\\]|::|--test\b|-run\b|"
                r"--test-name-pattern\b|\.test\.|\.spec\.)",
                command,
                re.IGNORECASE,
            )
        )
        return (2 if targeted else 1, 0 if broad else 1, command)

    selected = max(candidates, key=score)
    if FORBIDDEN_COMMAND.search(selected):
        raise ValueError("derived command contains forbidden operation")
    return f"set -euo pipefail; {selected}"


def _derive_environment_prelude(eval_script: str) -> list[str]:
    """Extract only inert environment setup needed by the pinned test command."""
    prelude: list[str] = []
    for line in _logical_shell_lines(eval_script):
        if re.fullmatch(
            r"source\s+/opt/miniconda3/etc/profile\.d/conda\.sh", line
        ) or re.fullmatch(r"conda\s+activate\s+[A-Za-z0-9_.-]+", line):
            prelude.append(line)
            continue
        if re.fullmatch(
            r"export\s+[A-Za-z_][A-Za-z0-9_]*="
            r"(?:\"[^\"`$()]*?(?:\$[A-Za-z_][A-Za-z0-9_]*)?[^\"`$()]*?\"|"
            r"'[^'`$()]*'|[A-Za-z0-9_./:+-]+)",
            line,
        ):
            prelude.append(line)
    if any(line.startswith("conda activate ") for line in prelude) and not any(
        line.startswith("source /opt/miniconda3/") for line in prelude
    ):
        raise ValueError("conda activation lacks pinned initialization prelude")
    return prelude


def _classify_license(content: bytes) -> str:
    text = content.decode("utf-8", errors="replace").lower()
    for spdx, markers in LICENSE_MARKERS:
        if all(marker in text for marker in markers):
            if spdx == "BSD-2-Clause" and "neither the name" in text:
                continue
            return spdx
    raise ValueError("license text is not recognized as a supported permissive license")


def adapt_menvdata_multilang_row(
    row: Mapping[str, Any], *, source_sha256: str = SOURCE_FILE_SHA256
) -> dict[str, Any]:
    source_language = str(row.get("language", ""))
    language = TARGET_LANGUAGES.get(source_language)
    if language is None:
        raise ValueError("row language is not selected")
    repository = str(row.get("repo", ""))
    instance_id = str(row.get("instance_id", ""))
    commit = str(row.get("base_commit", ""))
    patch = str(row.get("patch", ""))
    test_patch = str(row.get("test_patch", ""))
    env_script = str(row.get("env_setup_script", ""))
    eval_script = str(row.get("eval_script", ""))
    if not instance_id or "/" not in repository:
        raise ValueError("immutable row identity missing")
    if not FULL_COMMIT.fullmatch(commit):
        raise ValueError("exact base commit missing")
    if not patch.strip() or not test_patch.strip():
        raise ValueError("solution or test patch missing")
    if not env_script.strip() or not eval_script.strip():
        raise ValueError("environment or evaluation script missing")
    targeted = derive_targeted_command(eval_script, language)
    image_name = _image_tag(row)
    task_type, task_evidence = derive_repository_task_type(
        {
            "title": str(row.get("problem_statement", "")).splitlines()[0],
            "body": str(row.get("problem_statement", "")),
            "fix_patch": patch,
            "test_patch": test_patch,
        }
    )
    eval_hash = sha256_bytes(eval_script.encode())
    test_patch_hash = sha256_bytes(test_patch.encode())
    return {
        "schema_version": "general_coding_replay_candidate_source_v3",
        "case_id": f"gc-replay-{SOURCE_ID}-{instance_id}",
        "source_id": SOURCE_ID,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_row_id": instance_id,
        "source_file": SOURCE_FILE,
        "source_file_sha256": source_sha256,
        "source_record_sha256": sha256_bytes(canonical_json(row).encode()),
        "source_lineage_id": f"{DATASET_ID}@{DATASET_REVISION}:{instance_id}",
        "upstream_repository": f"https://github.com/{repository}.git",
        "repository": repository,
        "base_commit": commit,
        "problem_id": instance_id,
        "problem_statement": str(row.get("problem_statement", "")),
        "target_patch": patch,
        "target_patch_sha256": sha256_bytes(patch.encode()),
        "normalized_patch_hash": sha256_bytes(_normalized_patch(patch).encode()),
        "test_patch": test_patch,
        "test_patch_sha256": test_patch_hash,
        "test_set_hash": sha256_bytes(
            canonical_json({"test_patch_sha256": test_patch_hash}).encode()
        ),
        "env_setup_script": env_script,
        "env_setup_script_sha256": sha256_bytes(env_script.encode()),
        "eval_script": eval_script,
        "eval_script_sha256": eval_hash,
        "execution_plan": {
            "derivation": "language_test_command_from_pinned_eval_script_v1",
            "eval_script_sha256": eval_hash,
            "install_command": "true",
            "compile_or_typecheck_command": "true",
            "targeted_test_command": targeted,
            "full_regression_command": "true",
        },
        "image_name": image_name,
        "primary_language": language,
        "primary_task_type": task_type,
        "task_type_evidence": task_evidence,
        "dataset_license_spdx": "Apache-2.0",
        "network_policy": "disabled",
        "gpu_required": False,
        "verified": False,
        "local_replay_passes": 0,
    }


def _iter_existing_rows(control_root: Path) -> Iterable[dict[str, Any]]:
    source = (
        control_root
        / "source-cache"
        / "menvdata-swe"
        / DATASET_REVISION
        / SOURCE_FILE
    ).resolve()
    paths = set(control_root.glob("*.jsonl"))
    paths.update(control_root.glob("**/*receipt.json"))
    paths.update(control_root.glob("**/verified.json"))
    for path in sorted(paths):
        if path.resolve() == source or path.name.startswith("menvdata-swe-multilang-"):
            continue
        try:
            if path.suffix == ".jsonl":
                yield from read_jsonl(path)
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    yield value
        except (OSError, ValueError, json.JSONDecodeError):
            continue


def _all_exclusion_dimensions(control_root: Path) -> dict[str, set[str]]:
    registry = json.loads(
        (control_root / "exclusion-registry.json").read_text(encoding="utf-8")
    )
    existing = [
        *_supplemental_exclusions(control_root),
        *(row for row in registry.get("entries", []) if isinstance(row, dict)),
        *_iter_existing_rows(control_root),
    ]
    dimensions = _exclusion_dimensions(registry, existing)
    dimensions["base"] = {
        canonical_json(
            [
                _normalized_repository(
                    str(row.get("upstream_repository") or row.get("repository") or "")
                ),
                str(row.get("base_commit", "")),
            ]
        )
        for row in existing
        if row.get("base_commit")
    }
    dimensions["base"].discard(canonical_json(["", ""]))
    return dimensions


def _overlap_reasons(
    row: Mapping[str, Any], dimensions: Mapping[str, set[str]]
) -> list[str]:
    reasons = overlap_reasons(row, dimensions)
    base = canonical_json(
        [
            _normalized_repository(str(row.get("upstream_repository", ""))),
            str(row.get("base_commit", "")),
        ]
    )
    if base in dimensions["base"]:
        reasons.append("base")
    return sorted(set(reasons))


def _add_dimensions(row: Mapping[str, Any], dimensions: dict[str, set[str]]) -> None:
    dimensions["repository"].add(
        _normalized_repository(str(row["upstream_repository"]))
    )
    dimensions["problem"].add(str(row["problem_id"]))
    dimensions["lineage"].add(str(row["source_lineage_id"]))
    dimensions["patch"].add(str(row["normalized_patch_hash"]))
    dimensions["test"].add(str(row["test_set_hash"]))
    dimensions["base"].add(
        canonical_json(
            [
                _normalized_repository(str(row["upstream_repository"])),
                str(row["base_commit"]),
            ]
        )
    )


def _add_wave2_dimensions(
    row: Mapping[str, Any], dimensions: dict[str, set[str]]
) -> None:
    """Deduplicate wave-2 rows without turning its repository cap into cap one."""
    dimensions["problem"].add(str(row["problem_id"]))
    dimensions["lineage"].add(str(row["source_lineage_id"]))
    dimensions["patch"].add(str(row["normalized_patch_hash"]))
    dimensions["test"].add(str(row["test_set_hash"]))
    dimensions["base"].add(
        canonical_json(
            [
                _normalized_wave2_repository(str(row["upstream_repository"])),
                str(row["base_commit"]),
            ]
        )
    )


def _normalized_wave2_repository(value: str) -> str:
    """Normalize URL spelling, including a trailing slash after ``.git``."""
    return _normalized_repository(value.strip().rstrip("/"))


def _wave_paths(control_root: Path, wave: int) -> dict[str, Path]:
    if wave not in (1, 2):
        raise ValueError("wave must be 1 or 2")
    prefix = "menvdata-swe-multilang" + ("-wave2" if wave == 2 else "")
    return {
        "manifest": control_root / f"{prefix}-candidate-source-manifest.jsonl",
        "freeze_rejections": control_root / f"{prefix}-freeze-rejections.jsonl",
        "freeze_report": control_root / f"{prefix}-freeze-report.json",
        "executable_manifest": control_root / f"{prefix}-executable-manifest.jsonl",
        "image_rejections": control_root / f"{prefix}-image-rejections.jsonl",
        "image_report": control_root / f"{prefix}-image-report.json",
        "image_state": control_root / f"{prefix}-image-state",
        "verification": control_root / f"{prefix}-verification" / "verified",
        "preflight_report": control_root / f"{prefix}-preflight-report.json",
        "status_report": control_root / f"{prefix}-status-report.json",
    }


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _candidate_order_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    command = str((row.get("execution_plan") or {}).get("targeted_test_command", ""))
    return (
        0
        if re.search(
            r"(?:--test\b|-run\b|::|\.test\.|\.spec\.|/tests?/)",
            command,
            re.IGNORECASE,
        )
        else 1,
        len(command),
        sha256_bytes(
            f"{DATASET_REVISION}\0{row['problem_id']}\0"
            f"{row['normalized_patch_hash']}".encode()
        ),
    )


def _wave2_language_targets(target_rows: int) -> dict[str, int]:
    if target_rows < len(TARGET_LANGUAGES):
        raise ValueError(
            f"wave-2 target must be at least {len(TARGET_LANGUAGES)} rows"
        )
    languages = sorted(TARGET_LANGUAGES.values())
    quotient, remainder = divmod(target_rows, len(languages))
    return {
        language: quotient + (index < remainder)
        for index, language in enumerate(languages)
    }


def _wave2_exclusion_dimensions(control_root: Path) -> dict[str, set[str]]:
    """Keep global exclusions while handling prior MEnv rows explicitly."""
    registry = json.loads(
        (control_root / "exclusion-registry.json").read_text(encoding="utf-8")
    )
    existing = [
        *_supplemental_exclusions(control_root),
        *(row for row in registry.get("entries", []) if isinstance(row, dict)),
        *(
            row
            for row in _iter_existing_rows(control_root)
            if row.get("source_id") != SOURCE_ID
        ),
    ]
    dimensions = _exclusion_dimensions(registry, existing)
    dimensions["repository"] = {
        _normalized_wave2_repository(value) for value in dimensions["repository"]
    }
    dimensions["base"] = {
        canonical_json(
            [
                _normalized_wave2_repository(
                    str(row.get("upstream_repository") or row.get("repository") or "")
                ),
                str(row.get("base_commit", "")),
            ]
        )
        for row in existing
        if row.get("base_commit")
    }
    dimensions["base"].discard(canonical_json(["", ""]))
    return dimensions


def _reserved_dev_repositories(control_root: Path) -> set[str]:
    path = control_root / "general-coding-dev-reservations.jsonl"
    return {
        _normalized_wave2_repository(
            str(row.get("upstream_repository") or row.get("repository") or "")
        )
        for row in _read_jsonl_if_present(path)
        if row.get("upstream_repository") or row.get("repository")
    }


def _known_wave1_immutable_failures(control_root: Path) -> set[str]:
    path = _wave_paths(control_root, 1)["freeze_rejections"]
    return {
        str(row.get("instance_id"))
        for row in _read_jsonl_if_present(path)
        if row.get("reason") == "immutable_license_or_image"
        and row.get("instance_id")
    }


def _verified_menvdata_rows(control_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wave in (1, 2):
        root = _wave_paths(control_root, wave)["verification"]
        for path in sorted(root.glob("*/verified.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(row, dict)
                and row.get("source_id") == SOURCE_ID
                and row.get("dataset_revision") == DATASET_REVISION
                and row.get("primary_language") in TARGET_LANGUAGES.values()
            ):
                rows.append(row)
    return rows


def _repository_counts(
    existing_wave2: Sequence[Mapping[str, Any]],
    verified_rows: Sequence[Mapping[str, Any]],
) -> tuple[Counter[str], int]:
    by_case: dict[str, Mapping[str, Any]] = {}
    for row in [*verified_rows, *existing_wave2]:
        case_id = str(row.get("case_id", ""))
        if case_id:
            by_case[case_id] = row
    counts: Counter[str] = Counter(
        _normalized_wave2_repository(
            str(row.get("upstream_repository") or row.get("repository") or "")
        )
        for row in by_case.values()
    )
    counts.pop("", None)
    return counts, len({str(row.get("case_id")) for row in verified_rows})


def _deduplicate_rejections(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = (
            str(row.get("instance_id", "")),
            str(row.get("reason", "")),
            canonical_json(row.get("dimensions", row.get("detail", ""))),
        )
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def _freeze_license(control_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    repository = str(row["repository"])
    commit = str(row["base_commit"])
    root = control_root / "license-evidence" / "menvdata-swe-multilang"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{repository.replace('/', '__')}.{commit}"
    provenance_path = root / f"{stem}.license-source.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        evidence = control_root / str(provenance["license_evidence_path"])
        if (
            provenance.get("base_commit") != commit
            or provenance.get("repository") != repository
            or provenance.get("spdx") not in PERMISSIVE_SOURCE_LICENSES
            or not evidence.is_file()
            or sha256_file(evidence) != provenance.get("license_evidence_sha256")
        ):
            raise RuntimeError(f"{repository}@{commit}: cached license evidence drift")
        return provenance
    errors = []
    for filename in LICENSE_FILENAMES:
        url = f"https://raw.githubusercontent.com/{repository}/{commit}/{filename}"
        response = requests.get(url, timeout=(20, 90))
        if response.status_code == 404:
            continue
        try:
            response.raise_for_status()
            spdx = _classify_license(response.content)
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{filename}: {exc}")
            continue
        license_path = root / f"{stem}.LICENSE"
        atomic_write(license_path, response.content)
        provenance = {
            "schema_version": "immutable_upstream_license_evidence_v1",
            "repository": repository,
            "base_commit": commit,
            "spdx": spdx,
            "license_file": filename,
            "license_source_url": url,
            "license_evidence_path": license_path.relative_to(control_root).as_posix(),
            "license_evidence_sha256": sha256_file(license_path),
            "license_evidence_method": "classified_immutable_license_file",
        }
        write_json(provenance_path, provenance)
        return provenance
    raise RuntimeError(
        f"{repository}@{commit}: no exact-commit permissive license evidence"
        + (f" ({'; '.join(errors)})" if errors else "")
    )


def _materialize_artifacts(control_root: Path, row: dict[str, Any]) -> None:
    root = (
        control_root
        / "source-cache"
        / "menvdata-swe"
        / DATASET_REVISION
        / "candidates"
        / str(row["problem_id"])
    )
    artifacts = {
        "target.patch": str(row.pop("target_patch")).encode(),
        "test.patch": str(row.pop("test_patch")).encode(),
        "problem.md": str(row.pop("problem_statement")).encode(),
        "env_setup.sh": str(row.pop("env_setup_script")).encode(),
        "eval.sh": str(row.pop("eval_script")).encode(),
    }
    for name, content in artifacts.items():
        path = root / name
        if path.is_file() and path.read_bytes() != content:
            raise RuntimeError(f"{row['case_id']}: frozen artifact drift at {name}")
        if not path.is_file():
            atomic_write(path, content)
    row.update(
        {
            "target_patch_path": (root / "target.patch").relative_to(control_root).as_posix(),
            "test_patch_path": (root / "test.patch").relative_to(control_root).as_posix(),
            "problem_statement_path": (root / "problem.md")
            .relative_to(control_root)
            .as_posix(),
            "env_setup_script_path": (root / "env_setup.sh")
            .relative_to(control_root)
            .as_posix(),
            "eval_script_path": (root / "eval.sh").relative_to(control_root).as_posix(),
        }
    )


def _freeze_menvdata_multilang_wave2(
    control_root: Path, *, target_rows: int, repository_cap: int
) -> dict[str, Any]:
    if repository_cap < 1:
        raise ValueError("wave-2 repository cap must be positive")
    language_targets = _wave2_language_targets(target_rows)
    source = (
        control_root
        / "source-cache"
        / "menvdata-swe"
        / DATASET_REVISION
        / SOURCE_FILE
    )
    if not source.is_file() or sha256_file(source) != SOURCE_FILE_SHA256:
        raise RuntimeError("pinned MEnvData-SWE source file missing or drifted")
    raw_rows = read_jsonl(source)
    if len(raw_rows) != SOURCE_ROWS:
        raise RuntimeError("pinned MEnvData-SWE row count drifted")

    paths = _wave_paths(control_root, 2)
    existing_wave2 = _read_jsonl_if_present(paths["manifest"])
    if len(existing_wave2) > target_rows:
        raise RuntimeError("existing wave-2 manifest exceeds requested target")
    existing_ids = [str(row.get("problem_id", "")) for row in existing_wave2]
    if len(existing_ids) != len(set(existing_ids)):
        raise RuntimeError("existing wave-2 manifest contains duplicate instance IDs")
    wave1_rows = _read_jsonl_if_present(_wave_paths(control_root, 1)["manifest"])
    wave1_ids = {
        str(row.get("problem_id") or row.get("dataset_row_id") or "")
        for row in wave1_rows
    }
    if wave1_ids.intersection(existing_ids):
        raise RuntimeError("existing wave-2 manifest overlaps frozen wave-1 IDs")

    verified_rows = _verified_menvdata_rows(control_root)
    repository_counts, verified_receipts = _repository_counts(
        existing_wave2, verified_rows
    )
    if any(count > repository_cap for count in repository_counts.values()):
        raise RuntimeError("existing eligible rows already exceed wave-2 repository cap")
    selected_counts: Counter[str] = Counter(
        str(row["primary_language"]) for row in existing_wave2
    )
    for language, count in selected_counts.items():
        if count > language_targets.get(language, 0):
            raise RuntimeError(
                f"existing wave-2 manifest exceeds {language} balanced target"
            )

    dimensions = _wave2_exclusion_dimensions(control_root)
    for row in existing_wave2:
        _add_wave2_dimensions(row, dimensions)
    reserved_repositories = _reserved_dev_repositories(control_root)
    known_failures = _known_wave1_immutable_failures(control_root)
    rejections: list[dict[str, Any]] = _read_jsonl_if_present(
        paths["freeze_rejections"]
    )
    eligible: dict[str, list[dict[str, Any]]] = {
        language: [] for language in TARGET_LANGUAGES.values()
    }
    existing_id_set = set(existing_ids)
    for raw in raw_rows:
        if raw.get("language") not in TARGET_LANGUAGES:
            continue
        instance_id = str(raw.get("instance_id", ""))
        if instance_id in existing_id_set:
            continue
        if instance_id in wave1_ids:
            rejections.append(
                {"instance_id": instance_id, "reason": "wave1_instance_frozen"}
            )
            continue
        if instance_id in known_failures:
            rejections.append(
                {
                    "instance_id": instance_id,
                    "reason": "known_wave1_immutable_failure",
                }
            )
            continue
        try:
            candidate = adapt_menvdata_multilang_row(raw)
        except ValueError as exc:
            rejections.append(
                {
                    "instance_id": instance_id,
                    "reason": "schema_or_execution",
                    "detail": str(exc),
                }
            )
            continue
        repository = _normalized_wave2_repository(
            str(candidate["upstream_repository"])
        )
        if repository in reserved_repositories:
            rejections.append(
                {
                    "instance_id": instance_id,
                    "reason": "dev_repository_reserved",
                    "repository": repository,
                }
            )
            continue
        reasons = _overlap_reasons(candidate, dimensions)
        if reasons:
            rejections.append(
                {
                    "instance_id": instance_id,
                    "reason": "existing_overlap",
                    "dimensions": reasons,
                }
            )
            continue
        eligible[str(candidate["primary_language"])].append(candidate)
    for rows in eligible.values():
        rows.sort(key=_candidate_order_key)

    frozen = list(existing_wave2)
    added_this_run = 0
    image_cache: dict[str, dict[str, str]] = {}
    for language in sorted(eligible):
        for candidate in eligible[language]:
            if selected_counts[language] >= language_targets[language]:
                break
            repository = _normalized_wave2_repository(
                str(candidate["upstream_repository"])
            )
            if repository_counts[repository] >= repository_cap:
                rejections.append(
                    {
                        "instance_id": candidate["problem_id"],
                        "reason": "repository_cap",
                        "repository": repository,
                    }
                )
                continue
            reasons = _overlap_reasons(candidate, dimensions)
            if reasons:
                rejections.append(
                    {
                        "instance_id": candidate["problem_id"],
                        "reason": "selected_overlap",
                        "dimensions": reasons,
                    }
                )
                continue
            try:
                license_evidence = _freeze_license(control_root, candidate)
                image_name = str(candidate.pop("image_name"))
                if image_name not in image_cache:
                    image_cache[image_name] = _docker_hub_digest(image_name)
                candidate.update(
                    {
                        "license_spdx": license_evidence["spdx"],
                        "license_evidence_path": license_evidence[
                            "license_evidence_path"
                        ],
                        "license_evidence_sha256": license_evidence[
                            "license_evidence_sha256"
                        ],
                        "license_evidence_method": license_evidence[
                            "license_evidence_method"
                        ],
                        "license_source_url": license_evidence["license_source_url"],
                        "container_image": image_cache[image_name],
                    }
                )
                _materialize_artifacts(control_root, candidate)
                candidate["_manifest_dir"] = str(control_root.resolve())
                frozen.append(candidate)
                selected_counts[language] += 1
                repository_counts[repository] += 1
                added_this_run += 1
                _add_wave2_dimensions(candidate, dimensions)
                frozen.sort(key=lambda row: str(row["case_id"]))
                # Persist each success so an interrupted network freeze can resume.
                write_jsonl(paths["manifest"], frozen)
            except (OSError, RuntimeError, requests.RequestException) as exc:
                rejections.append(
                    {
                        "instance_id": candidate["problem_id"],
                        "reason": "immutable_license_or_image",
                        "detail": str(exc)[-2000:],
                    }
                )

    frozen.sort(key=lambda row: str(row["case_id"]))
    write_jsonl(paths["manifest"], frozen)
    rejections = _deduplicate_rejections(rejections)
    write_jsonl(paths["freeze_rejections"], rejections)
    deficits = {
        language: language_targets[language] - selected_counts[language]
        for language in sorted(language_targets)
        if selected_counts[language] < language_targets[language]
    }
    frozen_repository_counts = Counter(
        _normalized_wave2_repository(str(row["upstream_repository"])) for row in frozen
    )
    report = {
        "schema_version": "general_coding_replay_menvdata_swe_multilang_wave2_freeze_v1",
        "status": "strict_candidate_pool_ready" if not deficits else "capacity_deficit",
        "wave": 2,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_file_sha256": SOURCE_FILE_SHA256,
        "source_rows": len(raw_rows),
        "target_rows": target_rows,
        "language_targets": language_targets,
        "strict_capacity": len(frozen),
        "added_this_run": added_this_run,
        "selected_by_language": dict(sorted(selected_counts.items())),
        "selected_by_repository": dict(sorted(frozen_repository_counts.items())),
        "selected_repositories": len(frozen_repository_counts),
        "cap_counts_by_repository": dict(sorted(repository_counts.items())),
        "deficits": deficits,
        "rejections_by_reason": dict(
            sorted(Counter(str(row["reason"]) for row in rejections).items())
        ),
        "manifest": {
            "path": paths["manifest"].name,
            "rows": len(frozen),
            "sha256": sha256_file(paths["manifest"]),
        },
        "repository_cap_policy": {
            "cap": repository_cap,
            "identity": "normalized_repository_v1",
            "counts_existing_eligible_verified_receipts": True,
            "eligible_verified_receipts": verified_receipts,
            "counts_existing_and_new_wave2_rows": True,
        },
        "wave1_manifest_mutated": False,
        "resume_safe_accumulation": True,
        "required_local_replay_passes": 2,
        "freeze_network_policy": "license_and_registry_metadata_only",
        "verification_network_policy": "disabled",
        "gpu_policy": "cpu_only",
        "accepted_jsonl_mutated": False,
    }
    write_json(paths["freeze_report"], report)
    if deficits:
        raise RuntimeError(
            f"MEnvData-SWE multilang wave-2 strict capacity deficit: {deficits}"
        )
    return report


def freeze_menvdata_multilang(
    control_root: Path,
    *,
    per_language: int = DEFAULT_PER_LANGUAGE,
    wave: int = 1,
    target_rows: int = WAVE2_TARGET_ROWS,
    repository_cap: int = WAVE2_REPOSITORY_CAP,
) -> dict[str, Any]:
    if wave == 2:
        return _freeze_menvdata_multilang_wave2(
            control_root,
            target_rows=target_rows,
            repository_cap=repository_cap,
        )
    if wave != 1:
        raise ValueError("wave must be 1 or 2")
    if per_language < 1:
        raise ValueError("per-language target must be positive")
    source = (
        control_root
        / "source-cache"
        / "menvdata-swe"
        / DATASET_REVISION
        / SOURCE_FILE
    )
    if not source.is_file() or sha256_file(source) != SOURCE_FILE_SHA256:
        raise RuntimeError("pinned MEnvData-SWE source file missing or drifted")
    raw_rows = read_jsonl(source)
    if len(raw_rows) != SOURCE_ROWS:
        raise RuntimeError("pinned MEnvData-SWE row count drifted")
    dimensions = _all_exclusion_dimensions(control_root)
    eligible: dict[str, list[dict[str, Any]]] = {
        language: [] for language in TARGET_LANGUAGES.values()
    }
    rejections: list[dict[str, Any]] = []
    for raw in raw_rows:
        if raw.get("language") not in TARGET_LANGUAGES:
            continue
        try:
            candidate = adapt_menvdata_multilang_row(raw)
        except ValueError as exc:
            rejections.append(
                {
                    "instance_id": raw.get("instance_id"),
                    "reason": "schema_or_execution",
                    "detail": str(exc),
                }
            )
            continue
        reasons = _overlap_reasons(candidate, dimensions)
        if reasons:
            rejections.append(
                {
                    "instance_id": raw.get("instance_id"),
                    "reason": "existing_overlap",
                    "dimensions": reasons,
                }
            )
            continue
        eligible[str(candidate["primary_language"])].append(candidate)
    for rows in eligible.values():
        rows.sort(key=_candidate_order_key)
    frozen: list[dict[str, Any]] = []
    image_cache: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    # Round-robin keeps each language independently capacity-checked.
    for language in sorted(eligible):
        for candidate in eligible[language]:
            if counts[language] >= per_language:
                break
            reasons = _overlap_reasons(candidate, dimensions)
            if reasons:
                rejections.append(
                    {
                        "instance_id": candidate["problem_id"],
                        "reason": "selected_overlap",
                        "dimensions": reasons,
                    }
                )
                continue
            try:
                license_evidence = _freeze_license(control_root, candidate)
                image_name = str(candidate.pop("image_name"))
                if image_name not in image_cache:
                    image_cache[image_name] = _docker_hub_digest(image_name)
                candidate.update(
                    {
                        "license_spdx": license_evidence["spdx"],
                        "license_evidence_path": license_evidence[
                            "license_evidence_path"
                        ],
                        "license_evidence_sha256": license_evidence[
                            "license_evidence_sha256"
                        ],
                        "license_evidence_method": license_evidence[
                            "license_evidence_method"
                        ],
                        "license_source_url": license_evidence["license_source_url"],
                        "container_image": image_cache[image_name],
                    }
                )
                _materialize_artifacts(control_root, candidate)
                candidate["_manifest_dir"] = str(control_root.resolve())
                frozen.append(candidate)
                counts[language] += 1
                _add_dimensions(candidate, dimensions)
            except (OSError, RuntimeError, requests.RequestException) as exc:
                rejections.append(
                    {
                        "instance_id": candidate["problem_id"],
                        "reason": "immutable_license_or_image",
                        "detail": str(exc)[-2000:],
                    }
                )
    deficits = {
        language: per_language - counts[language]
        for language in sorted(eligible)
        if counts[language] < per_language
    }
    if deficits:
        raise RuntimeError(f"MEnvData-SWE multilang strict capacity deficit: {deficits}")
    frozen.sort(key=lambda row: str(row["case_id"]))
    manifest = control_root / "menvdata-swe-multilang-candidate-source-manifest.jsonl"
    _write_frozen_jsonl(manifest, frozen)
    write_jsonl(
        control_root / "menvdata-swe-multilang-freeze-rejections.jsonl", rejections
    )
    report = {
        "schema_version": "general_coding_replay_menvdata_swe_multilang_freeze_v1",
        "status": "strict_candidate_pool_ready",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_file_sha256": SOURCE_FILE_SHA256,
        "source_rows": len(raw_rows),
        "target_rows": per_language * len(TARGET_LANGUAGES),
        "strict_capacity": len(frozen),
        "selected_by_language": dict(sorted(counts.items())),
        "selected_repositories": len({row["repository"] for row in frozen}),
        "rejections_by_reason": dict(
            sorted(Counter(str(row["reason"]) for row in rejections).items())
        ),
        "manifest": {
            "path": manifest.name,
            "rows": len(frozen),
            "sha256": sha256_file(manifest),
        },
        "dedup_dimensions": [
            "repository",
            "problem",
            "lineage",
            "base",
            "patch",
            "test",
        ],
        "required_local_replay_passes": 2,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
        "accepted_jsonl_mutated": False,
    }
    write_json(control_root / "menvdata-swe-multilang-freeze-report.json", report)
    return report


def _select_rows(
    rows: Sequence[dict[str, Any]],
    *,
    case_ids: Sequence[str],
    languages: Sequence[str],
    one_per_language: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    def planned_test(row: Mapping[str, Any]) -> str:
        return str((row.get("execution_plan") or {}).get("targeted_test_command", ""))

    selected = list(rows)
    if case_ids:
        requested = set(case_ids)
        selected = [row for row in selected if str(row["case_id"]) in requested]
        missing = sorted(requested - {str(row["case_id"]) for row in selected})
        if missing:
            raise ValueError("unknown MEnvData-SWE case IDs: " + ", ".join(missing))
    if languages:
        requested_languages = set(languages)
        unknown = requested_languages - set(TARGET_LANGUAGES.values())
        if unknown:
            raise ValueError("unknown languages: " + ", ".join(sorted(unknown)))
        selected = [
            row for row in selected if str(row["primary_language"]) in requested_languages
        ]
    if one_per_language:
        by_language: dict[str, dict[str, Any]] = {}
        for row in sorted(
            selected,
            key=lambda item: (
                1
                if re.search(
                    r"(?:\./\.\.\.|--all(?:-targets|-features)?\b|"
                    r"(?:npm|yarn|pnpm)\s+(?:run\s+)?test\s*$)",
                    planned_test(item),
                    re.IGNORECASE,
                )
                else 0,
                0
                if re.search(
                    r"(?:--test\b|-run\b|::|\.test\.|\.spec\.|/tests?/|"
                    r"\btests?/[^ ]+)",
                    planned_test(item),
                    re.IGNORECASE,
                )
                else 1,
                len(planned_test(item)),
                str(item["case_id"]),
            ),
        ):
            by_language.setdefault(str(row["primary_language"]), row)
        selected = [by_language[key] for key in sorted(by_language)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if len(selected) > MAX_PREPARE_IMAGES:
        raise ValueError(
            f"bounded preparation permits at most {MAX_PREPARE_IMAGES} images; "
            "use --limit, --case-id, or --one-per-language"
        )
    return selected


def _validated_commands(row: Mapping[str, Any], control_root: Path) -> dict[str, str]:
    plan = dict(row.get("execution_plan") or {})
    eval_path = control_root / str(row["eval_script_path"])
    if not eval_path.is_file() or sha256_file(eval_path) != row.get("eval_script_sha256"):
        raise ValueError("frozen eval script missing or hash drifted")
    if plan.get("eval_script_sha256") != row.get("eval_script_sha256"):
        raise ValueError("execution plan eval script hash drifted")
    derived = derive_targeted_command(
        eval_path.read_text(encoding="utf-8"), str(row["primary_language"])
    )
    if derived != plan.get("targeted_test_command"):
        raise ValueError("source-derived targeted command drifted")
    commands = {
        name: str(plan.get(name, ""))
        for name in (
            "install_command",
            "compile_or_typecheck_command",
            "targeted_test_command",
            "full_regression_command",
        )
    }
    prelude = _derive_environment_prelude(eval_path.read_text(encoding="utf-8"))
    if prelude:
        target = commands["targeted_test_command"]
        prefix = "set -euo pipefail; "
        if not target.startswith(prefix):
            raise ValueError("derived targeted command lacks fail-closed prefix")
        commands["targeted_test_command"] = (
            " && ".join(prelude + ["set -euo pipefail"])
            + "; "
            + target.removeprefix(prefix)
        )
    if not all(commands.values()) or FORBIDDEN_COMMAND.search("\n".join(commands.values())):
        raise ValueError("execution plan is incomplete or not offline")
    return commands


def _source_workdir(row: Mapping[str, Any], control_root: Path) -> str:
    env_path = control_root / str(row["env_setup_script_path"])
    if not env_path.is_file() or sha256_file(env_path) != row.get(
        "env_setup_script_sha256"
    ):
        raise ValueError("frozen environment script missing or hash drifted")
    workdirs = re.findall(
        r"(?m)^\s*cd\s+(?:--\s+)?['\"]?(/[A-Za-z0-9_./-]+)['\"]?\s*$",
        env_path.read_text(encoding="utf-8"),
    )
    if not workdirs:
        raise ValueError("environment script has no absolute repository workdir")
    workdir = workdirs[-1].rstrip("/") or "/"
    if workdir == "/" or ".." in Path(workdir).parts:
        raise ValueError("environment script repository workdir is unsafe")
    return workdir


def prepare_menvdata_multilang(
    control_root: Path,
    *,
    wave: int = 1,
    case_ids: Sequence[str] = (),
    languages: Sequence[str] = (),
    one_per_language: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    paths = _wave_paths(control_root, wave)
    rows = _select_rows(
        read_jsonl(paths["manifest"]),
        case_ids=case_ids,
        languages=languages,
        one_per_language=one_per_language,
        limit=limit,
    )
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    state_root = paths["image_state"]
    state_root.mkdir(parents=True, exist_ok=True)
    for raw in rows:
        row = dict(raw)
        try:
            commands = _validated_commands(row, control_root)
            workdir = _source_workdir(row, control_root)
            expected = dict(row["container_image"])
            lock_path = state_root / f"{row['case_id']}.lock"
            with lock_path.open("a+", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                actual = _docker_image_identity(str(expected["repo_tag"]))
                if actual is None or actual["repo_digest"] != expected["repo_digest"]:
                    pull = subprocess.run(
                        ["docker", "pull", str(expected["repo_digest"])],
                        capture_output=True,
                        text=True,
                        timeout=3600,
                        check=False,
                    )
                    if pull.returncode:
                        raise RuntimeError(pull.stderr[-2000:])
                    tag = subprocess.run(
                        [
                            "docker",
                            "tag",
                            str(expected["repo_digest"]),
                            str(expected["repo_tag"]),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    if tag.returncode:
                        raise RuntimeError(tag.stderr[-2000:])
                    actual = _docker_image_identity(str(expected["repo_tag"]))
                if actual is None or actual["repo_digest"] != expected["repo_digest"]:
                    raise RuntimeError("image digest mismatch")
                write_json(
                    state_root / f"{row['case_id']}.json",
                    {
                        "case_id": row["case_id"],
                        "expected_repo_digest": expected["repo_digest"],
                        "actual": actual,
                        "status": "prepared",
                    },
                )
            patch_path = control_root / str(row["target_patch_path"])
            row.update(
                {
                    "container_image": actual,
                    "container_workdir": workdir,
                    "container_reset_command": f"git reset --hard {row['base_commit']}",
                    "container_clean_command": "git clean -fd",
                    "verification_backend": "docker",
                    "allowed_paths": _patch_paths(
                        patch_path.read_text(encoding="utf-8")
                    ),
                    **commands,
                    "benchmark_exclusion_gate": "ready",
                    "timeouts": {
                        "checkout_dependency": 120,
                        "parent": 3600,
                        "target": 3600,
                        "regression": 600,
                        "fresh_total": 10800,
                    },
                }
            )
            prepared.append(row)
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.TimeoutExpired,
        ) as exc:
            rejected.append({"case_id": raw["case_id"], "detail": str(exc)[-2000:]})
    manifest = paths["executable_manifest"]
    with (state_root / ".manifest.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        write_jsonl(manifest, prepared)
        write_jsonl(paths["image_rejections"], rejected)
        report = {
            "schema_version": "general_coding_replay_menvdata_swe_multilang_images_v1",
            "scope": "bounded",
            "requested": len(rows),
            "prepared": len(prepared),
            "rejected": len(rejected),
            "prepared_by_language": dict(
                sorted(Counter(str(row["primary_language"]) for row in prepared).items())
            ),
            "manifest": {
                "path": manifest.name,
                "rows": len(prepared),
                "sha256": sha256_file(manifest),
            },
        }
        if wave == 2:
            report.update({"wave": 2, "workflow": "explicit_per_batch_manifest"})
        write_json(paths["image_report"], report)
    return report


def verify_menvdata_multilang(
    control_root: Path,
    *,
    wave: int = 1,
    case_ids: Sequence[str] = (),
    languages: Sequence[str] = (),
    one_per_language: bool = False,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    images = prepare_menvdata_multilang(
        control_root,
        wave=wave,
        case_ids=case_ids,
        languages=languages,
        one_per_language=one_per_language,
        limit=limit,
    )
    if images["prepared"] == 0:
        return {"status": "blocked_image_gate", "image_report": images, "verified_rows": 0}
    verification = verify_candidates(
        _wave_paths(control_root, wave)["executable_manifest"],
        _wave_paths(control_root, wave)["verification"],
        resume=True,
        workers=workers,
    )
    report = {
        "schema_version": "general_coding_replay_menvdata_swe_multilang_preflight_v1",
        "status": (
            "passed"
            if images["prepared"] == images["requested"]
            and verification["verified"] + verification.get("skipped_completed", 0)
            == images["prepared"]
            and verification["rejected"] == 0
            else "partial_image_gate"
            if images["prepared"] != images["requested"]
            else "failed"
        ),
        "scope": "bounded",
        "image_report": images,
        "verification": verification,
        "verified_rows": verification["verified"],
        "required_local_replay_passes": 2,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
    }
    if wave == 2:
        report["wave"] = 2
    write_json(_wave_paths(control_root, wave)["preflight_report"], report)
    build_menvdata_multilang_status_report(control_root, wave=wave)
    return report


def verify_all_menvdata_multilang(
    control_root: Path,
    *,
    wave: int,
    batch_size: int = 25,
    workers: int = 1,
) -> dict[str, Any]:
    """Verify a frozen wave in bounded, resume-safe sequential batches."""

    if not 1 <= batch_size <= 25:
        raise ValueError("batch_size must be between 1 and 25")
    rows = sorted(
        read_jsonl(_wave_paths(control_root, wave)["manifest"]),
        key=lambda row: str(row["case_id"]),
    )
    batches = []
    for offset in range(0, len(rows), batch_size):
        case_ids = [
            str(row["case_id"]) for row in rows[offset : offset + batch_size]
        ]
        report = verify_menvdata_multilang(
            control_root,
            wave=wave,
            case_ids=case_ids,
            workers=workers,
        )
        batches.append(
            {
                "index": len(batches),
                "rows": len(case_ids),
                "status": report["status"],
            }
        )
    status = build_menvdata_multilang_status_report(control_root, wave=wave)
    attempted = int(status["preflight_attempted_unique"])
    verified = int(status["preflight_verified_unique"])
    rejected = int(status["preflight_rejected_unique"])
    complete = attempted == len(rows)
    return {
        "schema_version": "general_coding_replay_menvdata_swe_multilang_all_batches_v1",
        "status": (
            "passed"
            if complete and rejected == 0
            else "completed_with_rejections"
            if complete
            else "blocked"
        ),
        "wave": wave,
        "frozen_rows": len(rows),
        "attempted": attempted,
        "verified": verified,
        "rejected": rejected,
        "batch_size": batch_size,
        "workers": workers,
        "batches": batches,
        "network_policy": "disabled_during_replay",
        "gpu_policy": "cpu_only",
    }


def build_menvdata_multilang_status_report(
    control_root: Path, *, wave: int = 1
) -> dict[str, Any]:
    paths = _wave_paths(control_root, wave)
    frozen = read_jsonl(paths["manifest"])
    by_case = {str(row["case_id"]): row for row in frozen}
    verification_root = paths["verification"]
    statuses = []
    for path in sorted(verification_root.glob("*/status.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("case_id") in by_case:
            statuses.append(value)
    verified = [row for row in statuses if row.get("status") == "verified"]
    rejected = [row for row in statuses if row.get("status") == "rejected"]
    prepared_receipts = list(
        paths["image_state"].glob("*.json")
    )
    report = {
        "schema_version": "general_coding_replay_menvdata_swe_multilang_status_v1",
        "status": (
            "five_language_preflight_passed"
            if {
                str(by_case[str(row["case_id"])]["primary_language"])
                for row in verified
            }
            == set(TARGET_LANGUAGES.values())
            else "preflight_language_gap"
        ),
        "frozen_unique_candidates": len(frozen),
        "strict_executable_plan_capacity": sum(
            bool(row.get("execution_plan")) for row in frozen
        ),
        "image_prepared_unique": len(prepared_receipts),
        "preflight_attempted_unique": len(statuses),
        "preflight_verified_unique": len(verified),
        "preflight_rejected_unique": len(rejected),
        "verified_by_language": dict(
            sorted(
                Counter(
                    str(by_case[str(row["case_id"])]["primary_language"])
                    for row in verified
                ).items()
            )
        ),
        "blockers": dict(
            sorted(
                Counter(str(row.get("rejection_reason", "unknown")) for row in rejected)
                .items()
            )
        ),
        "dataset_revision": DATASET_REVISION,
        "source_file_sha256": SOURCE_FILE_SHA256,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
        "required_local_replay_passes": 2,
        "accepted_jsonl_mutated": False,
    }
    if wave == 2:
        report["wave"] = 2
    write_json(paths["status_report"], report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--per-language", type=int, default=DEFAULT_PER_LANGUAGE)
    freeze.add_argument("--wave", type=int, choices=(1, 2), default=1)
    freeze.add_argument("--target-rows", type=int, default=WAVE2_TARGET_ROWS)
    freeze.add_argument("--repo-cap", type=int, default=WAVE2_REPOSITORY_CAP)
    for name in ("prepare", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--wave", type=int, choices=(1, 2), default=1)
        command.add_argument("--case-id", action="append", default=[])
        command.add_argument(
            "--language",
            action="append",
            choices=sorted(TARGET_LANGUAGES.values()),
            default=[],
        )
        command.add_argument("--one-per-language", action="store_true")
        command.add_argument("--limit", type=int)
        if name == "verify":
            command.add_argument("--workers", type=int, default=1)
            command.add_argument("--all-batches", action="store_true")
            command.add_argument("--batch-size", type=int, default=25)
    status = commands.add_parser("status")
    status.add_argument("--wave", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        report = freeze_menvdata_multilang(
            args.control_root,
            per_language=args.per_language,
            wave=args.wave,
            target_rows=args.target_rows,
            repository_cap=args.repo_cap,
        )
    elif args.command == "prepare":
        report = prepare_menvdata_multilang(
            args.control_root,
            wave=args.wave,
            case_ids=args.case_id,
            languages=args.language,
            one_per_language=args.one_per_language,
            limit=args.limit,
        )
    elif args.command == "verify":
        if args.all_batches:
            if args.case_id or args.language or args.one_per_language or args.limit:
                parser.error(
                    "--all-batches cannot be combined with bounded selection options"
                )
            report = verify_all_menvdata_multilang(
                args.control_root,
                wave=args.wave,
                batch_size=args.batch_size,
                workers=args.workers,
            )
        else:
            report = verify_menvdata_multilang(
                args.control_root,
                wave=args.wave,
                case_ids=args.case_id,
                languages=args.language,
                one_per_language=args.one_per_language,
                limit=args.limit,
                workers=args.workers,
            )
    else:
        report = build_menvdata_multilang_status_report(
            args.control_root, wave=args.wave
        )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
