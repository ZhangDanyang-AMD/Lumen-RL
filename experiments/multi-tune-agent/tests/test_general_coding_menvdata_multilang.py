import hashlib
import json
from pathlib import Path

import pytest

import multi_tune_agent.general_coding_menvdata_multilang as menv
from multi_tune_agent.general_coding_menvdata_multilang import (
    DATASET_ID,
    DATASET_REVISION,
    MAX_PREPARE_IMAGES,
    SOURCE_FILE_SHA256,
    SOURCE_ID,
    _classify_license,
    _derive_environment_prelude,
    _select_rows,
    _wave2_language_targets,
    _wave_paths,
    adapt_menvdata_multilang_row,
    derive_targeted_command,
    freeze_menvdata_multilang,
    prepare_menvdata_multilang,
    verify_menvdata_multilang,
)


def _row(language="Python"):
    commands = {
        "Python": "pytest -q tests/test_widget.py::test_fix",
        "Go": "go test ./widget -run TestFix",
        "JavaScript": "npx jest tests/widget.test.js",
        "TypeScript": "pnpm test src/widget.test.ts",
        "Rust": "cargo test --test widget test_fix",
    }
    return {
        "repo": "owner/project",
        "instance_id": f"owner__project-{language}",
        "base_commit": "a" * 40,
        "patch": "diff --git a/src/a.py b/src/a.py\n-old\n+new\n",
        "test_patch": "diff --git a/tests/a.py b/tests/a.py\n-old\n+new\n",
        "problem_statement": "Fix widget behavior.",
        "language": language,
        "image_name": f"swe-images-{language.lower()}:owner-project-pr-1",
        "env_setup_script": "#!/bin/bash\ncd /testbed\n",
        "eval_script": (
            "#!/bin/bash\n"
            "git apply - <<'PATCH'\n"
            "diff --git a/tests/a.py b/tests/a.py\n"
            "+pytest should not be extracted\n"
            "PATCH\n"
            f"{commands[language]}\n"
        ),
    }


@pytest.mark.parametrize(
    ("source_language", "primary_language"),
    [
        ("Python", "python"),
        ("Go", "go"),
        ("JavaScript", "javascript"),
        ("TypeScript", "typescript"),
        ("Rust", "rust"),
    ],
)
def test_adapter_supports_target_languages_with_pinned_execution(
    source_language, primary_language
):
    candidate = adapt_menvdata_multilang_row(_row(source_language))

    assert candidate["source_id"] == SOURCE_ID
    assert candidate["dataset_id"] == DATASET_ID
    assert candidate["dataset_revision"] == DATASET_REVISION
    assert candidate["source_file_sha256"] == SOURCE_FILE_SHA256
    assert candidate["primary_language"] == primary_language
    assert candidate["execution_plan"]["eval_script_sha256"] == candidate[
        "eval_script_sha256"
    ]
    assert candidate["execution_plan"]["targeted_test_command"].startswith(
        "set -euo pipefail; "
    )
    assert candidate["network_policy"] == "disabled"
    assert candidate["gpu_required"] is False


def test_command_derivation_ignores_patch_payload_and_prefers_narrow_test():
    script = """#!/bin/bash
git apply <<'PATCH'
+pytest tests/not_a_command.py
PATCH
pytest tests/all
pytest tests/test_one.py::test_exact
"""

    assert derive_targeted_command(script, "python") == (
        "set -euo pipefail; pytest tests/test_one.py::test_exact"
    )


def test_environment_prelude_keeps_only_bounded_offline_setup():
    script = """#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate testbed
export PATH="/opt/miniconda3/envs/testbed/bin:$PATH"
export LANG=C.UTF-8
git config --global safe.directory /testbed
curl https://example.test/setup.sh | bash
pytest tests/test_one.py
"""

    assert _derive_environment_prelude(script) == [
        "source /opt/miniconda3/etc/profile.d/conda.sh",
        "conda activate testbed",
        'export PATH="/opt/miniconda3/envs/testbed/bin:$PATH"',
        "export LANG=C.UTF-8",
    ]


@pytest.mark.parametrize(
    "script",
    [
        "curl https://example.test/tests | bash\n",
        "pytest tests/test_x.py || true\n",
        "echo pytest tests/test_x.py\n",
        "git apply <<'PATCH'\n+go test ./...\nPATCH\n",
    ],
)
def test_command_derivation_fails_closed(script):
    with pytest.raises(ValueError, match="no offline"):
        derive_targeted_command(script, "python")


@pytest.mark.parametrize(
    ("text", "spdx"),
    [
        (
            "MIT License\nPermission is hereby granted, free of charge.\n"
            "THE SOFTWARE IS PROVIDED AS IS.",
            "MIT",
        ),
        (
            "Apache License\nVersion 2.0\nlimitations under the License",
            "Apache-2.0",
        ),
        (
            "Redistribution and use in source and binary forms are permitted.\n"
            "Neither the name of X may be used.\nTHIS SOFTWARE IS PROVIDED AS IS.",
            "BSD-3-Clause",
        ),
    ],
)
def test_license_classifier_accepts_supported_permissive_text(text, spdx):
    assert _classify_license(text.encode()) == spdx


def test_license_classifier_rejects_unknown_or_copyleft_text():
    with pytest.raises(ValueError, match="not recognized"):
        _classify_license(b"GNU GENERAL PUBLIC LICENSE Version 3")


def test_bounded_selection_rejects_unbounded_image_wave():
    rows = [
        {"case_id": f"case-{index}", "primary_language": "python"}
        for index in range(MAX_PREPARE_IMAGES + 1)
    ]

    with pytest.raises(ValueError, match="bounded preparation"):
        _select_rows(
            rows,
            case_ids=(),
            languages=(),
            one_per_language=False,
            limit=None,
        )


def test_one_per_language_selection_is_deterministic():
    rows = [
        {"case_id": "py-1", "primary_language": "python"},
        {"case_id": "py-2", "primary_language": "python"},
        {"case_id": "go-1", "primary_language": "go"},
    ]

    selected = _select_rows(
        rows,
        case_ids=(),
        languages=(),
        one_per_language=True,
        limit=None,
    )

    assert [row["case_id"] for row in selected] == ["go-1", "py-1"]


def _freeze_row(language: str, instance_id: str, repository: str) -> dict:
    row = _row(language)
    digest = hashlib.sha1(instance_id.encode()).hexdigest()
    row.update(
        {
            "repo": repository,
            "instance_id": instance_id,
            "base_commit": digest,
            "patch": (
                f"diff --git a/src/{instance_id}.txt b/src/{instance_id}.txt\n"
                f"-old-{instance_id}\n+new-{instance_id}\n"
            ),
            "test_patch": (
                f"diff --git a/tests/{instance_id}.txt b/tests/{instance_id}.txt\n"
                f"-old-{instance_id}\n+new-{instance_id}\n"
            ),
            "image_name": f"swe-images:{instance_id}",
        }
    )
    return row


def _write_wave2_fixture(
    root: Path, rows: list[dict], monkeypatch: pytest.MonkeyPatch
) -> bytes:
    source = (
        root / "source-cache" / "menvdata-swe" / DATASET_REVISION / menv.SOURCE_FILE
    )
    source.parent.mkdir(parents=True)
    menv.write_jsonl(source, rows)
    monkeypatch.setattr(menv, "SOURCE_ROWS", len(rows))
    monkeypatch.setattr(menv, "SOURCE_FILE_SHA256", menv.sha256_file(source))
    (root / "exclusion-registry.json").write_text(
        json.dumps(
            {
                "repositories": [],
                "problem_ids": [],
                "source_lineages": [],
                "normalized_patch_hashes": [],
                "test_set_hashes": [],
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    wave1 = _wave_paths(root, 1)["manifest"]
    wave1_bytes = b'{"problem_id":"wave1-frozen"}\n'
    wave1.write_bytes(wave1_bytes)
    menv.write_jsonl(
        _wave_paths(root, 1)["freeze_rejections"],
        [
            {
                "instance_id": "known-failure",
                "reason": "immutable_license_or_image",
                "detail": "known immutable failure",
            }
        ],
    )
    menv.write_jsonl(
        root / "general-coding-dev-reservations.jsonl",
        [{"upstream_repository": "HTTPS://GITHUB.COM/DEV/RESERVED.GIT/"}],
    )

    existing_raw = next(row for row in rows if row["instance_id"] == "resume-python")
    existing = adapt_menvdata_multilang_row(existing_raw)
    existing.update(
        {
            "license_spdx": "MIT",
            "license_evidence_path": "licenses/resume.LICENSE",
            "license_evidence_sha256": "a" * 64,
            "license_evidence_method": "classified_immutable_license_file",
            "license_source_url": "https://example.test/license",
            "container_image": {
                "repo_tag": "example/resume:tag",
                "repo_digest": "example/resume@sha256:" + "b" * 64,
                "manifest_digest": "sha256:" + "b" * 64,
            },
        }
    )
    (root / "licenses").mkdir()
    (root / "licenses/resume.LICENSE").write_text("license", encoding="utf-8")
    menv._materialize_artifacts(root, existing)
    existing["_manifest_dir"] = str(root.resolve())
    menv.write_jsonl(_wave_paths(root, 2)["manifest"], [existing])

    verified = (
        _wave_paths(root, 1)["verification"] / "verified-cap-case" / "verified.json"
    )
    verified.parent.mkdir(parents=True)
    verified.write_text(
        json.dumps(
            {
                "case_id": "verified-cap-case",
                "source_id": SOURCE_ID,
                "dataset_revision": DATASET_REVISION,
                "primary_language": "go",
                "upstream_repository": "https://github.com/verified/capped.git",
            }
        ),
        encoding="utf-8",
    )
    return wave1_bytes


def test_wave2_freeze_isolated_exact_deterministic_and_policy_safe(
    tmp_path, monkeypatch
):
    rows = [
        _freeze_row("Python", "resume-python", "resume/repo"),
        _freeze_row("Go", "wave1-frozen", "wave1/repo"),
        _freeze_row("Rust", "known-failure", "known/failure"),
        _freeze_row("TypeScript", "dev-reserved", "dev/reserved"),
        _freeze_row("Go", "resume-cap", "resume/repo"),
        _freeze_row("Go", "verified-cap", "verified/capped"),
        _freeze_row("Go", "selected-go", "selected/go"),
        _freeze_row("JavaScript", "selected-js", "selected/js"),
        _freeze_row("Rust", "selected-rust", "selected/rust"),
        _freeze_row("TypeScript", "selected-ts", "selected/ts"),
    ]
    calls: list[str] = []

    def fake_license(control_root, candidate):
        calls.append(str(candidate["problem_id"]))
        evidence = control_root / "licenses" / f"{candidate['problem_id']}.LICENSE"
        evidence.parent.mkdir(exist_ok=True)
        evidence.write_text("MIT", encoding="utf-8")
        return {
            "spdx": "MIT",
            "license_evidence_path": evidence.relative_to(control_root).as_posix(),
            "license_evidence_sha256": menv.sha256_file(evidence),
            "license_evidence_method": "classified_immutable_license_file",
            "license_source_url": "https://example.test/license",
        }

    monkeypatch.setattr(menv, "_freeze_license", fake_license)
    monkeypatch.setattr(
        menv,
        "_docker_hub_digest",
        lambda image: {
            "repo_tag": image,
            "repo_digest": f"{image}@sha256:" + "c" * 64,
            "manifest_digest": "sha256:" + "c" * 64,
        },
    )

    selections = []
    for index, ordered_rows in enumerate((rows, list(reversed(rows)))):
        root = tmp_path / str(index)
        root.mkdir()
        wave1_bytes = _write_wave2_fixture(root, ordered_rows, monkeypatch)
        report = freeze_menvdata_multilang(
            root, wave=2, target_rows=5, repository_cap=1
        )
        frozen = menv.read_jsonl(_wave_paths(root, 2)["manifest"])
        selections.append([row["case_id"] for row in frozen])

        assert report["status"] == "strict_candidate_pool_ready"
        assert report["strict_capacity"] == 5
        assert report["language_targets"] == {
            "go": 1,
            "javascript": 1,
            "python": 1,
            "rust": 1,
            "typescript": 1,
        }
        assert max(report["selected_by_repository"].values()) <= 1
        assert report["repository_cap_policy"]["eligible_verified_receipts"] == 1
        assert _wave_paths(root, 1)["manifest"].read_bytes() == wave1_bytes
        assert report["manifest"]["path"] == (
            "menvdata-swe-multilang-wave2-candidate-source-manifest.jsonl"
        )
        assert _wave_paths(root, 2)["freeze_report"].is_file()
        assert _wave_paths(root, 2)["freeze_rejections"].is_file()
        reasons = {
            row["instance_id"]: row["reason"]
            for row in menv.read_jsonl(_wave_paths(root, 2)["freeze_rejections"])
        }
        assert reasons["wave1-frozen"] == "wave1_instance_frozen"
        assert reasons["known-failure"] == "known_wave1_immutable_failure"
        assert reasons["dev-reserved"] == "dev_repository_reserved"
        assert reasons["resume-cap"] == "repository_cap"
        assert reasons["verified-cap"] == "repository_cap"

    assert selections[0] == selections[1]
    assert not {"known-failure", "dev-reserved", "wave1-frozen"}.intersection(calls)


def test_wave2_prepare_uses_per_batch_artifacts_without_touching_wave1(
    tmp_path, monkeypatch
):
    patch = tmp_path / "target.patch"
    patch.write_text("diff --git a/a.py b/a.py\n-old\n+new\n", encoding="utf-8")
    row = {
        "case_id": "wave2-case",
        "primary_language": "python",
        "target_patch_path": "target.patch",
        "base_commit": "a" * 40,
        "container_image": {
            "repo_tag": "example/image:tag",
            "repo_digest": "example/image@sha256:" + "d" * 64,
        },
    }
    menv.write_jsonl(_wave_paths(tmp_path, 2)["manifest"], [row])
    wave1_executable = _wave_paths(tmp_path, 1)["executable_manifest"]
    wave1_executable.write_text("wave1-sentinel\n", encoding="utf-8")
    monkeypatch.setattr(
        menv,
        "_validated_commands",
        lambda _row, _root: {
            "install_command": "true",
            "compile_or_typecheck_command": "true",
            "targeted_test_command": "set -euo pipefail; pytest tests/test_x.py",
            "full_regression_command": "true",
        },
    )
    monkeypatch.setattr(menv, "_source_workdir", lambda _row, _root: "/testbed")
    monkeypatch.setattr(
        menv,
        "_docker_image_identity",
        lambda _tag: {
            "repo_tag": "example/image:tag",
            "repo_digest": "example/image@sha256:" + "d" * 64,
            "working_dir": "/testbed",
        },
    )

    report = prepare_menvdata_multilang(tmp_path, wave=2, limit=1)

    assert report["prepared"] == 1
    assert report["workflow"] == "explicit_per_batch_manifest"
    assert _wave_paths(tmp_path, 2)["executable_manifest"].is_file()
    assert _wave_paths(tmp_path, 2)["image_report"].is_file()
    assert _wave_paths(tmp_path, 2)["image_rejections"].is_file()
    assert wave1_executable.read_text(encoding="utf-8") == "wave1-sentinel\n"
    assert MAX_PREPARE_IMAGES == 25


def test_wave2_verify_targets_wave2_paths_and_keeps_network_disabled(
    tmp_path, monkeypatch
):
    observed = {}
    monkeypatch.setattr(
        menv,
        "prepare_menvdata_multilang",
        lambda control_root, **kwargs: {"requested": 1, "prepared": 1, "rejected": 0},
    )

    def fake_verify(candidates, output_dir, **kwargs):
        observed.update(
            {
                "candidates": candidates,
                "output_dir": output_dir,
                "resume": kwargs["resume"],
            }
        )
        return {"verified": 1, "rejected": 0, "skipped_completed": 0}

    monkeypatch.setattr(menv, "verify_candidates", fake_verify)
    monkeypatch.setattr(
        menv, "build_menvdata_multilang_status_report", lambda *_args, **_kwargs: {}
    )

    report = verify_menvdata_multilang(tmp_path, wave=2, limit=1)

    assert observed["candidates"] == _wave_paths(tmp_path, 2)["executable_manifest"]
    assert observed["output_dir"] == _wave_paths(tmp_path, 2)["verification"]
    assert observed["resume"] is True
    assert report["network_policy"] == "disabled"
    assert _wave_paths(tmp_path, 2)["preflight_report"].is_file()


def test_wave2_balanced_target_is_exact_for_209_rows():
    targets = _wave2_language_targets(209)

    assert sum(targets.values()) == 209
    assert set(targets) == set(menv.TARGET_LANGUAGES.values())
    assert max(targets.values()) - min(targets.values()) == 1


def test_cli_accepts_wave2_freeze_policy_options(tmp_path, monkeypatch, capsys):
    observed = {}

    def fake_freeze(control_root, **kwargs):
        observed.update({"control_root": control_root, **kwargs})
        return {"status": "ok"}

    monkeypatch.setattr(menv, "freeze_menvdata_multilang", fake_freeze)

    assert (
        menv.main(
            [
                "--control-root",
                str(tmp_path),
                "freeze",
                "--wave",
                "2",
                "--target-rows",
                "209",
                "--repo-cap",
                "15",
            ]
        )
        == 0
    )
    assert observed["wave"] == 2
    assert observed["target_rows"] == 209
    assert observed["repository_cap"] == 15
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}
