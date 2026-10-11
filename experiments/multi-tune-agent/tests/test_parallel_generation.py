from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from multi_tune_agent.parallel_generation import (
    build_parser,
    initialize_from_base_catalog,
    load_requests,
    merge_catalogs,
    prepare_shards,
    run_shards,
    shard_requests,
)


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base-config.yaml"
    _write_yaml(
        path,
        {
            "geak_root": str(tmp_path / "geak"),
            "cases_path": str(tmp_path / "unused.yaml"),
            "trajectory_root": str(tmp_path / "unused-runs"),
            "generated_template_root": str(tmp_path / "unused-templates"),
            "bootstrap_auto_promote": True,
        },
    )
    return path


def _requests(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": f"case-{index:02d}",
            "request": f"generate case {index}",
            "seed_provenance": {"split_group": "train"},
        }
        for index in range(count)
    ]


def test_load_and_shard_requests_are_deterministic_and_exclude_held_out(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "requests.yaml"
    requests = list(reversed(_requests(10)))
    requests.extend(
        [
            {
                "id": f"aaa-dev-{index:02d}",
                "request": f"generate dev case {index}",
                "seed_provenance": {"split_group": "dev"},
            }
            for index in range(3)
        ]
    )
    _write_yaml(manifest, {"version": 1, "requests": requests})

    loaded = load_requests(manifest, smoke_limit=8)
    dev = load_requests(manifest, smoke_limit=2, split="dev")
    shards = shard_requests(loaded)

    assert [item["id"] for item in loaded] == [f"case-{i:02d}" for i in range(8)]
    assert [item["id"] for item in dev] == ["aaa-dev-00", "aaa-dev-01"]
    assert [item["id"] for item in shards[1]] == ["case-00", "case-07"]
    assert [item["id"] for item in shards[7]] == ["case-06"]
    assert build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--manifest",
            "requests.yaml",
            "--base-catalog",
            "base.yaml",
            "--output-catalog",
            "cases.yaml",
            "--work-root",
            "work",
        ]
    ).split == "train"

    requests[0]["seed_provenance"] = {"split_group": "held_out"}
    _write_yaml(manifest, {"version": 1, "requests": requests})
    with pytest.raises(ValueError, match="held_out request is forbidden"):
        load_requests(manifest)


def test_smoke_sample_alternates_target_lanes(tmp_path: Path) -> None:
    manifest = tmp_path / "requests.yaml"
    requests = [
        {
            "id": f"{lane}-{index}",
            "request": f"generate {lane} {index}",
            "seed_provenance": {
                "split_group": "train",
                "target_lane": lane,
            },
        }
        for lane in ("triton_gfx942", "hip_gfx942")
        for index in range(4)
    ]
    _write_yaml(manifest, {"version": 1, "requests": requests})

    selected = load_requests(manifest, smoke_limit=7)

    assert [
        item["seed_provenance"]["target_lane"] for item in selected
    ] == [
        "hip_gfx942",
        "triton_gfx942",
        "hip_gfx942",
        "triton_gfx942",
        "hip_gfx942",
        "triton_gfx942",
        "hip_gfx942",
    ]


def test_load_requests_can_select_only_top10_families(tmp_path: Path) -> None:
    manifest = tmp_path / "requests.yaml"
    requests = _requests(3)
    requests[1]["seed_provenance"]["top10_family"] = "mha"
    _write_yaml(manifest, {"version": 1, "requests": requests})

    selected = load_requests(manifest, top10_only=True)

    assert [item["id"] for item in selected] == ["case-01"]


def test_prepare_and_run_shards_use_isolated_paths_and_mocked_subprocesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "requests.yaml"
    requests = _requests(8)
    _write_yaml(manifest, {"version": 1, "schema_version": "test", "requests": requests})
    shards = prepare_shards(
        _base_config(tmp_path), manifest, tmp_path / "work", requests
    )
    calls: list[tuple[list[str], Path]] = []

    class Process:
        def wait(self) -> int:
            return 0

    def popen(command, *, cwd, stdout, stderr):
        calls.append((command, cwd))
        stdout.write("mock generation\n")
        return Process()

    monkeypatch.setattr(
        "multi_tune_agent.parallel_generation.subprocess.Popen", popen
    )
    result = run_shards(shards, python="/mock/python")

    assert result == {gpu_id: 0 for gpu_id in range(1, 8)}
    assert len(calls) == 7
    assert all(call[0][:3] == ["/mock/python", "-m", "multi_tune_agent.cli"] for call in calls)
    configs = [yaml.safe_load(shards[gpu]["config"].read_text()) for gpu in shards]
    assert {config["gpu_ids"] for config in configs} == {str(i) for i in range(1, 8)}
    assert len({config["trajectory_root"] for config in configs}) == 7
    assert len({config["generated_template_root"] for config in configs}) == 7
    assert all(config["sft_enabled"] is False for config in configs)
    assert all(shards[gpu]["log"].read_text().endswith("mock generation\n") for gpu in shards)


def _trusted_shard(
    tmp_path: Path,
    case_id: str,
    *,
    status: str = "generated",
    split: str = "train",
) -> dict[str, Path]:
    root = tmp_path / case_id
    template_root = root / "generated-templates"
    contract_hash = ("a" if case_id.endswith("a") else "b") * 64
    task_dir = template_root / contract_hash
    (task_dir / "scripts").mkdir(parents=True)
    for relative in ("config.yaml", "kernel.py", "scripts/task_runner.py"):
        (task_dir / relative).write_text("ok\n", encoding="utf-8")
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "contract_hash": contract_hash,
                "trust": {
                    "trusted": True,
                    "static_valid": True,
                    "compiled": True,
                    "correct": True,
                    "performance_valid": True,
                    "commands": {
                        name: {
                            "ok": True,
                            "returncode": 0,
                            "timed_out": False,
                        }
                        for name in ("compile", "correctness", "performance")
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    templates = template_root / "templates.yaml"
    _write_yaml(
        templates,
        {
            "templates": [
                {
                    "contract_hash": contract_hash,
                    "template_path": str(task_dir),
                }
            ]
        },
    )
    catalog = root / "generated-cases.yaml"
    _write_yaml(
        catalog,
        {
            "tasks": [
                {
                    "id": case_id,
                    "type": "aiter_generated",
                    "kernel_path": str(task_dir),
                    "contract_hash": contract_hash,
                    "provenance": {
                        "contract_hash": contract_hash,
                        "generation_method": "test",
                        "case_seed": {"split_group": split},
                    },
                }
            ]
        },
    )
    manifest = root / "generation-requests.yaml"
    _write_yaml(manifest, {"version": 1, "requests": []})
    trajectory = root / "trajectories"
    results = trajectory / "requests" / "generation-requests-generation-results.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text(
        json.dumps({"case_id": case_id, "status": status}) + "\n",
        encoding="utf-8",
    )
    return {
        "catalog": catalog,
        "templates": templates,
        "manifest": manifest,
        "trajectory_root": trajectory,
    }


def test_merge_catalogs_accepts_only_successful_verified_tasks_and_is_idempotent(
    tmp_path: Path,
) -> None:
    good = _trusted_shard(tmp_path, "case-a")
    failed = _trusted_shard(tmp_path, "case-b", status="failed")
    output = tmp_path / "production-cases.yaml"
    _write_yaml(output, {"description": "preserved", "tasks": []})
    requests = [
        {"id": "case-a", "request": "a"},
        {"id": "case-b", "request": "b"},
    ]

    assert merge_catalogs(output, {1: good, 2: failed}, requests) == 1
    first = output.read_text(encoding="utf-8")
    assert merge_catalogs(output, {1: good, 2: failed}, requests) == 1

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert output.read_text(encoding="utf-8") == first
    assert payload["description"] == "preserved"
    assert [task["id"] for task in payload["tasks"]] == ["case-a"]
    assert Path(payload["tasks"][0]["kernel_path"]).is_absolute()


def test_merge_rejects_catalog_task_without_verified_registry(tmp_path: Path) -> None:
    shard = _trusted_shard(tmp_path, "case-a")
    _write_yaml(shard["templates"], {"templates": []})

    with pytest.raises(ValueError, match="not backed by its verified registry"):
        merge_catalogs(
            tmp_path / "production.yaml",
            {1: shard},
            [{"id": "case-a", "request": "a"}],
        )


def test_merge_rejects_task_from_unselected_split(tmp_path: Path) -> None:
    shard = _trusted_shard(tmp_path, "case-a", split="dev")

    with pytest.raises(ValueError, match="does not match selected split 'train'"):
        merge_catalogs(
            tmp_path / "production.yaml",
            {1: shard},
            [{"id": "case-a", "request": "a"}],
            split="train",
        )


def test_initialize_from_trusted_base_is_atomic_and_preserves_ids(
    tmp_path: Path,
) -> None:
    first = _trusted_shard(tmp_path, "case-a")
    second = _trusted_shard(tmp_path, "case-b")
    first_task = yaml.safe_load(first["catalog"].read_text())["tasks"][0]
    second_task = yaml.safe_load(second["catalog"].read_text())["tasks"][0]
    base = tmp_path / "base.yaml"
    output = tmp_path / "output.yaml"
    _write_yaml(base, {"wave": "trusted-300", "tasks": [first_task, second_task]})

    assert initialize_from_base_catalog(base, output, split="train") == 2
    before = output.read_text(encoding="utf-8")
    assert initialize_from_base_catalog(base, output, split="train") == 2

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert output.read_text(encoding="utf-8") == before
    assert payload["wave"] == "trusted-300"
    assert [task["id"] for task in payload["tasks"]] == ["case-a", "case-b"]


def test_initialize_rejects_incomplete_trust_and_split_mismatch(
    tmp_path: Path,
) -> None:
    shard = _trusted_shard(tmp_path, "case-a")
    task = yaml.safe_load(shard["catalog"].read_text())["tasks"][0]
    base = tmp_path / "base.yaml"
    _write_yaml(base, {"tasks": [task]})
    metadata_path = Path(task["kernel_path"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["trust"]["performance_valid"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="complete metadata trust flags"):
        initialize_from_base_catalog(base, tmp_path / "output.yaml")

    replacement = _trusted_shard(tmp_path, "case-a-dev", split="dev")
    dev_task = yaml.safe_load(replacement["catalog"].read_text())["tasks"][0]
    _write_yaml(base, {"tasks": [dev_task]})
    with pytest.raises(ValueError, match="does not match selected split 'train'"):
        initialize_from_base_catalog(base, tmp_path / "output.yaml", split="train")


def test_generated_task_cannot_displace_seeded_base_id(tmp_path: Path) -> None:
    base_shard = _trusted_shard(tmp_path, "case-a")
    base_task = yaml.safe_load(base_shard["catalog"].read_text())["tasks"][0]
    output = tmp_path / "production.yaml"
    _write_yaml(output, {"tasks": [base_task]})
    generated = _trusted_shard(tmp_path, "other-case-a")
    generated_task = yaml.safe_load(generated["catalog"].read_text())["tasks"][0]
    generated_task["id"] = "case-a"
    _write_yaml(generated["catalog"], {"tasks": [generated_task]})
    results = (
        generated["trajectory_root"]
        / "requests"
        / "generation-requests-generation-results.jsonl"
    )
    results.write_text(
        json.dumps({"case_id": "case-a", "status": "generated"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="would displace existing ID"):
        merge_catalogs(
            output,
            {1: generated},
            [{"id": "case-a", "request": "new"}],
            split="train",
        )
