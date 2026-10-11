import json
from pathlib import Path

import yaml

from multi_tune_agent.phase1_splits import (
    assess_registered_seeds,
    assign_groups,
    build_v3,
    build_v4,
    load_existing_assignments,
    promote_requests,
    stable_split,
)


def _candidate(
    candidate_id: str,
    lineage: str,
    *,
    lanes: list[str] | None = None,
) -> dict:
    return {
        "id": candidate_id,
        "priority": "P0",
        "contract_family_id": f"CF-{candidate_id}",
        "source_lineage_id": lineage,
        "source": {
            "revision": "source",
            "test_path": "test_source.py",
            "test_id": "test_kernel",
            "source_language": "triton",
            "source_backend": "triton",
        },
        "contract": {
            "operator": "gemm",
            "shape": {"M": 1, "N": 2, "K": 3},
            "dtype": {"input": "bf16", "weight": "bf16", "output": "bf16"},
        },
        "oracle": {"tier": "pure_torch", "method": "torch.matmul"},
        "target_lanes": lanes or ["triton_gfx942", "hip_gfx942"],
    }


def _catalog(tmp_path: Path, candidates: list[dict]) -> dict:
    (tmp_path / "test_source.py").write_text("def test_kernel():\n    pass\n")
    return {
        "source_revisions": {
            "source": {
                "repository": "https://example.invalid/source.git",
                "branch": "main",
                "license": "MIT",
                "local_root": str(tmp_path),
            }
        },
        "candidates": candidates,
    }


def test_stable_assignment_does_not_change_when_candidates_are_appended():
    initial = [_candidate("cand-a", "lineage-a"), _candidate("cand-b", "lineage-b")]
    before = assign_groups(initial)
    after = assign_groups(initial + [_candidate("cand-c", "lineage-c")])

    assert {row["source_lineage_id"]: row["split"] for row in before} == {
        row["source_lineage_id"]: row["split"]
        for row in after
        if row["source_lineage_id"] != "lineage-c"
    }


def test_language_variants_of_source_lineage_share_one_group():
    rows = assign_groups(
        [
            _candidate("cand-triton", "shared", lanes=["triton_gfx942"]),
            _candidate("cand-hip", "shared", lanes=["hip_gfx942"]),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["candidate_ids"] == ["cand-hip", "cand-triton"]
    assert rows[0]["target_lanes"] == ["hip_gfx942", "triton_gfx942"]


def test_preserved_wave_lineage_is_train_even_if_hash_selects_held_out():
    lineage = next(
        f"lineage-{index}"
        for index in range(1000)
        if stable_split(
            f"lineage-{index}",
            salt="test",
            dev_fraction=0,
            held_out_fraction=0.99,
        )
        == "held_out"
    )

    rows = assign_groups(
        [_candidate("cand-existing", lineage)],
        preserve_train=[lineage],
        salt="test",
        dev_fraction=0,
        held_out_fraction=0.99,
    )

    assert rows[0]["split"] == "train"
    assert rows[0]["assignment_reason"] == "preserved_existing_train"


def test_prior_assignments_are_immutable_and_conflicts_fail(tmp_path):
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "dev_groups.jsonl").write_text(
        json.dumps({"source_lineage_id": "lineage-a"}) + "\n"
    )
    assignments = load_existing_assignments(prior)

    rows = assign_groups(
        [_candidate("cand-a", "lineage-a")],
        existing_assignments=assignments,
    )
    assert rows[0]["split"] == "dev"

    try:
        assign_groups(
            [_candidate("cand-a", "lineage-a")],
            preserve_train=["lineage-a"],
            existing_assignments=assignments,
        )
    except ValueError as exc:
        assert "conflict" in str(exc)
    else:
        raise AssertionError("conflicting frozen assignments must fail")


def test_held_out_candidates_never_become_generation_requests(tmp_path):
    held_out = _candidate("cand-held", "lineage-held")
    train = _candidate("cand-train", "lineage-train")
    catalog = _catalog(tmp_path, [held_out, train])
    groups = [
        {"source_lineage_id": "lineage-held", "split": "held_out"},
        {"source_lineage_id": "lineage-train", "split": "train"},
    ]

    requests, report = promote_requests(catalog, groups)

    assert {item["seed_provenance"]["source_lineage_id"] for item in requests} == {
        "lineage-train"
    }
    assert all(
        item["seed_provenance"]["gpu_verification_status"] == "not_run"
        for item in requests
    )
    assert all(
        item["recognized_contract"]["dimensions"] == {"M": 1, "N": 2, "K": 3}
        and item["recognized_contract"]["shapes"] == [[1, 2, 3]]
        and item["recognized_contract"]["format"] == "bf16"
        and item["recognized_contract"]["input_dtype"] == "bf16"
        and item["recognized_contract"]["weight_dtype"] == "bf16"
        and item["recognized_contract"]["output_dtype"] == "bf16"
        for item in requests
    )
    held_report = next(item for item in report if item["id"] == "cand-held")
    assert held_report["status"] == "reserved_held_out"


def test_incomplete_source_is_reported_not_fabricated(tmp_path):
    candidate = _candidate("cand-incomplete", "lineage-incomplete")
    del candidate["source"]["test_id"]
    catalog = _catalog(tmp_path, [candidate])

    requests, report = promote_requests(
        catalog,
        [{"source_lineage_id": "lineage-incomplete", "split": "train"}],
    )

    assert requests == []
    assert report[0]["status"] == "skipped"
    assert "test_id" in report[0]["reason"]


def test_mismatched_pinned_source_revision_is_not_promoted(tmp_path, monkeypatch):
    candidate = _candidate("cand-stale", "lineage-stale")
    catalog = _catalog(tmp_path, [candidate])
    catalog["source_revisions"]["source"]["git_sha"] = "pinned-sha"

    class Result:
        returncode = 0
        stdout = "different-sha\n"

    monkeypatch.setattr(
        "multi_tune_agent.phase1_splits.subprocess.run",
        lambda *args, **kwargs: Result(),
    )
    requests, report = promote_requests(
        catalog,
        [{"source_lineage_id": "lineage-stale", "split": "train"}],
    )

    assert requests == []
    assert report[0]["status"] == "skipped"
    assert "revision mismatch" in report[0]["reason"]


def test_legacy_registered_seed_missing_lineage_is_assessed_not_promoted(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "seeds": [
                    {
                        "id": "legacy-seed",
                        "architecture": "gfx942",
                        "source_test_path": "tests/test_op.py",
                    }
                ]
            }
        )
    )

    report = assess_registered_seeds([path])

    assert report[0]["status"] == "skipped"
    assert "source_lineage_id" in report[0]["reason"]
    assert "contract" in report[0]["reason"]


def test_build_v3_writes_reproducible_scoped_artifacts(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    candidates = [
        _candidate("cand-existing", "lineage-existing"),
        _candidate("cand-new", "lineage-new"),
        {
            **_candidate("cand-gfx950", "lineage-gfx950"),
            "target_lanes": ["triton_gfx950"],
        },
    ]
    catalog = _catalog(source_root, candidates)
    catalog_path = tmp_path / "candidates.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False))
    evidence_path = tmp_path / "production.yaml"
    evidence_path.write_text(
        yaml.safe_dump({"case_seed": {"source_lineage_id": "lineage-existing"}})
    )
    output = tmp_path / "v3"

    first = build_v3(
        catalog_path=catalog_path,
        output_dir=output,
        preserve_train_paths=[evidence_path],
    )
    first_files = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    second = build_v3(
        catalog_path=catalog_path,
        output_dir=output,
        preserve_train_paths=[evidence_path],
    )

    assert first == second
    assert first_files == {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    assert first["preserved_train_lineages"] == ["lineage-existing"]
    assert "lineage-gfx950" not in "".join(
        (output / name).read_text()
        for name in ("train_groups.jsonl", "dev_groups.jsonl", "held_out_groups.jsonl")
    )
    document = yaml.safe_load((output / "generation-requests.yaml").read_text())
    assert document["scope"]["lanes"] == {
        "triton_gfx942": 1000,
        "hip_gfx942": 1000,
    }
    assert document["selection"]["held_out_collection"] == "forbidden"


def test_build_v4_inherits_v3_assignments_and_excludes_held_out(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    candidates = [
        {**_candidate("cand-dev", "lineage-dev"), "top10_family": "gemm"},
        {**_candidate("cand-held", "lineage-held"), "top10_family": "mha"},
        {**_candidate("cand-new", "lineage-new"), "top10_family": "sampling"},
    ]
    catalog = _catalog(source_root, candidates)
    catalog_path = tmp_path / "top10.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False))
    prior = tmp_path / "v3"
    prior.mkdir()
    prior_rows = {
        "train": {"source_lineage_id": "lineage-old-train", "candidate_ids": ["old"]},
        "dev": {"source_lineage_id": "lineage-dev", "candidate_ids": ["old-dev"]},
        "held_out": {
            "source_lineage_id": "lineage-held",
            "candidate_ids": ["old-held"],
        },
    }
    for split, row in prior_rows.items():
        (prior / f"{split}_groups.jsonl").write_text(json.dumps(row) + "\n")

    output = tmp_path / "v4"
    manifest = build_v4(
        catalog_path=catalog_path,
        output_dir=output,
        prior_split_dir=prior,
        dev_fraction=0,
        held_out_fraction=0,
    )

    assignments = load_existing_assignments(output)
    assert assignments["lineage-old-train"] == "train"
    assert assignments["lineage-dev"] == "dev"
    assert assignments["lineage-held"] == "held_out"
    assert assignments["lineage-new"] == "train"
    requests = yaml.safe_load((output / "generation-requests.yaml").read_text())[
        "requests"
    ]
    assert {row["seed_provenance"]["source_lineage_id"] for row in requests} == {
        "lineage-dev",
        "lineage-new",
    }
    assert all(row["seed_provenance"]["split_version"] == "v4" for row in requests)
    assert manifest["version"] == "v4"
    assert manifest["prior_split_dir"] == str(prior.resolve())
