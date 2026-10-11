"""Deterministic, append-safe Phase 1 lineage splits and request promotion.

This module deliberately operates on source lineages, not individual cases.
It performs no GPU validation and never promotes an entry to ``validated_case``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


SPLITS = ("train", "dev", "held_out")
COLLECTABLE_SPLITS = frozenset(("train", "dev"))
REQUIRED_SOURCE_FIELDS = (
    "revision",
    "test_path",
    "test_id",
    "source_language",
    "source_backend",
)
REQUIRED_REGISTERED_SEED_FIELDS = (
    "source_lineage_id",
    "contract_family_id",
    "source_test_id",
    "source_test_path",
    "source_language",
    "source_backend",
    "contract",
    "oracle",
    "target_lanes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(stream)
        return json.load(stream)


def extract_lineages(value: object) -> set[str]:
    """Recursively extract source lineage IDs from JSON/YAML evidence."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        lineage = value.get("source_lineage_id")
        if isinstance(lineage, str) and lineage.strip():
            found.add(lineage.strip())
        for child in value.values():
            found.update(extract_lineages(child))
    elif isinstance(value, list):
        for child in value:
            found.update(extract_lineages(child))
    return found


def load_existing_assignments(split_dir: Path | None) -> dict[str, str]:
    """Load immutable assignments from an earlier split directory."""
    assignments: dict[str, str] = {}
    if split_dir is None:
        return assignments
    for split in SPLITS:
        path = split_dir / f"{split}_groups.jsonl"
        if not path.exists():
            continue
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            lineage = str(row.get("source_lineage_id", "")).strip()
            if not lineage:
                raise ValueError(f"{path}:{line_number}: missing source_lineage_id")
            previous = assignments.setdefault(lineage, split)
            if previous != split:
                raise ValueError(f"lineage {lineage!r} appears in multiple splits")
    return assignments


def load_prior_groups(split_dir: Path | None) -> list[dict[str, Any]]:
    """Load complete prior rows so a successor split is append-only."""
    if split_dir is None:
        return []
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = split_dir / f"{split}_groups.jsonl"
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                row = dict(json.loads(raw))
                row["split"] = split
                rows.append(row)
    return sorted(rows, key=lambda row: str(row["source_lineage_id"]))


def stable_split(
    lineage: str,
    *,
    salt: str,
    dev_fraction: float,
    held_out_fraction: float,
) -> str:
    """Assign a lineage independently so appending inputs cannot move it."""
    if not 0 <= dev_fraction <= 1 or not 0 <= held_out_fraction <= 1:
        raise ValueError("split fractions must be between zero and one")
    if dev_fraction + held_out_fraction >= 1:
        raise ValueError("dev_fraction + held_out_fraction must be less than one")
    value = int.from_bytes(
        hashlib.sha256(f"{salt}\0{lineage}".encode()).digest()[:8], "big"
    ) / 2**64
    if value < held_out_fraction:
        return "held_out"
    if value < held_out_fraction + dev_fraction:
        return "dev"
    return "train"


def _gfx942_candidates(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for raw in catalog.get("candidates", []):
        candidate = dict(raw)
        lanes = [
            lane
            for lane in candidate.get("target_lanes", [])
            if lane in {"triton_gfx942", "hip_gfx942"}
        ]
        if candidate.get("priority") in {"P0", "P1"} and lanes:
            candidate["target_lanes"] = lanes
            candidates.append(candidate)
    return candidates


def assign_groups(
    candidates: Iterable[Mapping[str, Any]],
    *,
    preserve_train: Iterable[str] = (),
    existing_assignments: Mapping[str, str] | None = None,
    salt: str = "geak-phase1-gfx942-v3",
    dev_fraction: float = 0.15,
    held_out_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    """Group language variants by source lineage and assign whole groups."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        lineage = str(candidate.get("source_lineage_id", "")).strip()
        if not lineage:
            raise ValueError(f"candidate {candidate.get('id')!r} has no source lineage")
        grouped[lineage].append(candidate)

    frozen_train = set(preserve_train)
    previous = dict(existing_assignments or {})
    conflicts = frozen_train.intersection(
        lineage for lineage, split in previous.items() if split != "train"
    )
    if conflicts:
        raise ValueError(
            "preserved train lineages conflict with prior assignments: "
            + ", ".join(sorted(conflicts))
        )

    rows = []
    for lineage in sorted(grouped):
        members = grouped[lineage]
        split = (
            "train"
            if lineage in frozen_train
            else previous.get(lineage)
            or stable_split(
                lineage,
                salt=salt,
                dev_fraction=dev_fraction,
                held_out_fraction=held_out_fraction,
            )
        )
        if split not in SPLITS:
            raise ValueError(f"invalid existing split {split!r} for {lineage}")
        lanes = sorted(
            {
                lane
                for candidate in members
                for lane in candidate.get("target_lanes", [])
            }
        )
        rows.append(
            {
                "source_lineage_id": lineage,
                "split": split,
                "candidate_ids": sorted(str(item["id"]) for item in members),
                "contract_family_ids": sorted(
                    {str(item["contract_family_id"]) for item in members}
                ),
                "implementation_family_ids": sorted(
                    {
                        str(item.get("implementation_family_id"))
                        if item.get("implementation_family_id")
                        else f"independent:{item['id']}:{lane.split('_', 1)[0]}"
                        for item in members
                        for lane in item.get("target_lanes", [])
                    }
                ),
                "target_lanes": lanes,
                "top10_families": sorted(
                    {
                        str(item["top10_family"])
                        for item in members
                        if item.get("top10_family")
                    }
                ),
                "assignment_reason": (
                    "preserved_existing_train"
                    if lineage in frozen_train
                    else "preserved_prior_assignment"
                    if lineage in previous
                    else "stable_hash"
                ),
            }
        )
    return rows


def inherit_prior_groups(
    groups: Iterable[Mapping[str, Any]],
    prior_groups: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge new members into prior rows without changing prior assignments."""
    merged = {
        str(row["source_lineage_id"]): dict(row)
        for row in prior_groups
    }
    list_fields = (
        "candidate_ids",
        "contract_family_ids",
        "implementation_family_ids",
        "target_lanes",
        "top10_families",
    )
    for raw in groups:
        row = dict(raw)
        lineage = str(row["source_lineage_id"])
        prior = merged.get(lineage)
        if prior is None:
            merged[lineage] = row
            continue
        if prior["split"] != row["split"]:
            raise ValueError(f"prior split changed for lineage {lineage!r}")
        for field in list_fields:
            prior[field] = sorted(set(prior.get(field, ())).union(row.get(field, ())))
        prior["assignment_reason"] = "preserved_prior_assignment"
    return [merged[lineage] for lineage in sorted(merged)]


def _source_path(
    candidate: Mapping[str, Any], source_revisions: Mapping[str, Any]
) -> tuple[Path | None, str | None]:
    source = candidate.get("source")
    if not isinstance(source, Mapping):
        return None, "missing source mapping"
    missing = [field for field in REQUIRED_SOURCE_FIELDS if not source.get(field)]
    if missing:
        return None, "missing source fields: " + ", ".join(missing)
    revision = source_revisions.get(source["revision"])
    if not isinstance(revision, Mapping) or not revision.get("local_root"):
        return None, f"source revision {source['revision']!r} has no local_root"
    root = Path(str(revision["local_root"])).resolve()
    pinned_sha = revision.get("git_sha")
    if pinned_sha:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None, f"cannot verify source revision at {root}"
        actual_sha = result.stdout.strip()
        if actual_sha != pinned_sha:
            return None, (
                f"source revision mismatch: pinned {pinned_sha}, actual {actual_sha}"
            )
    path = (root / str(source["test_path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None, "source test path escapes local_root"
    if not path.is_file():
        return None, f"source test file not found: {path}"
    return path, None


def _request(candidate: Mapping[str, Any], language: str) -> str:
    contract = _canonical_json(candidate["contract"])
    oracle = _canonical_json(candidate["oracle"])
    operator = candidate["contract"]["operator"]
    return (
        f"Generate a standalone {language.upper()} {operator} kernel for AMD gfx942. "
        f"Frozen contract JSON: {contract}. Independent oracle JSON: {oracle}. "
        "Implement independently; do not call AITER, CK, ASM, OPUS, hipBLASLt, "
        "torch operators, or the oracle at runtime. Create a deterministic GEAK "
        "harness and require compile, correctness, performance, and "
        "fresh-workspace verification."
    )


def _recognized_contract(
    candidate: Mapping[str, Any], language: str
) -> dict[str, Any]:
    """Translate the frozen candidate contract into the CLI recognizer schema."""
    contract = candidate["contract"]
    shape = contract.get("shape")
    source_shapes = contract.get("shapes")
    recognized_shapes: list[list[Any]]
    if isinstance(shape, Mapping):
        dimensions = dict(shape)
        recognized_shapes = [list(dimensions.values())]
    elif isinstance(shape, (list, tuple)) and shape:
        dimensions = {
            f"dim_{index}": value for index, value in enumerate(shape)
        }
        recognized_shapes = [list(shape)]
    elif isinstance(source_shapes, list) and source_shapes:
        first_shape = source_shapes[0]
        if isinstance(first_shape, Mapping):
            dimensions = dict(first_shape)
            keys = list(dimensions)
            recognized_shapes = [
                [item[key] for key in keys]
                for item in source_shapes
                if isinstance(item, Mapping) and all(key in item for key in keys)
            ]
        elif isinstance(first_shape, (list, tuple)) and first_shape:
            dimensions = {
                f"dim_{index}": value for index, value in enumerate(first_shape)
            }
            recognized_shapes = [
                list(item)
                for item in source_shapes
                if isinstance(item, (list, tuple))
            ]
        else:
            raise ValueError(f"candidate {candidate.get('id')!r} has no usable shape")
    else:
        raise ValueError(f"candidate {candidate.get('id')!r} has no usable shape")

    dtype = contract.get("dtype")
    dtype = dtype if isinstance(dtype, Mapping) else {}
    input_dtype = (
        contract.get("input_dtype")
        or dtype.get("input")
        or dtype.get("points")
        or dtype.get("activation")
    )
    weight_dtype = contract.get("weight_dtype") or dtype.get("weight")
    output_dtype = contract.get("output_dtype") or dtype.get("output")
    if not input_dtype:
        raise ValueError(f"candidate {candidate.get('id')!r} has no input dtype")

    scale = contract.get("scale")
    scale = scale if isinstance(scale, Mapping) else {}
    recognized = {
        "operator": contract["operator"],
        "target_gpu": "gfx942",
        "language": language,
        "format": input_dtype,
        "input_dtype": input_dtype,
        "weight_dtype": weight_dtype,
        "output_dtype": output_dtype,
        "dimensions": dimensions,
        "shapes": recognized_shapes,
        "input_scale_granularity": (
            contract.get("input_scale_granularity")
            or scale.get("activation")
            or scale.get("input")
            or scale.get("scale")
        ),
        "weight_scale_granularity": (
            contract.get("weight_scale_granularity")
            or scale.get("weight")
        ),
        "block_size": contract.get("block_size"),
        "contract": contract,
    }
    return {key: value for key, value in recognized.items() if value is not None}


def promote_requests(
    catalog: Mapping[str, Any],
    groups: Iterable[Mapping[str, Any]],
    *,
    split_version: str = "v3",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create collection requests only for complete train/dev gfx942 sources."""
    split_by_lineage = {
        str(group["source_lineage_id"]): str(group["split"]) for group in groups
    }
    revisions = catalog.get("source_revisions", {})
    requests: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for candidate in _gfx942_candidates(catalog):
        candidate_id = str(candidate.get("id", ""))
        lineage = str(candidate.get("source_lineage_id", ""))
        split = split_by_lineage[lineage]
        path, error = _source_path(candidate, revisions)
        if error:
            report.append({"id": candidate_id, "status": "skipped", "reason": error})
            continue
        if not isinstance(candidate.get("contract"), Mapping) or not isinstance(
            candidate.get("oracle"), Mapping
        ):
            report.append(
                {
                    "id": candidate_id,
                    "status": "skipped",
                    "reason": "missing contract or oracle mapping",
                }
            )
            continue
        assert path is not None
        source = candidate["source"]
        revision = revisions[source["revision"]]
        source_hash = _sha256(path)
        if split == "held_out":
            report.append(
                {
                    "id": candidate_id,
                    "status": "reserved_held_out",
                    "source_test_sha256": source_hash,
                }
            )
            continue
        for lane in candidate["target_lanes"]:
            language = lane.split("_", 1)[0]
            requests.append(
                {
                    "id": f"phase1-{language}-gfx942-{candidate_id.removeprefix('cand-')}",
                    "request": _request(candidate, language),
                    "seed_provenance": {
                        "source_repo": revision.get("repository"),
                        "source_branch": revision.get("branch"),
                        "source_sha": revision.get("git_sha"),
                        "source_license": revision.get("license"),
                        "source_candidate_id": candidate_id,
                        "source_test_path": source["test_path"],
                        "source_test_id": source["test_id"],
                        "source_test_sha256": source_hash,
                        "contract_family_id": candidate["contract_family_id"],
                        "top10_family": candidate.get("top10_family"),
                        "source_lineage_id": lineage,
                        "implementation_family_id": (
                            candidate.get("implementation_family_id")
                            or f"independent:{candidate_id}:{language}"
                        ),
                        "split_group": split,
                        "split_version": split_version,
                        "target_lane": lane,
                        "gpu_verification_status": "not_run",
                    },
                    "recognized_contract": _recognized_contract(candidate, language),
                }
            )
        report.append(
            {
                "id": candidate_id,
                "status": "promoted_to_generation_requests",
                "split": split,
                "request_count": len(candidate["target_lanes"]),
                "source_test_sha256": source_hash,
            }
        )
    requests.sort(key=lambda item: item["id"])
    report.sort(key=lambda item: item["id"])
    return requests, report


def assess_registered_seeds(
    paths: Iterable[Path],
) -> list[dict[str, Any]]:
    """Assess legacy registered seeds without filling absent contract fields."""
    report: list[dict[str, Any]] = []
    for path in paths:
        document = _load(path)
        if not isinstance(document, Mapping):
            raise ValueError(f"registered seed catalog must be a mapping: {path}")
        for raw in document.get("seeds", []):
            seed = dict(raw)
            seed_id = str(seed.get("id", ""))
            if seed.get("architecture") != "gfx942":
                report.append(
                    {
                        "id": seed_id,
                        "catalog": str(path.resolve()),
                        "status": "deferred_out_of_scope",
                        "reason": "architecture is not gfx942",
                    }
                )
                continue
            missing = [field for field in REQUIRED_REGISTERED_SEED_FIELDS if not seed.get(field)]
            if missing:
                report.append(
                    {
                        "id": seed_id,
                        "catalog": str(path.resolve()),
                        "status": "skipped",
                        "reason": "missing promotion fields: " + ", ".join(missing),
                    }
                )
                continue
            report.append(
                {
                    "id": seed_id,
                    "catalog": str(path.resolve()),
                    "status": "eligible_but_not_promoted",
                    "reason": (
                        "registered seed schema requires explicit conversion into "
                        "the candidate catalog before split assignment"
                    ),
                }
            )
    return sorted(report, key=lambda item: (item["catalog"], item["id"]))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_v3(
    *,
    catalog_path: Path,
    output_dir: Path,
    preserve_train_paths: Iterable[Path],
    registered_seed_paths: Iterable[Path] = (),
    prior_split_dir: Path | None = None,
    salt: str = "geak-phase1-gfx942-v3",
    dev_fraction: float = 0.15,
    held_out_fraction: float = 0.25,
    split_version: str = "v3",
) -> dict[str, Any]:
    catalog = _load(catalog_path)
    if not isinstance(catalog, Mapping):
        raise ValueError("candidate catalog must be a mapping")
    preserved: set[str] = set()
    for path in preserve_train_paths:
        preserved.update(extract_lineages(_load(path)))
    new_groups = assign_groups(
        _gfx942_candidates(catalog),
        preserve_train=preserved,
        existing_assignments=load_existing_assignments(prior_split_dir),
        salt=salt,
        dev_fraction=dev_fraction,
        held_out_fraction=held_out_fraction,
    )
    groups = inherit_prior_groups(new_groups, load_prior_groups(prior_split_dir))
    requests, promotion_report = promote_requests(
        catalog, groups, split_version=split_version
    )
    promotion_report.extend(assess_registered_seeds(registered_seed_paths))
    promotion_report.sort(key=lambda item: (item.get("catalog", ""), item["id"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = [group for group in groups if group["split"] == split]
        text = "".join(_canonical_json(row) + "\n" for row in rows)
        (output_dir / f"{split}_groups.jsonl").write_text(text, encoding="utf-8")
    request_document = {
        "version": 1,
        "schema_version": "geak_phase1_generation_requests_v1",
        "scope": {
            "architecture": "gfx942",
            "lanes": {"triton_gfx942": 1000, "hip_gfx942": 1000},
            "gfx950": "deferred",
            "general_coding_replay": "deferred",
        },
        "split_version": split_version,
        "selection": {
            "priorities": ["P0", "P1"],
            "collectable_splits": sorted(COLLECTABLE_SPLITS),
            "held_out_collection": "forbidden",
            "gpu_verification": "not_run",
        },
        "requests": requests,
    }
    request_path = output_dir / "generation-requests.yaml"
    request_path.write_text(
        yaml.safe_dump(request_document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _write_json(output_dir / "promotion_report.json", promotion_report)

    files = [
        "train_groups.jsonl",
        "dev_groups.jsonl",
        "held_out_groups.jsonl",
        "generation-requests.yaml",
        "promotion_report.json",
    ]
    counts = {split: sum(group["split"] == split for group in groups) for split in SPLITS}
    manifest = {
        "schema_version": "geak_phase1_split_manifest_v2",
        "version": split_version,
        "architecture": "gfx942",
        "assignment_unit": "source_lineage_id",
        "assignment": {
            "method": "sha256_threshold",
            "salt": salt,
            "dev_fraction": dev_fraction,
            "held_out_fraction": held_out_fraction,
            "append_safe": True,
        },
        "source_catalog": str(catalog_path.resolve()),
        "source_catalog_sha256": _sha256(catalog_path),
        "prior_split_dir": str(prior_split_dir.resolve()) if prior_split_dir else None,
        "prior_manifest_sha256": (
            _sha256(prior_split_dir / "manifest.json")
            if prior_split_dir and (prior_split_dir / "manifest.json").is_file()
            else None
        ),
        "preserved_train_evidence": [str(path.resolve()) for path in preserve_train_paths],
        "preserved_train_lineages": sorted(preserved),
        "assessed_registered_seed_catalogs": [
            str(path.resolve()) for path in registered_seed_paths
        ],
        "counts": counts,
        "generation_request_count": len(requests),
        "held_out_collection_forbidden": True,
        "file_sha256": {name: _sha256(output_dir / name) for name in files},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_v4(
    *,
    catalog_path: Path,
    output_dir: Path,
    prior_split_dir: Path,
    preserve_train_paths: Iterable[Path] = (),
    registered_seed_paths: Iterable[Path] = (),
    salt: str = "geak-phase1-gfx942-v4",
    dev_fraction: float = 0.15,
    held_out_fraction: float = 0.25,
) -> dict[str, Any]:
    """Build append-only v4 while inheriting every v3 lineage assignment."""
    if not prior_split_dir.is_dir():
        raise ValueError("v4 requires an existing prior split directory")
    return build_v3(
        catalog_path=catalog_path,
        output_dir=output_dir,
        preserve_train_paths=preserve_train_paths,
        registered_seed_paths=registered_seed_paths,
        prior_split_dir=prior_split_dir,
        salt=salt,
        dev_fraction=dev_fraction,
        held_out_fraction=held_out_fraction,
        split_version="v4",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preserve-train",
        type=Path,
        action="append",
        default=[],
        help="JSON/YAML production evidence whose lineages must remain train",
    )
    parser.add_argument("--prior-split-dir", type=Path)
    parser.add_argument(
        "--registered-seeds",
        type=Path,
        action="append",
        default=[],
        help="legacy registered seed catalogs to assess without fabricating fields",
    )
    parser.add_argument("--salt", default="geak-phase1-gfx942-v3")
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--held-out-fraction", type=float, default=0.25)
    parser.add_argument("--split-version", choices=("v3", "v4"), default="v3")
    args = parser.parse_args(argv)
    kwargs = {
        "catalog_path": args.catalog,
        "output_dir": args.output_dir,
        "preserve_train_paths": args.preserve_train,
        "registered_seed_paths": args.registered_seeds,
        "prior_split_dir": args.prior_split_dir,
        "salt": args.salt,
        "dev_fraction": args.dev_fraction,
        "held_out_fraction": args.held_out_fraction,
    }
    if args.split_version == "v4":
        if args.prior_split_dir is None:
            parser.error("--split-version v4 requires --prior-split-dir")
        if args.salt == "geak-phase1-gfx942-v3":
            kwargs["salt"] = "geak-phase1-gfx942-v4"
        manifest = build_v4(**kwargs)
    else:
        manifest = build_v3(**kwargs)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
