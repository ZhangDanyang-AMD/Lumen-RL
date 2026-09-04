from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from geak_utils.local_templates import (
    VerifiedTemplateRecord,
    find_verified_template,
    load_verified_templates,
    register_verified_template,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def make_template(root: Path, contract_hash: str = HASH_A, *, trusted=True) -> Path:
    root.mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "config.yaml").write_text("source_file_path: [kernel.py]\n")
    (root / "kernel.py").write_text("def candidate():\n    pass\n")
    (root / "scripts" / "task_runner.py").write_text("pass\n")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "contract_hash": contract_hash,
                "trust": {"trusted": trusted},
            }
        )
    )
    return root


def make_record(
    template: Path, contract_hash: str = HASH_A
) -> VerifiedTemplateRecord:
    return VerifiedTemplateRecord(
        contract_hash=contract_hash,
        operator="gemm",
        template_path=template,
        architecture="gfx942",
        language="triton",
        backend="aiter",
        provenance={"generator": "unit-test"},
        direction="forward",
    )


def test_missing_and_empty_registry(tmp_path: Path) -> None:
    missing = tmp_path / "registry.yaml"
    assert load_verified_templates(missing) == []
    missing.write_text("")
    assert load_verified_templates(missing) == []
    assert find_verified_template(missing, HASH_A) is None


def test_round_trip_sort_find_and_preserve_unknown_root_keys(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text("version: 7\nowner: local\n")
    template_b = make_template(tmp_path / "templates" / "b", HASH_B)
    template_a = make_template(tmp_path / "templates" / "a", HASH_A)

    register_verified_template(registry, make_record(template_b, HASH_B))
    registered_a = register_verified_template(registry, make_record(template_a, HASH_A))

    loaded = load_verified_templates(registry)
    assert [record.contract_hash for record in loaded] == [HASH_A, HASH_B]
    assert loaded[0] == registered_a
    assert loaded[0].case_type == "aiter_generated"
    assert find_verified_template(registry, HASH_B) == loaded[1]
    payload = yaml.safe_load(registry.read_text())
    assert payload["version"] == 7
    assert payload["owner"] == "local"
    assert [item["contract_hash"] for item in payload["templates"]] == [HASH_A, HASH_B]


def test_relative_template_path_resolves_from_registry_parent(tmp_path: Path) -> None:
    registry = tmp_path / "state" / "registry.yaml"
    template = make_template(tmp_path / "state" / "verified" / HASH_A)
    registry.parent.mkdir(exist_ok=True)
    mapping = make_record(template).to_mapping()
    mapping["template_path"] = "verified/" + HASH_A
    registry.write_text(yaml.safe_dump({"templates": [mapping]}))

    loaded = load_verified_templates(registry)

    assert loaded[0].template_path == template.resolve()


def test_register_is_idempotent_and_does_not_rewrite(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    record = make_record(make_template(tmp_path / "verified" / HASH_A))
    first = register_verified_template(registry, record)
    original = registry.read_bytes()

    second = register_verified_template(registry, record)

    assert second == first
    assert registry.read_bytes() == original
    assert len(load_verified_templates(registry)) == 1


def test_conflicting_duplicate_hash_is_rejected(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    first = make_template(tmp_path / "one", HASH_A)
    second = make_template(tmp_path / "two", HASH_A)
    register_verified_template(registry, make_record(first))

    with pytest.raises(ValueError, match="conflicting duplicate"):
        register_verified_template(registry, make_record(second))

    mappings = [make_record(first).to_mapping(), make_record(second).to_mapping()]
    registry.write_text(yaml.safe_dump({"templates": mappings}))
    with pytest.raises(ValueError, match="conflicting duplicate"):
        load_verified_templates(registry)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"templates": {}},
        {"templates": ["not-a-mapping"]},
        {"templates": [{"contract_hash": HASH_A}]},
        {
            "templates": [
                {
                    **{
                        key: value
                        for key, value in make_record(Path("/unused")).to_mapping().items()
                    },
                    "extra": True,
                }
            ]
        },
    ],
)
def test_malformed_registry_entries_are_rejected(
    tmp_path: Path, payload: object
) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        load_verified_templates(registry)


def test_verified_root_containment(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    verified_root = tmp_path / "verified"
    inside = make_template(verified_root / HASH_A)
    outside = make_template(tmp_path / "outside", HASH_B)

    register_verified_template(
        registry, make_record(inside), verified_root=verified_root
    )
    with pytest.raises(ValueError, match="escapes verified_root"):
        register_verified_template(
            registry,
            make_record(outside, HASH_B),
            verified_root=verified_root,
        )


@pytest.mark.parametrize("failure", ["missing", "untrusted", "hash"])
def test_missing_untrusted_or_mismatched_template_is_rejected(
    tmp_path: Path, failure: str
) -> None:
    contract_hash = HASH_B if failure == "hash" else HASH_A
    template = make_template(
        tmp_path / "template",
        contract_hash=contract_hash,
        trusted=False if failure == "untrusted" else True,
    )
    if failure == "missing":
        (template / "kernel.py").unlink()

    with pytest.raises(ValueError):
        register_verified_template(
            tmp_path / "registry.yaml", make_record(template, HASH_A)
        )


def test_template_symlinks_are_rejected(tmp_path: Path) -> None:
    template = make_template(tmp_path / "template")
    (template / "linked").symlink_to(template / "kernel.py")
    with pytest.raises(ValueError, match="symlink"):
        register_verified_template(
            tmp_path / "registry.yaml", make_record(template)
        )
