import hashlib
from collections import Counter

import yaml

from multi_tune_agent.phase1_splits import assign_groups
from multi_tune_agent.top10_inventory import (
    AITER_GIT_SHA,
    AITER_ROOT,
    SOURCES,
    TOP10_FAMILIES,
    allocate_weighted_quotas,
    build_inventory,
    merge_with_catalog,
)


def test_inventory_has_all_top10_families_and_locked_sources():
    inventory = build_inventory()

    assert tuple(inventory["families"]) == TOP10_FAMILIES
    assert {item["top10_family"] for item in inventory["candidates"]} == set(
        TOP10_FAMILIES
    )
    assert inventory["source_revisions"]["aiter_locked_gfx942"]["git_sha"] == AITER_GIT_SHA
    assert all(
        hashlib.sha256((AITER_ROOT / source["path"]).read_bytes()).hexdigest()
        == source["sha"]
        for source in SOURCES
    )


def test_inventory_lane_rules_and_representative_lineages():
    candidates = build_inventory()["candidates"]
    by_family = Counter(item["top10_family"] for item in candidates)

    assert all(by_family[family] >= 2 for family in TOP10_FAMILIES)
    all_reduce = [item for item in candidates if item["top10_family"] == "all_reduce"]
    assert {item["contract"]["world_size"] for item in all_reduce} == {2, 4}
    assert all(item["target_lanes"] == ["hip_gfx942"] for item in all_reduce)
    assert all(
        set(item["target_lanes"]) == {"hip_gfx942", "triton_gfx942"}
        for item in candidates
        if item["top10_family"] != "all_reduce"
    )
    assert all(
        len(
            {
                item["source_lineage_id"]
                for item in candidates
                if item["top10_family"] == family
            }
        )
        >= 2
        for family in TOP10_FAMILIES
    )
    assert all(
        item["contract"]["output_dtype"] == "int32"
        for item in candidates
        if item["top10_family"] == "sampling"
    )


def test_every_family_has_a_naturally_assigned_train_contract():
    inventory = build_inventory()
    groups = assign_groups(
        inventory["candidates"],
        salt="geak-phase1-gfx942-v4",
        dev_fraction=0.15,
        held_out_fraction=0.25,
    )

    train_families = {
        family
        for group in groups
        if group["split"] == "train"
        for family in group["top10_families"]
    }
    assert train_families == set(TOP10_FAMILIES)


def test_inventory_appends_to_existing_catalog_without_replacement(tmp_path):
    base = tmp_path / "base.yaml"
    existing = {
        "source_revisions": {"existing": {"git_sha": "abc"}},
        "candidates": [{"id": "existing-case", "priority": "P0"}],
    }
    base.write_text(yaml.safe_dump(existing), encoding="utf-8")

    merged = merge_with_catalog(build_inventory(), base)

    assert merged["candidates"][0]["id"] == "existing-case"
    assert len(merged["candidates"]) == 1 + len(build_inventory()["candidates"])
    assert merged["source_revisions"]["existing"] == {"git_sha": "abc"}
    assert merged["top10_inventory"]["candidate_count"] == len(
        build_inventory()["candidates"]
    )


def test_source_lineage_weights_allocate_exact_checkpoint_quota():
    inventory = build_inventory()

    quotas = allocate_weighted_quotas(1169, inventory["family_weights"])

    assert sum(quotas.values()) == 1169
    assert quotas["gemm"] == 167
    assert quotas["all_reduce"] == 111
    assert set(quotas) == set(TOP10_FAMILIES)
