import json
import threading
from types import SimpleNamespace

import pytest
import yaml

import multi_tune_agent.production_controller as production_controller
from multi_tune_agent.production_controller import (
    AttemptSpec,
    CELL_QUOTAS,
    DEFAULT_PROFILE,
    LANE_QUOTAS,
    MODE_QUOTAS,
    ProductionController,
    TOP10_FAMILIES,
    WAVE_2000_PROFILE,
    WaveProfile,
    atomic_json,
    canonical_top10_family,
    load_cases,
    load_wave_profile,
    parse_gpus,
    quotas_pass,
    resolve_seed_manifest,
    sample_counts,
    seed_sample_keys,
    select_samples,
    write_input_manifest,
)


def _sample(number, mode, lane):
    return {
        "sample_id": f"sample-{number:04d}",
        "task_type": mode,
        "provenance": {
            "lane": lane,
            "run_id": f"run-{number:04d}",
            "candidate_id": f"candidate-{number:04d}",
        },
    }


def _top10_profile(family_minimums=None, checkpoint_sample_count=0):
    modes = {
        "cold_start": 4,
        "profile_guided": 4,
        "direction_conditioned": 4,
        "error_recovery": 4,
        "regression_balance": 4,
    }
    lanes = {"triton_gfx942": 10, "hip_gfx942": 10}
    cells = {
        mode: {"triton_gfx942": 2, "hip_gfx942": 2} for mode in modes
    }
    return WaveProfile(
        "top10-test",
        "top10-test-v1",
        modes,
        lanes,
        cells,
        family_minimum_quotas=family_minimums
        or {family: 1 for family in TOP10_FAMILIES},
        checkpoint_sample_count=checkpoint_sample_count,
    )


def test_custom_zero_checkpoint_profile_does_not_inherit_default_seed(tmp_path):
    default_seed = tmp_path / "train-seed.json"
    explicit_seed = tmp_path / "explicit-seed.json"
    profile = _top10_profile(checkpoint_sample_count=0)

    assert resolve_seed_manifest(
        None, default_seed, custom_profile=True, profile=profile
    ) is None
    assert (
        resolve_seed_manifest(
            explicit_seed, default_seed, custom_profile=True, profile=profile
        )
        == explicit_seed
    )
    assert (
        resolve_seed_manifest(
            None, default_seed, custom_profile=False, profile=profile
        )
        == default_seed
    )


@pytest.mark.parametrize(
    ("operator", "family"),
    [
        ("gemm", "gemm"),
        ("batched_gemm", "gemm"),
        ("gated_gemm", "gemm"),
        ("gemm_activation", "gemm"),
        ("scaled_quant_gemm", "gemm"),
        ("rms_norm", "rms_norm"),
        ("fused_add_rms_norm", "rms_norm"),
        ("fused_silu_mul", "fused_moe"),
        ("silu_and_mul", "fused_moe"),
        ("grouped_gemm", "fused_moe"),
        ("fused_moe_fp8_blockscale", "fused_moe"),
        ("mha", "mha"),
        ("mla", "mla"),
        ("paged_attention", "paged_attention"),
        ("rope_kv_cache", "rope_kv_cache"),
        ("blockscale_gemm", "blockscale_gemm"),
        ("all_reduce", "all_reduce"),
        ("sampling", "sampling"),
    ],
)
def test_top10_operator_aliases_are_strict(operator, family):
    assert canonical_top10_family(operator) == family
    assert canonical_top10_family(f"prefix_{operator}") is None


def test_load_cases_prefers_explicit_canonical_family(tmp_path):
    path = tmp_path / "cases.yaml"
    _cases(path)
    payload = yaml.safe_load(path.read_text())
    payload["tasks"][0]["operator"] = "unclassified"
    payload["tasks"][0]["top10_family"] = "mla"
    payload["tasks"][1]["operator"] = "fused_add_rms_norm"
    path.write_text(yaml.safe_dump(payload))

    cases = load_cases(path)

    assert {case.case_id: case.top10_family for case in cases} == {
        "triton-case": "mla",
        "hip-case": "rms_norm",
    }
    payload["tasks"][0]["top10_family"] = "not-a-family"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="ten canonical"):
        load_cases(path)


def test_sample_selection_satisfies_both_exact_quota_dimensions():
    samples = []
    number = 0
    for mode, lanes in CELL_QUOTAS.items():
        for lane, count in lanes.items():
            for _ in range(count + 3):
                samples.append(_sample(number, mode, lane))
                number += 1

    selected = select_samples(samples)
    selected_ids = {
        candidate_id for candidate_ids in selected.values() for candidate_id in candidate_ids
    }
    chosen = [
        item for item in samples if item["provenance"]["candidate_id"] in selected_ids
    ]
    counts = sample_counts(chosen)

    assert quotas_pass(counts)
    assert counts["task_type"] == MODE_QUOTAS
    assert counts["lane"] == LANE_QUOTAS
    assert counts["cell"] == CELL_QUOTAS


def test_2000_profile_is_internally_consistent_and_selectable():
    samples = []
    number = 0
    for mode, lanes in WAVE_2000_PROFILE.cell_quotas.items():
        for lane, count in lanes.items():
            for _ in range(count):
                samples.append(_sample(number, mode, lane))
                number += 1

    selected = select_samples(samples, profile=WAVE_2000_PROFILE)
    selected_ids = {
        candidate_id for candidate_ids in selected.values() for candidate_id in candidate_ids
    }
    chosen = [
        sample
        for sample in samples
        if sample["provenance"]["candidate_id"] in selected_ids
    ]

    assert WAVE_2000_PROFILE.sample_count == 2000
    assert quotas_pass(sample_counts(chosen, WAVE_2000_PROFILE), WAVE_2000_PROFILE)
    assert WAVE_2000_PROFILE.mode_quotas == {
        "cold_start": 307,
        "profile_guided": 293,
        "direction_conditioned": 907,
        "error_recovery": 293,
        "regression_balance": 200,
    }
    assert WAVE_2000_PROFILE.lane_quotas == {
        "triton_gfx942": 1000,
        "hip_gfx942": 1000,
    }


def test_wave_profile_rejects_inconsistent_cells():
    modes = {
        "cold_start": 1,
        "profile_guided": 0,
        "direction_conditioned": 0,
        "error_recovery": 0,
        "regression_balance": 0,
    }
    empty_cells = {
        mode: {"triton_gfx942": 0, "hip_gfx942": 0} for mode in modes
    }
    with pytest.raises(ValueError, match="rows must equal"):
        WaveProfile(
            "broken",
            "broken-v1",
            modes,
            {"triton_gfx942": 1, "hip_gfx942": 0},
            empty_cells,
        )


def test_profile_file_loads_all_top10_family_minimums(tmp_path):
    profile = _top10_profile(checkpoint_sample_count=3)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile.as_dict()))

    loaded = load_wave_profile(path)

    assert loaded == profile
    assert set(loaded.family_minimum_quotas) == set(TOP10_FAMILIES)
    assert loaded.checkpoint_sample_count == 3


def test_dev_200_profile_is_dev_only_and_exact():
    path = (
        production_controller.DEFAULT_CONFIG.parent
        / "phase1-dev-wave-200.yaml"
    )

    profile = load_wave_profile(path)

    assert profile.sample_count == 200
    assert profile.allowed_case_splits == ("dev",)
    assert profile.mode_quotas == {
        "cold_start": 30,
        "profile_guided": 30,
        "direction_conditioned": 90,
        "error_recovery": 30,
        "regression_balance": 20,
    }
    assert profile.lane_quotas == {
        "triton_gfx942": 100,
        "hip_gfx942": 100,
    }
    assert all(
        sum(profile.cell_quotas[mode].values()) == quota
        for mode, quota in profile.mode_quotas.items()
    )


def test_current_330_checkpoint_accepts_inventory_weighted_70_percent_profile():
    minimums = {
        "mha": 112,
        "mla": 112,
        "paged_attention": 112,
        "fused_moe": 111,
        "gemm": 167,
        "rms_norm": 111,
        "rope_kv_cache": 111,
        "blockscale_gemm": 111,
        "all_reduce": 111,
        "sampling": 111,
    }
    profile = WaveProfile(
        WAVE_2000_PROFILE.name,
        WAVE_2000_PROFILE.dataset_version,
        WAVE_2000_PROFILE.mode_quotas,
        WAVE_2000_PROFILE.lane_quotas,
        WAVE_2000_PROFILE.cell_quotas,
        WAVE_2000_PROFILE.allowed_case_splits,
        family_minimum_quotas=minimums,
        checkpoint_sample_count=330,
    )

    assert profile.sample_count - profile.checkpoint_sample_count == 1670
    assert sum(profile.family_minimum_quotas.values()) == 1169


def test_seed_samples_are_pinned_before_deficit_fill(tmp_path):
    mode = "cold_start"
    lane = "triton_gfx942"
    quota = CELL_QUOTAS[mode][lane]
    seed_sample = _sample(9999, mode, lane)
    seed_sample["provenance"]["run_id"] = "seed-run"
    seed_sample["provenance"]["candidate_id"] = "seed-candidate"
    samples = [_sample(number, mode, lane) for number in range(quota)]
    samples.append(seed_sample)
    for other_mode, lanes in CELL_QUOTAS.items():
        for other_lane, count in lanes.items():
            if (other_mode, other_lane) == (mode, lane):
                continue
            start = len(samples) + 10000
            samples.extend(
                _sample(start + offset, other_mode, other_lane)
                for offset in range(count)
            )
    seed_manifest = tmp_path / "seed.json"
    seed_manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "path": "/legacy/seed-run",
                        "candidate_ids": ["seed-candidate"],
                    }
                ]
            }
        )
    )

    pinned = seed_sample_keys(samples, seed_manifest)
    selected = select_samples(samples, pinned_keys=pinned)

    assert selected["seed-run"] == {"seed-candidate"}
    assert sum(len(ids) for ids in selected.values()) == 300


def test_seed_samples_fail_clearly_when_they_exceed_cell_quota():
    quota = CELL_QUOTAS["cold_start"]["triton_gfx942"]
    samples = [
        _sample(number, "cold_start", "triton_gfx942")
        for number in range(quota + 1)
    ]
    pins = {
        (sample["provenance"]["run_id"], sample["provenance"]["candidate_id"])
        for sample in samples
    }

    with pytest.raises(ValueError, match="pinned seed samples exceed wave cell quota"):
        select_samples(samples, pinned_keys=pins)


def test_selection_enforces_exact_cells_and_family_minimums_with_pins():
    profile = _top10_profile(checkpoint_sample_count=1)
    samples = []
    number = 0
    families = iter(TOP10_FAMILIES)
    pin = None
    for mode, lanes in profile.cell_quotas.items():
        for lane, quota in lanes.items():
            for _ in range(quota):
                sample = _sample(number, mode, lane)
                family = next(families, None)
                if family:
                    sample["provenance"]["top10_family"] = family
                samples.append(sample)
                if number == 0:
                    pin = (
                        sample["provenance"]["run_id"],
                        sample["provenance"]["candidate_id"],
                    )
                number += 1
            for _ in range(2):
                sample = _sample(number, mode, lane)
                if number == 22:
                    sample["provenance"]["top10_family"] = "mha"
                samples.append(sample)
                number += 1
    assert pin is not None

    selected = select_samples(samples, profile=profile, pinned_keys={pin})
    selected_ids = {candidate for values in selected.values() for candidate in values}
    chosen = [
        sample
        for sample in samples
        if sample["provenance"]["candidate_id"] in selected_ids
    ]
    counts = sample_counts(chosen, profile, pinned_keys={pin})

    assert pin[1] in selected[pin[0]]
    assert quotas_pass(counts, profile)
    assert counts["family_status"]["met"] is True
    assert counts["checkpoint_top10_family"]["mha"] == 1
    assert all(
        counts["post_checkpoint_top10_family"][family] >= 1
        for family in TOP10_FAMILIES
    )


def test_selection_fails_closed_when_family_minimums_are_infeasible():
    profile = _top10_profile()
    samples = []
    number = 0
    for mode, lanes in profile.cell_quotas.items():
        for lane, quota in lanes.items():
            for _ in range(quota + 2):
                sample = _sample(number, mode, lane)
                sample["provenance"]["top10_family"] = "gemm"
                samples.append(sample)
                number += 1

    with pytest.raises(ValueError, match="family minimum quotas are infeasible"):
        select_samples(samples, profile=profile)


def test_pinned_checkpoint_family_does_not_satisfy_remaining_wave_minimum():
    minimums = {family: 0 for family in TOP10_FAMILIES}
    minimums["mha"] = 1
    profile = _top10_profile(minimums, checkpoint_sample_count=1)
    samples = []
    number = 0
    pin = None
    for mode, lanes in profile.cell_quotas.items():
        for lane, quota in lanes.items():
            for _ in range(quota):
                sample = _sample(number, mode, lane)
                if number == 0:
                    sample["provenance"]["top10_family"] = "mha"
                    pin = (
                        sample["provenance"]["run_id"],
                        sample["provenance"]["candidate_id"],
                    )
                samples.append(sample)
                number += 1
    assert pin is not None

    with pytest.raises(ValueError, match="family minimum quotas are infeasible"):
        select_samples(samples, profile=profile, pinned_keys={pin})


def test_in_progress_selection_remains_partial_and_family_weighted():
    minimums = {family: 0 for family in TOP10_FAMILIES}
    minimums["mla"] = 5
    profile = _top10_profile(minimums)
    samples = []
    for number, family in enumerate(("gemm", "mla", "gemm")):
        sample = _sample(number, "cold_start", "triton_gfx942")
        sample["provenance"]["top10_family"] = family
        samples.append(sample)

    selected = select_samples(samples, profile=profile, require_complete=False)

    assert selected == {
        "run-0000": {"candidate-0000"},
        "run-0001": {"candidate-0001"},
    }


def test_atomic_state_round_trip_and_gpu_ranges(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"attempts": 7})

    assert json.loads(path.read_text()) == {"attempts": 7}
    assert parse_gpus("2-4,7") == (2, 3, 4, 7)
    assert not list(tmp_path.glob("*.tmp-*"))


def test_input_manifest_preserves_seed_runs_blobs_and_overrides(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(
        json.dumps(
            {
                "blob_root": "/seed/blobs",
                "runs": [{"path": "/seed/runs/run-a", "candidate_files": []}],
                "environment_overrides": {"sha256:image": {"software": {}}},
            }
        )
    )
    output = tmp_path / "input.json"

    write_input_manifest(
        output,
        [],
        tmp_path / "dataset",
        tmp_path / "new-blobs",
        seed_manifest=seed,
    )

    value = json.loads(output.read_text())
    assert value["runs"][0]["blob_root"] == "/seed/blobs"
    assert "sha256:image" in value["environment_overrides"]


def test_input_manifest_uses_profile_version_and_coverage_targets(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"runs": [{"path": "/seed/runs/run-a"}]}))
    output = tmp_path / "input.json"

    write_input_manifest(
        output,
        [],
        tmp_path / "dataset",
        tmp_path / "new-blobs",
        seed_manifest=seed,
        profile=WAVE_2000_PROFILE,
    )

    value = json.loads(output.read_text())
    assert value["dataset_version"] == "phase1-production-wave-2000-v1"
    assert value["coverage_targets"] == {
        "sample_count": 2000,
        "task_type": WAVE_2000_PROFILE.mode_quotas,
        "lane": WAVE_2000_PROFILE.lane_quotas,
    }


def _cases(path):
    tasks = []
    for lane in LANE_QUOTAS:
        backend = lane.split("_", 1)[0]
        tasks.append(
            {
                "id": f"{backend}-case",
                "backend": backend,
                "architecture": "gfx942",
                "direction": "optimize",
                "provenance": {
                    "case_seed": {
                        "target_lane": lane,
                        "split_group": "train",
                        "split_version": "v2",
                        "source_lineage_id": f"lineage-{backend}",
                    }
                },
            }
        )
    path.write_text(yaml.safe_dump({"tasks": tasks}))


def _set_all_reduce(path, world_size, *, lane=None):
    payload = yaml.safe_load(path.read_text())
    for task in payload["tasks"]:
        task_lane = task["provenance"]["case_seed"]["target_lane"]
        if lane is not None and task_lane != lane:
            continue
        task["top10_family"] = "all_reduce"
        task["recognized_contract"] = {
            "top10_family": "all_reduce",
            "contract": {"operator": "all_reduce", "world_size": world_size},
        }
    path.write_text(yaml.safe_dump(payload))


def _controller(tmp_path, *, gpus=(1, 2, 3, 4, 5, 6, 7), max_attempts=None):
    cases = tmp_path / "cases.yaml"
    config = tmp_path / "config.yaml"
    _cases(cases)
    config.write_text(yaml.safe_dump({"geak_root": str(tmp_path)}))
    return ProductionController(
        cases_path=cases,
        trajectory_root=tmp_path / "trajectories",
        dataset_root=tmp_path / "dataset",
        base_config=config,
        gpus=gpus,
        max_attempts=max_attempts,
        poll_seconds=0,
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_reduce_case_loads_contract_world_size(tmp_path, world_size):
    cases = tmp_path / "cases.yaml"
    _cases(cases)
    _set_all_reduce(cases, world_size, lane="hip_gfx942")

    loaded = {case.case_id: case for case in load_cases(cases)}

    assert loaded["hip-case"].world_size == world_size
    assert loaded["triton-case"].world_size == 1


@pytest.mark.parametrize(
    ("world_size", "expected"),
    [(2, (1, 2)), (4, (1, 2, 3, 4))],
)
def test_all_reduce_atomically_reserves_distinct_gang_gpus(
    tmp_path, world_size, expected
):
    controller = _controller(tmp_path)
    _set_all_reduce(controller.cases_path, world_size)
    controller.cases = load_cases(controller.cases_path)
    controller._available_gpus_for_next_spec = controller.gpus

    spec = controller._next_spec(1, set())

    assert spec is not None
    assert spec.gpus == expected
    assert spec.world_size == world_size
    attempt = controller.state["attempts"][-1]
    assert attempt["gpu"] == 1
    assert attempt["gpus"] == list(expected)
    assert attempt["world_size"] == world_size


def test_all_reduce_waits_for_full_gang_without_partial_reservation(tmp_path):
    controller = _controller(tmp_path)
    _set_all_reduce(controller.cases_path, 4)
    controller.cases = load_cases(controller.cases_path)
    controller._available_gpus_for_next_spec = (1, 2, 3)

    assert controller._next_spec(1, set()) is None
    assert controller.state["attempts_started"] == 0
    assert controller.state["attempts"] == []


def test_single_card_job_fills_remainder_while_gang_is_running(tmp_path):
    controller = _controller(tmp_path)
    payload = yaml.safe_load(controller.cases_path.read_text())
    hip = next(task for task in payload["tasks"] if task["id"] == "hip-case")
    hip["top10_family"] = "all_reduce"
    hip["recognized_contract"] = {
        "top10_family": "all_reduce",
        "contract": {"operator": "all_reduce", "world_size": 4},
    }
    single = json.loads(json.dumps(hip))
    single["id"] = "hip-single"
    single["top10_family"] = "gemm"
    single["recognized_contract"]["top10_family"] = "gemm"
    single["recognized_contract"]["contract"] = {"operator": "gemm"}
    single["provenance"]["case_seed"]["source_lineage_id"] = "lineage-hip-single"
    payload["tasks"].append(single)
    controller.cases_path.write_text(yaml.safe_dump(payload))
    controller.cases = load_cases(controller.cases_path)

    controller._available_gpus_for_next_spec = controller.gpus
    gang = controller._next_spec(1, set())
    assert gang is not None and gang.gpus == (1, 2, 3, 4)

    controller._available_gpus_for_next_spec = (5, 6, 7)
    controller._gang_running_for_next_spec = True
    single_spec = controller._next_spec(5, set())

    assert single_spec is not None
    assert single_spec.case_id == "hip-single"
    assert single_spec.gpus == (5,)
    assert set(gang.gpus).isdisjoint(single_spec.gpus)


def test_gpu_zero_is_excluded_from_controller_configuration(tmp_path):
    with pytest.raises(ValueError, match="GPU0"):
        parse_gpus("0-2")
    with pytest.raises(ValueError, match="GPU0"):
        _controller(tmp_path, gpus=(0, 1, 2))


def test_gang_does_not_overlap_and_free_gpu_refills_while_it_runs(tmp_path):
    gang_release = threading.Event()
    refill_started = threading.Event()
    active_gpus = set()
    active_lock = threading.Lock()

    class FakeController:
        control_root = tmp_path
        gpus = (1, 2, 3)
        poll_seconds = 0.01
        max_attempts = 3
        profile = DEFAULT_PROFILE
        state = {
            "attempts_started": 0,
            "attempts": [],
            "harvested": {},
        }

        def rebuild(self):
            return {
                "counts": {"task_type": {}, "lane": {}, "cell": {}},
                "quality": {"status": "pass"},
                "leakage": {"status": "pass"},
                "coverage": {"status": "fail"},
            }

        def harvest(self):
            return None

        def _gpu_busy(self, gpu):
            return False

        def _next_spec(self, gpu, reserved):
            number = self.state["attempts_started"] + 1
            if number > self.max_attempts:
                return None
            self.state["attempts_started"] = number
            gpus = (1, 2) if number == 1 else (3,)
            spec = AttemptSpec(
                number,
                f"case-{number}",
                "hip_gfx942",
                "direction_conditioned",
                gpus[0],
                gpus=gpus,
                world_size=len(gpus),
            )
            self.state["attempts"].append({"number": number, "status": "started"})
            return spec

        def _run_attempt(self, spec):
            with active_lock:
                assert active_gpus.isdisjoint(spec.gpus)
                active_gpus.update(spec.gpus)
            if spec.number == 1:
                assert gang_release.wait(timeout=2)
            elif spec.number == 3:
                refill_started.set()
                gang_release.set()
            with active_lock:
                active_gpus.difference_update(spec.gpus)
            return {"status": "completed", "positive_candidates": 0}

        def _save(self):
            return None

    assert ProductionController.run(FakeController()) == 2
    assert refill_started.is_set()


def test_attempt_config_and_environment_use_gang_gpu_list(tmp_path, monkeypatch):
    controller = _controller(tmp_path)
    spec = AttemptSpec(
        1,
        "hip-case",
        "hip_gfx942",
        "direction_conditioned",
        2,
        gpus=(2, 4, 6, 7),
        world_size=4,
    )
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(production_controller.subprocess, "run", fake_run)

    result = controller._run_attempt(spec)
    config = yaml.safe_load(
        (controller.attempt_root / "00001" / "config.yaml").read_text()
    )

    assert result["status"] == "completed"
    assert config["gpu_ids"] == "2,4,6,7"
    assert captured["env"]["HIP_VISIBLE_DEVICES"] == "2,4,6,7"
    assert captured["env"]["GEAK_GPU_ALLOWED"] == "2,4,6,7"


def test_stale_gang_attempt_is_preserved_and_released_on_resume(tmp_path):
    first = _controller(tmp_path, max_attempts=1)
    _set_all_reduce(first.cases_path, 2)
    first.cases = load_cases(first.cases_path)
    first._available_gpus_for_next_spec = first.gpus
    spec = first._next_spec(1, set())
    assert spec is not None and spec.gpus == (1, 2)

    resumed = ProductionController(
        cases_path=first.cases_path,
        trajectory_root=first.trajectory_root,
        dataset_root=first.dataset_root,
        base_config=first.base_config,
        gpus=first.gpus,
        max_attempts=1,
        poll_seconds=0,
    )
    resumed.rebuild = lambda: {
        "counts": {"task_type": {}, "lane": {}, "cell": {}},
        "quality": {"status": "pass"},
        "leakage": {"status": "pass"},
        "coverage": {"status": "fail"},
    }
    resumed.harvest = lambda: None

    assert resumed.run() == 2
    stale = resumed.state["attempts"][0]
    assert stale["status"] == "controller_restarted"
    assert stale["gpus"] == [1, 2]
    assert stale["world_size"] == 2


def test_2000_profile_filters_dev_cases_from_collection(tmp_path):
    cases_path = tmp_path / "cases.yaml"
    _cases(cases_path)
    payload = yaml.safe_load(cases_path.read_text())
    dev_case = dict(payload["tasks"][0])
    dev_case["id"] = "triton-dev-case"
    dev_case["provenance"] = {
        "case_seed": {
            **dev_case["provenance"]["case_seed"],
            "split_group": "dev",
            "source_lineage_id": "lineage-triton-dev",
        }
    }
    payload["tasks"].append(dev_case)
    cases_path.write_text(yaml.safe_dump(payload))

    cases = load_cases(cases_path, WAVE_2000_PROFILE)

    assert {case.case_id for case in cases} == {"triton-case", "hip-case"}
    assert "dev" not in WAVE_2000_PROFILE.allowed_case_splits


def test_held_out_case_is_always_forbidden(tmp_path):
    cases_path = tmp_path / "cases.yaml"
    _cases(cases_path)
    payload = yaml.safe_load(cases_path.read_text())
    payload["tasks"][0]["provenance"]["case_seed"]["split_group"] = "held_out"
    cases_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="held_out cases are forbidden"):
        load_cases(cases_path)


def test_controller_resume_preserves_attempt_ceiling_and_primary_priority(tmp_path):
    cases = tmp_path / "cases.yaml"
    config = tmp_path / "config.yaml"
    trajectory = tmp_path / "trajectories"
    dataset = tmp_path / "dataset"
    _cases(cases)
    config.write_text(
        yaml.safe_dump(
            {
                "geak_root": str(tmp_path),
                "cases_path": str(cases),
                "trajectory_root": str(trajectory),
            }
        )
    )
    first = ProductionController(
        cases_path=cases,
        trajectory_root=trajectory,
        dataset_root=dataset,
        base_config=config,
        gpus=(2,),
        max_attempts=1,
        poll_seconds=0,
    )
    spec = first._next_spec(2, set())
    assert spec is not None
    assert spec.mode in {"cold_start", "profile_guided", "direction_conditioned"}

    resumed = ProductionController(
        cases_path=cases,
        trajectory_root=trajectory,
        dataset_root=dataset,
        base_config=config,
        gpus=(2,),
        max_attempts=1,
        poll_seconds=0,
    )
    assert resumed.state["attempts_started"] == 1
    assert resumed._next_spec(2, set()) is None
    assert resumed.state["counts"]["top10_family"] == {
        family: 0 for family in TOP10_FAMILIES
    }
    assert resumed.state["counts"]["family_status"]["met"] is True


def test_choose_case_prioritizes_family_deficit_without_breaking_attempt_fairness(
    tmp_path,
):
    cases = tmp_path / "cases.yaml"
    config = tmp_path / "config.yaml"
    trajectory = tmp_path / "trajectories"
    dataset = tmp_path / "dataset"
    _cases(cases)
    payload = yaml.safe_load(cases.read_text())
    payload["tasks"][0]["operator"] = "mla"
    extra = json.loads(json.dumps(payload["tasks"][0]))
    extra["id"] = "triton-gemm-case"
    extra["operator"] = "gemm"
    extra["provenance"]["case_seed"]["source_lineage_id"] = "lineage-triton-gemm"
    payload["tasks"].append(extra)
    second_mla = json.loads(json.dumps(payload["tasks"][0]))
    second_mla["id"] = "triton-mla-second"
    second_mla["provenance"]["case_seed"]["source_lineage_id"] = (
        "lineage-triton-mla-second"
    )
    payload["tasks"].append(second_mla)
    cases.write_text(yaml.safe_dump(payload))
    config.write_text(yaml.safe_dump({"geak_root": str(tmp_path)}))
    minimums = {family: 0 for family in TOP10_FAMILIES}
    minimums["mla"] = 5
    minimums["gemm"] = 1
    controller = ProductionController(
        cases_path=cases,
        trajectory_root=trajectory,
        dataset_root=dataset,
        base_config=config,
        gpus=(2,),
        max_attempts=None,
        poll_seconds=0,
        profile=_top10_profile(minimums),
    )

    assert controller._choose_case("triton_gfx942", "cold_start").case_id == "triton-case"
    controller.state["attempted_case_modes"]["triton-case:cold_start"] = 1
    assert (
        controller._choose_case("triton_gfx942", "cold_start").case_id
        == "triton-gemm-case"
    )
    controller.state["attempted_case_modes"]["triton-gemm-case:cold_start"] = 1
    assert (
        controller._choose_case("triton_gfx942", "cold_start").case_id
        == "triton-mla-second"
    )


def test_fallback_targets_lane_that_can_fill_largest_family_deficit(tmp_path):
    cases = tmp_path / "cases.yaml"
    config = tmp_path / "config.yaml"
    _cases(cases)
    payload = yaml.safe_load(cases.read_text())
    payload["tasks"][0]["top10_family"] = "sampling"
    payload["tasks"][1]["top10_family"] = "blockscale_gemm"
    cases.write_text(yaml.safe_dump(payload))
    config.write_text(yaml.safe_dump({"geak_root": str(tmp_path)}))
    minimums = {family: 0 for family in TOP10_FAMILIES}
    minimums["sampling"] = 1
    minimums["blockscale_gemm"] = 3
    profile = _top10_profile(minimums)
    controller = ProductionController(
        cases_path=cases,
        trajectory_root=tmp_path / "trajectories",
        dataset_root=tmp_path / "dataset",
        base_config=config,
        gpus=(1,),
        max_attempts=None,
        poll_seconds=0,
        profile=profile,
    )
    controller.state["counts"].update(
        {
            "accepted": profile.sample_count,
            "lane": dict(profile.lane_quotas),
            "cell": {
                mode: dict(lanes) for mode, lanes in profile.cell_quotas.items()
            },
            "post_checkpoint_top10_family": dict.fromkeys(TOP10_FAMILIES, 0),
        }
    )
    controller._available_gpus_for_next_spec = (1,)

    spec = controller._next_spec(1, set())

    assert spec is not None
    assert spec.lane == "hip_gfx942"
    assert spec.case_id == "hip-case"


def test_controller_rejects_resume_with_different_profile(tmp_path):
    cases = tmp_path / "cases.yaml"
    config = tmp_path / "config.yaml"
    trajectory = tmp_path / "trajectories"
    dataset = tmp_path / "dataset"
    _cases(cases)
    config.write_text(yaml.safe_dump({"geak_root": str(tmp_path)}))
    first = ProductionController(
        cases_path=cases,
        trajectory_root=trajectory,
        dataset_root=dataset,
        base_config=config,
        gpus=(2,),
        max_attempts=1,
        poll_seconds=0,
    )
    first._save()

    with pytest.raises(ValueError, match="wave/profile mismatch"):
        ProductionController(
            cases_path=cases,
            trajectory_root=trajectory,
            dataset_root=dataset,
            base_config=config,
            gpus=(2,),
            max_attempts=1,
            poll_seconds=0,
            profile=WAVE_2000_PROFILE,
        )


def test_controller_upgrades_matching_legacy_profile_at_frozen_checkpoint(tmp_path):
    cases = tmp_path / "cases.yaml"
    config = tmp_path / "config.yaml"
    trajectory = tmp_path / "trajectories"
    dataset = tmp_path / "dataset"
    _cases(cases)
    config.write_text(yaml.safe_dump({"geak_root": str(tmp_path)}))
    legacy_profile = _top10_profile(
        {family: 0 for family in TOP10_FAMILIES},
    )
    legacy_profile = WaveProfile(
        legacy_profile.name,
        legacy_profile.dataset_version,
        legacy_profile.mode_quotas,
        legacy_profile.lane_quotas,
        legacy_profile.cell_quotas,
    )
    first = ProductionController(
        cases_path=cases,
        trajectory_root=trajectory,
        dataset_root=dataset,
        base_config=config,
        gpus=(2,),
        max_attempts=None,
        poll_seconds=0,
        profile=legacy_profile,
    )
    first.state["counts"]["accepted"] = 1
    first._save()
    minimums = {family: 0 for family in TOP10_FAMILIES}
    minimums["mha"] = 1
    upgraded_profile = WaveProfile(
        legacy_profile.name,
        legacy_profile.dataset_version,
        legacy_profile.mode_quotas,
        legacy_profile.lane_quotas,
        legacy_profile.cell_quotas,
        family_minimum_quotas=minimums,
        checkpoint_sample_count=1,
    )

    resumed = ProductionController(
        cases_path=cases,
        trajectory_root=trajectory,
        dataset_root=dataset,
        base_config=config,
        gpus=(2,),
        max_attempts=None,
        poll_seconds=0,
        profile=upgraded_profile,
    )

    assert resumed.state["wave_identity"]["profile"] == upgraded_profile.as_dict()
    assert "family_profile_upgraded_at" in resumed.state


def test_controller_interleaves_queued_regression_and_limits_retries(tmp_path):
    cases = tmp_path / "cases.yaml"
    config = tmp_path / "config.yaml"
    trajectory = tmp_path / "trajectories"
    dataset = tmp_path / "dataset"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _cases(cases)
    config.write_text(yaml.safe_dump({"geak_root": str(tmp_path)}))
    controller = ProductionController(
        cases_path=cases,
        trajectory_root=trajectory,
        dataset_root=dataset,
        base_config=config,
        gpus=(2,),
        max_attempts=None,
        poll_seconds=0,
    )
    harvest_id = "run:candidate:regression"
    item = {
        "harvest_id": harvest_id,
        "mode": "regression_balance",
        "case_id": "hip-case",
        "resume_workspace": str(workspace),
        "regression_constraints": {"maximum_ms": {"shape": 1.0}},
    }
    controller.state["queues"]["regression_balance"] = [item]
    controller.state["harvested"][harvest_id] = dict(item, consumed=False)
    controller.state["counts"]["task_type"] = {
        "cold_start": 36,
        "profile_guided": 33,
        "direction_conditioned": 124,
        "error_recovery": 16,
        "regression_balance": 0,
    }
    controller.state["counts"]["cell"] = {
        "cold_start": {"triton_gfx942": 20, "hip_gfx942": 20},
        "profile_guided": {"triton_gfx942": 20, "hip_gfx942": 20},
        "direction_conditioned": {"triton_gfx942": 64, "hip_gfx942": 64},
        "error_recovery": {"triton_gfx942": 10, "hip_gfx942": 10},
        "regression_balance": {"triton_gfx942": 0, "hip_gfx942": 0},
    }

    spec = controller._next_spec(2, set())
    assert spec is not None
    assert spec.mode == "regression_balance"
    assert spec.harvest_id == harvest_id

    controller.state["attempts"] = [
        {"number": number, "harvest_id": harvest_id} for number in range(1, 4)
    ]
    controller.state["attempts_started"] = 3
    fallback = controller._next_spec(2, set())
    assert fallback is not None
    assert fallback.mode in {"cold_start", "profile_guided", "direction_conditioned"}


def test_controller_refills_finished_gpu_without_waiting_for_slow_peer(tmp_path):
    slow_release = threading.Event()
    third_started = threading.Event()

    class FakeController:
        control_root = tmp_path
        gpus = (1, 2)
        poll_seconds = 0.01
        max_attempts = 3
        profile = DEFAULT_PROFILE
        state = {
            "attempts_started": 0,
            "attempts": [],
            "harvested": {},
        }

        def rebuild(self):
            return {
                "counts": {"task_type": {}, "lane": {}, "cell": {}},
                "quality": {"status": "pass"},
                "leakage": {"status": "pass"},
                "coverage": {"status": "fail"},
            }

        def harvest(self):
            return None

        def _gpu_busy(self, gpu):
            return False

        def _next_spec(self, gpu, reserved):
            if self.state["attempts_started"] >= self.max_attempts:
                return None
            number = self.state["attempts_started"] + 1
            self.state["attempts_started"] = number
            spec = AttemptSpec(
                number,
                f"case-{number}",
                "hip_gfx942",
                "direction_conditioned",
                gpu,
            )
            self.state["attempts"].append({"number": number, "status": "started"})
            return spec

        def _run_attempt(self, spec):
            if spec.number == 1:
                assert slow_release.wait(timeout=2)
            elif spec.number == 3:
                third_started.set()
                slow_release.set()
            return {"status": "completed", "positive_candidates": 0}

        def _save(self):
            return None

    controller = FakeController()

    assert ProductionController.run(controller) == 2
    assert third_started.is_set()
