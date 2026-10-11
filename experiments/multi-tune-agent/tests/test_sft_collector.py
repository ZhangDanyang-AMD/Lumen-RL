import json
import threading
from types import SimpleNamespace

import pytest

from multi_tune_agent.config import MultiTuneConfig
from multi_tune_agent.geak_tool import GEAKToolEnvironment
from multi_tune_agent.models import RewardBreakdown
from multi_tune_agent.sft_collector import SFTCollector
from multi_tune_agent.trajectory import TrajectoryWriter


class FakeEnvironment:
    def __init__(self, parent, child):
        self.states = {
            "parent": SimpleNamespace(
                workspace=parent,
                sandbox=SimpleNamespace(allowed_write_paths=["kernel.py"]),
            ),
            "child": SimpleNamespace(
                workspace=child,
                sandbox=SimpleNamespace(allowed_write_paths=["kernel.py"]),
            ),
        }

    def get(self, session_id):
        return self.states[session_id]


def make_config(tmp_path, **overrides):
    values = {
        "geak_root": tmp_path,
        "cases_path": tmp_path / "cases.yaml",
        "trajectory_root": tmp_path / "trajectories",
        "sft_enabled": True,
        "sft_dataset_root": tmp_path / "dataset",
    }
    values.update(overrides)
    return MultiTuneConfig(**values)


def test_sft_collector_records_plan_sources_patch_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        SFTCollector,
        "_environment",
        lambda self, config: {
            "gpu_architecture": {"ok": True, "stdout": "gfx942\n"},
            "gpu_inventory": {"ok": True, "stdout": "Card SKU: M3000108\n"},
            "software": {
                "rocm_version": "7.0",
                "compiler_version": "clang-20",
            },
            "lumen_git": {"dirty": False},
            "geak_git": {"dirty": False},
        },
    )
    config = make_config(tmp_path)
    run_dir = config.trajectory_root / "runs" / "demo"
    collector = SFTCollector(run_dir, config)
    trajectory = TrajectoryWriter(run_dir, sft_sink=collector.append)
    trajectory.append("role_response", {"api_key": "secret", "value": 3})
    collector.record_plan(
        1,
        {"directions": [{"id": "tile"}]},
        [{"direction_id": "tile"}],
        user_request="optimize",
    )
    collector.record_frozen_input(
        1,
        {
            "parent_source": {"kernel.py": "value = 1\n"},
            "source_lineage_id": "lineage-a",
            "split_group": "train",
        },
    )

    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    (parent / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (child / "kernel.py").write_text("value = 2\n", encoding="utf-8")
    record = collector.record_candidate(
        FakeEnvironment(parent, child),
        "parent",
        "child",
        {"candidate_id": "tile-r1", "accepted": True},
        {
            "ok": True,
            "verify_source": "multitune_independent",
            "verify_session_id": "verify-child",
            "evaluation": {
                "compiled": True,
                "correct": True,
                "candidate_ms": {"case": 0.8},
                "speedup_geomean": 1.25,
            },
        },
        round_index=1,
    )
    manifest = collector.finalize({"status": "success"})

    assert record["sft_positive_eligible"] is True
    assert record["patch_applies"] is True
    patch_path = (
        config.sft_dataset_root
        / "blobs"
        / "sha256"
        / record["patch_hash"][:2]
        / record["patch_hash"]
    )
    assert "-value = 1" in patch_path.read_text(encoding="utf-8")
    assert "+value = 2" in patch_path.read_text(encoding="utf-8")
    assert manifest["collector_complete"] is True
    assert manifest["positive_candidate_count"] == 1
    assert (run_dir / "round_1" / "plan.json").is_file()
    assert (run_dir / "round_1" / "candidates.jsonl").is_file()

    events = [
        json.loads(line)
        for line in (run_dir / "sft_events.jsonl").read_text().splitlines()
    ]
    role_event = next(item for item in events if item["event"] == "role_response")
    assert role_event["payload"]["api_key"] == "<redacted>"


@pytest.mark.parametrize(
    "task_type",
    ["cold_start", "profile_guided", "error_recovery", "regression_balance"],
)
def test_config_accepts_implemented_sft_modes(tmp_path, task_type):
    assert make_config(tmp_path, sft_task_type=task_type).sft_task_type == task_type


def test_independent_verify_uses_fresh_workspace_and_frozen_baseline(
    tmp_path, monkeypatch
):
    environment = object.__new__(GEAKToolEnvironment)
    environment._lock = threading.RLock()
    environment._baseline_cache = {
        "demo": {"per_case_ms": {"case": 1.0}, "geomean_ms": 1.0}
    }
    candidate = SimpleNamespace(
        case_id="demo", workspace=tmp_path / "candidate"
    )
    monkeypatch.setattr(environment, "get", lambda session_id: candidate)
    captured = {}

    def create(case_id, **kwargs):
        captured.update({"case_id": case_id, **kwargs})
        return "verify-session", {"workspace": str(tmp_path / "verify")}

    monkeypatch.setattr(environment, "create", create)
    reward = RewardBreakdown(1.0, 1.0, 0.0, 0.0, 1.0)
    monkeypatch.setattr(
        environment,
        "verify",
        lambda session_id: (
            {"ok": True, "evaluation": {"compiled": True, "correct": True}},
            reward,
            {},
        ),
    )

    result, _, _ = environment.independent_verify("candidate-session")
    assert captured["role"] == "verify_engineer"
    assert captured["source_path"] == candidate.workspace
    assert captured["baseline_override"]["per_case_ms"] == {"case": 1.0}
    assert result["verify_session_id"] == "verify-session"
    assert result["verify_source"] == "multitune_independent"
