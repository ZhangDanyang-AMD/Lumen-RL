from pathlib import Path

import pytest

from geak_utils import KernelSandbox, TaskSpec, load_tasks
from multi_tune_agent.config import MultiTuneConfig
from multi_tune_agent.geak_tool import GEAKToolEnvironment
from multi_tune_agent.models import Direction


def test_config_resolves_relative_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("GEAK_HOME", raising=False)
    monkeypatch.delenv("LUMEN_CODE_GEAK_ROOT", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "geak_root: geak",
                "cases_path: cases.yaml",
                "trajectory_root: runs",
                "max_rounds: 2",
            ]
        )
        + "\n"
    )
    config = MultiTuneConfig.from_yaml(config_path)
    assert config.geak_root == (tmp_path / "geak").resolve()
    assert config.cases_path == (tmp_path / "cases.yaml").resolve()
    assert config.trajectory_root == (tmp_path / "runs").resolve()
    assert config.max_rounds == 2


def test_config_allows_env_base_url_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "geak_root: geak\n"
        "cases_path: cases.yaml\n"
        "trajectory_root: runs\n"
        "base_url: http://yaml-host:8000/v1\n"
    )
    monkeypatch.setenv("LUMEN_CODE_BASE_URL", "http://remote-host:18000/v1")
    config = MultiTuneConfig.from_yaml(config_path)
    assert config.base_url == "http://remote-host:18000/v1"


def test_config_geak_environment_override_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("LUMEN_CODE_GEAK_ROOT", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "geak_root: yaml-geak\n"
        "cases_path: cases.yaml\n"
        "trajectory_root: runs\n"
    )
    monkeypatch.setenv("GEAK_HOME", "generic-geak")
    config = MultiTuneConfig.from_yaml(config_path)
    assert config.geak_root == (tmp_path / "generic-geak").resolve()

    monkeypatch.setenv("LUMEN_CODE_GEAK_ROOT", "specific-geak")
    config = MultiTuneConfig.from_yaml(config_path)
    assert config.geak_root == (tmp_path / "specific-geak").resolve()


def test_bootstrap_config_defaults_and_aiter_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "geak_root: geak\ncases_path: cases.yaml\ntrajectory_root: runs\n"
    )
    monkeypatch.setenv("AITER_HOME", str(tmp_path / "aiter-checkout"))
    config = MultiTuneConfig.from_yaml(config_path)
    assert config.bootstrap_enabled is True
    assert config.bootstrap_auto_promote is True
    assert config.aiter_root == (tmp_path / "aiter-checkout").resolve()
    assert config.generated_template_root.name == "generated"
    assert config.bootstrap_min_aiter_score == 85
    with pytest.raises(ValueError, match="between 0 and 100"):
        MultiTuneConfig(
            tmp_path,
            tmp_path / "cases.yaml",
            tmp_path / "runs",
            bootstrap_min_aiter_score=101,
        )
    with pytest.raises(ValueError, match="bootstrap_auto_promote must be a boolean"):
        MultiTuneConfig(
            tmp_path,
            tmp_path / "cases.yaml",
            tmp_path / "runs",
            bootstrap_auto_promote="yes",  # type: ignore[arg-type]
        )


def test_config_rejects_unknown_and_invalid_values(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "geak_root: x\ncases_path: y\ntrajectory_root: z\nunknown: true\n"
    )
    with pytest.raises(ValueError, match="unknown"):
        MultiTuneConfig.from_yaml(path)
    with pytest.raises(ValueError, match="positive"):
        MultiTuneConfig(Path("."), Path("."), Path("."), max_rounds=0)


def test_reward_has_correctness_gate_and_improvement_shaping():
    failed = GEAKToolEnvironment.reward(
        {"compiled": True, "correct": False, "speedup_geomean": 5.0}, 1.0
    )
    assert failed.total == -1.0
    assert failed.correctness == -1.0

    improved = GEAKToolEnvironment.reward(
        {"compiled": True, "correct": True, "speedup_geomean": 1.25}, 1.0
    )
    unchanged = GEAKToolEnvironment.reward(
        {"compiled": True, "correct": True, "speedup_geomean": 1.25}, 1.25
    )
    assert improved.total > unchanged.total > 1.0
    assert improved.improvement == pytest.approx(0.25)


def test_multitune_uses_geak_utils_and_restricts_case_types(tmp_path):
    assert TaskSpec and load_tasks and KernelSandbox
    task = tmp_path / "task"
    task.mkdir()
    (task / "config.yaml").write_text(
        "correctness_command: python check.py\n"
        "performance_command: python bench.py\n"
    )
    catalog = tmp_path / "tasks.yaml"
    catalog.write_text(
        "tasks:\n"
        "- id: unsupported\n"
        "  type: arbitrary-op\n"
        "  kernel_path: task\n"
    )
    config = MultiTuneConfig(tmp_path, catalog, tmp_path / "runs")
    with pytest.raises(ValueError, match="unsupported MultiTune case type"):
        GEAKToolEnvironment(config)


def test_direction_accepts_native_geak_schema():
    direction = Direction.from_mapping(
        {
            "id": "r1_d0",
            "title": "retile the kernel",
            "specialty": "algorithm",
            "prompt": "Change only kernel.py and measure the result.",
        },
        0,
    )
    assert direction.direction_id == "r1_d0"
    assert direction.strategy == "retile the kernel"
    assert direction.instructions == "Change only kernel.py and measure the result."

