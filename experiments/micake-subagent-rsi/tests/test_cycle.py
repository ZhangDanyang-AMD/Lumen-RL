from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "sandbox"))
sys.path.insert(0, str(EXPERIMENT_DIR))

from micake_rsi.cycle import challenger_config, prepare_updates, run_cycle  # noqa: E402
from micake_rsi.sft_data import positive_assistant_indices  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, tokenize=True):
        del tools, add_generation_prompt, tokenize
        ids = []
        for message in messages:
            ids.extend([ord(message["role"][0]), len(message.get("content") or "") + 1])
        return ids


def _sample(advantage: float, index: int = 2) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "call", "tool_calls": []},
        ],
        "tools": [],
        "steps": [{"assistant_message_index": index, "advantage": advantage, "execution_result": "measured"}],
    }


def _cfg(tmp_path: Path) -> dict:
    cfg = yaml.safe_load((EXPERIMENT_DIR / "configs" / "gen0_planner_critic.yaml").read_text(encoding="utf-8"))
    cfg["rsi"]["root"] = str(tmp_path)
    cfg["rsi"]["generation_id"] = "gen0"
    role_dir = tmp_path / "gen0" / "rl"
    role_dir.mkdir(parents=True)
    (role_dir / "planner.jsonl").write_text(json.dumps(_sample(1.5)) + "\n" + json.dumps(_sample(-0.2)) + "\n", encoding="utf-8")
    (role_dir / "critic.jsonl").write_text(json.dumps(_sample(0.4)) + "\n", encoding="utf-8")
    return cfg


def test_only_positive_advantage_turns_are_selected() -> None:
    assert positive_assistant_indices(_sample(0.1), 0.0) == [2]
    assert positive_assistant_indices(_sample(0.0), 0.0) == []
    assert positive_assistant_indices(_sample(-1.0), 0.0) == []


def test_prepare_writes_pinned_manifest_and_skips_non_positive(tmp_path: Path) -> None:
    jobs = {job["role"]: job for job in prepare_updates(_cfg(tmp_path), FakeTokenizer())}
    assert jobs["planner"]["rows"] == 1 and jobs["planner"]["skipped"] is False
    assert jobs["critic"]["rows"] == 1
    manifest = json.loads(Path(jobs["planner"]["data_dir"], "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "geak_training_data_manifest_v2"
    assert manifest["artifacts"]["train"]["rows"] == 1
    row = json.loads(Path(jobs["planner"]["data_dir"], "train.tokenized.jsonl").read_text(encoding="utf-8"))
    assert sum(row["loss_mask"]) > 0 and row["loss_mask"][0] == 0
    config = yaml.safe_load(Path(jobs["planner"]["config_path"]).read_text(encoding="utf-8"))
    assert config["training"]["max_steps"] == 1
    assert len(config["execution"]["data_manifest_sha256"]) == 64


def test_cycle_trains_merges_serves_then_rolls(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    trained: list[str] = []
    merged: list[tuple[str, str]] = []
    served: list[str] = []

    def train_role(path: Path) -> None:
        trained.append(path.parent.name)
        adapter = path.parent / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    def merge_role(base: Path, adapter: Path, output: Path) -> None:
        del base
        assert (adapter / "adapter_config.json").exists()
        output.mkdir(parents=True)
        merged.append((adapter.parent.name, str(output)))

    def serve(update: dict) -> None:
        served.append(f"{update['role']}:{update['port']}")

    def rollout(challenger: dict) -> dict:
        assert challenger["rsi"]["generation_id"] == "gen1"
        assert challenger["rsi"]["parent_generation_id"] == "gen0"
        assert challenger["specialists"]["planner"]["binding"]["adapter_version"] == "1"
        assert challenger["specialists"]["planner"]["endpoint"]["base_url"].endswith(":8100/v1")
        assert challenger["specialists"]["critic"]["endpoint"]["base_url"].endswith(":8101/v1")
        assert served == ["planner:8100", "critic:8101"]
        return {"promotion": {"decision": "HOLD"}}

    manifest = run_cycle(cfg, FakeTokenizer(), train_role, merge_role, serve, rollout)
    assert trained == ["planner", "critic"]
    assert [name for name, _ in merged] == ["planner", "critic"]
    assert manifest["promotion"]["decision"] == "HOLD"


def test_cycle_resumes_from_merged_checkpoints(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    for role in ("planner", "critic"):
        merged = tmp_path / "gen1" / "checkpoints" / role
        merged.mkdir(parents=True)
        (merged / "merge_manifest.json").write_text("{}", encoding="utf-8")
    served: list[dict] = []

    def fail(*_args: object) -> None:
        raise AssertionError("merged roles must not be retrained")

    manifest = run_cycle(cfg, FakeTokenizer(), fail, fail, served.append, lambda _c: {"promotion": {"decision": "HOLD"}})
    assert manifest["promotion"]["decision"] == "HOLD"
    assert [(u["role"], u["tp"]) for u in served] == [("planner", 2), ("critic", 1)]


def test_challenger_keeps_endpoint_when_role_has_no_update() -> None:
    cfg = yaml.safe_load((EXPERIMENT_DIR / "configs" / "gen0_planner_critic.yaml").read_text(encoding="utf-8"))
    served = {"planner": {"checkpoint": "/kept", "port": 8100}, "critic": {"checkpoint": "/new", "port": 8101}}
    nxt = challenger_config(cfg, "gen0", "gen1", served)
    assert nxt["specialists"]["planner"]["endpoint"]["model"] == "/kept"
    assert nxt["specialists"]["critic"]["binding"]["policy_version"] == "critic-gen1"
