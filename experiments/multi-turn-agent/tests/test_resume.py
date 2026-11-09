import json

from multi_tune_agent.resume import (
    find_latest_checkpoint,
    mark_checkpoint_continued,
)


def test_failed_run_recovers_workspace_baseline_and_last_failure(tmp_path):
    workspace = tmp_path / "sessions" / "engineer-1" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "kernel.py").write_text("candidate = True\n")
    run_dir = tmp_path / "runs" / "case_20260904_120000"
    run_dir.mkdir(parents=True)
    records = [
        {
            "event": "run_start",
            "payload": {
                "case": {
                    "case_id": "case",
                    "case_type": "gemm",
                    "user_request": "optimize",
                },
                "initial": {
                    "baseline": {
                        "geomean_ms": 1.0,
                        "per_case_ms": {"shape": 1.0},
                    }
                },
            },
        },
        {
            "event": "environment_create",
            "payload": {
                "role": "engineer",
                "session_id": "engineer-1",
                "workspace": str(workspace),
            },
        },
        {
            "event": "tool_result",
            "payload": {
                "action": "evaluate",
                "parameters": {"action": "evaluate", "mode": "compile"},
                "result": {"ok": False, "stderr": "compile failed"},
            },
        },
    ]
    (run_dir / "trajectory.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    checkpoint = find_latest_checkpoint(tmp_path)
    assert checkpoint is not None
    assert checkpoint.case_id == "case"
    assert checkpoint.workspace == workspace.resolve()
    assert checkpoint.baseline["per_case_ms"] == {"shape": 1.0}
    assert checkpoint.resume_context["last_failure"]["action"] == "evaluate"

    mark_checkpoint_continued(checkpoint)
    assert find_latest_checkpoint(tmp_path) is None
