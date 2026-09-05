import asyncio
import json
from types import SimpleNamespace

import pytest

from multi_tune_agent.config import MultiTuneConfig
from multi_tune_agent.flow import MultiTuneFlow
from multi_tune_agent.models import RewardBreakdown
from multi_tune_agent.runtime import ModelTurn


class FakeBackend:
    def generate(self, messages, tools=()):
        if tools:
            text = "completed private optimization"
        else:
            prompt = messages[-1]["content"]
            if "PHASE=plan_round" in prompt:
                text = json.dumps(
                    {
                        "directions": [
                            {
                                "direction_id": "tile",
                                "specialty": "algorithm",
                                "strategy": "retile",
                                "instructions": "retile",
                            },
                            {
                                "direction_id": "launch",
                                "specialty": "host_runtime",
                                "strategy": "reduce launch overhead",
                                "instructions": "reduce launch overhead",
                            },
                        ]
                    }
                )
            else:
                text = '{"status": "ok"}'
        return ModelTurn(
            text,
            {"role": "assistant", "content": text},
            [],
            [-0.1],
            {"completion_tokens": 1},
        )


class FakeEnvironment:
    def __init__(self, config, trajectory=None):
        self.config = config
        self.trajectory = trajectory
        self.cases = {"demo": SimpleNamespace(case_type="gemm")}
        self.states = {}
        self.created = 0

    def _case(self, case_id):
        return self.cases[case_id]

    def case_observation(self, case_id):
        return {"case_id": case_id, "case_type": "gemm"}

    def create(
        self,
        case_id,
        *,
        role,
        parent_session_id=None,
        establish_baseline=False,
    ):
        self.created += 1
        session_id = "%s-%d" % (role, self.created)
        if role == "director":
            speedup = 1.0
        elif role == "integrator":
            speedup = 1.3
        else:
            speedup = 1.1 + 0.05 * self.created
        self.states[session_id] = SimpleNamespace(
            last_reward=0.0, speedup=speedup, role=role
        )
        return session_id, {
            "session_id": session_id,
            "baseline": {"per_case_ms": {"x": 1.0}},
            "allowed_write_paths": ["kernel.py"],
        }

    def tool_schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "geak",
                    "description": "fake",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def execute(self, session_id, parameters):
        return {"ok": True}, RewardBreakdown(0, 0, 0, 0, 1), {}

    def get(self, session_id):
        return self.states[session_id]

    def release(self, session_id):
        return None

    def verify(self, session_id):
        state = self.states[session_id]
        evaluation = {
            "compiled": True,
            "correct": True,
            "speedup_geomean": state.speedup,
            "candidate_ms": {"x": 1.0 / state.speedup},
        }
        reward = RewardBreakdown(
            1.0 + state.speedup, 1.0, state.speedup - 1.0, 0.0, state.speedup
        )
        state.last_reward = reward.total
        return {"ok": True, "evaluation": evaluation}, reward, {}

    def independent_verify(self, session_id):
        result, reward, metrics = self.verify(session_id)
        return {
            **result,
            "verify_source": "multitune_independent",
            "verify_session_id": "verify-" + session_id,
        }, reward, metrics


class FakePrompts:
    def __init__(self, geak_root):
        self.geak_root = geak_root

    def system(self, role, case_type=""):
        return "You are the %s." % role


def test_multi_role_flow_runs_engineers_verifier_and_integrator(tmp_path, monkeypatch):
    import multi_tune_agent.flow as flow_module

    monkeypatch.setattr(flow_module, "GEAKToolEnvironment", FakeEnvironment)
    monkeypatch.setattr(flow_module, "RolePromptLibrary", FakePrompts)
    config = MultiTuneConfig(
        geak_root=tmp_path,
        cases_path=tmp_path / "cases.yaml",
        trajectory_root=tmp_path / "runs",
        max_rounds=1,
        engineers_per_round=2,
        engineer_tool_rounds=2,
        integrator_tool_rounds=2,
    )
    summary = asyncio.run(
        MultiTuneFlow(config, FakeBackend()).run_case(
            "demo", user_request="Optimize the requested MI308X GEMM shape."
        )
    )
    assert summary["status"] == "success"
    assert summary["user_request"] == "Optimize the requested MI308X GEMM shape."
    assert summary["final_session_id"].startswith("integrator-")
    assert summary["final_evaluation"]["speedup_geomean"] == 1.3
    performance = summary["final_kernel_performance"]
    assert performance["measurement_valid"] is True
    assert performance["baseline_geomean_ms"] == 1.0
    assert performance["final_kernel_geomean_ms"] == pytest.approx(1.0 / 1.3)
    assert performance["speedup_geomean"] == 1.3
    assert len(summary["rounds"][0]["candidates"]) == 2
    assert summary["rounds"][0]["committed"] is True
    assert (tmp_path / "runs" / "runs").is_dir()


def test_final_report_does_not_publish_incorrect_kernel_speed():
    performance = MultiTuneFlow._final_kernel_performance(
        {"per_case_ms": {"x": 1.0}},
        {
            "compiled": True,
            "correct": False,
            "speedup_geomean": 9.0,
            "candidate_ms": {"x": 0.1},
        },
    )
    assert performance["measurement_valid"] is False
    assert performance["final_kernel_geomean_ms"] is None
    assert performance["speedup_geomean"] is None
    assert performance["final_kernel_per_case_ms"] == {}

