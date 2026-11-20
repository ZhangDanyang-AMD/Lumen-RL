import asyncio
import contextlib
import json

import pytest

from lumenrl.agent.swe.environment import SweRolloutEnvironment
from lumenrl.agent.swe.swebench import Grade
from lumenrl.agent.trajectory import Trajectory


class FakeBash:
    submission = "diff --git a/x b/x\n"


class FakeSweEnv:
    def __init__(self, *, resolved=True, grade_fails=False, grade_error=None):
        self.resolved, self.grade_fails, self.grade_error = resolved, grade_fails, grade_error
        self.opened, self.graded = [], []

    @contextlib.asynccontextmanager
    async def agent_session(self, task):
        self.opened.append(task.instance_id)
        yield FakeBash()

    async def grade(self, task, patch):
        self.graded.append((task.instance_id, patch))
        if self.grade_error is not None:
            raise self.grade_error
        if self.grade_fails:
            raise RuntimeError("grading sandbox died")
        return Grade(self.resolved, True, "resolved" if self.resolved else "tests failed")


class FakeAgent:
    def __init__(self):
        self.sessions = None

    async def run(self, generate, messages, sampling_params, max_response_tokens, max_sequence_tokens, sessions=None):
        self.sessions = sessions
        return Trajectory(prompt_ids=[1, 2])


GT = json.dumps({"instance_id": "repo__repo-1", "image": "img"})


def _rollout(env, **kwargs):
    agent = FakeAgent()
    t = asyncio.run(SweRolloutEnvironment(env, **kwargs).rollout(agent, None, [], {}, 100, None, GT))
    return t, agent


def test_a_resolved_task_earns_the_resolved_reward():
    env = FakeSweEnv(resolved=True)
    t, agent = _rollout(env, resolved_reward=1.0, unresolved_reward=-1.0)

    assert env.opened == ["repo__repo-1"] and env.graded == [("repo__repo-1", FakeBash.submission)]
    assert isinstance(agent.sessions["bash"], FakeBash)
    assert t.env_reward == 1.0 and t.to_rollout_result()["env_reward"] == 1.0


def test_an_unresolved_task_earns_the_unresolved_reward():
    t, _ = _rollout(FakeSweEnv(resolved=False), resolved_reward=1.0, unresolved_reward=-1.0)

    assert t.env_reward == -1.0


def test_a_grading_failure_scores_unresolved_and_keeps_the_rollout():
    t, _ = _rollout(FakeSweEnv(grade_fails=True))

    assert t.env_reward == 0.0 and t.prompt_ids == [1, 2]


def test_a_missing_grading_harness_fails_the_rollout():
    # every grade would fail the same way and every rollout would score unresolved
    env = FakeSweEnv(grade_error=ModuleNotFoundError("No module named 'swebench'"))

    with pytest.raises(ModuleNotFoundError, match="swebench"):
        _rollout(env)
