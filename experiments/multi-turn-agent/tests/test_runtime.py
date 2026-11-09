import asyncio
import json

import requests

from multi_tune_agent.runtime import (
    ModelTurn,
    OpenAIModelBackend,
    StatefulTool,
    ToolAgentLoop,
    ToolCall,
    _messages_for_generation,
)


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=()):
        self.calls += 1
        if self.calls == 1:
            call = ToolCall("call-1", "fake", {"value": 3})
            return ModelTurn(
                "",
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": '{"value": 3}',
                            },
                        }
                    ],
                },
                [call],
                [-0.1],
                {"completion_tokens": 1},
            )
        return ModelTurn(
            "done",
            {"role": "assistant", "content": "done"},
            [],
            [-0.2],
            {"completion_tokens": 1},
        )


class FakeTool(StatefulTool):
    name = "fake"

    def __init__(self):
        self.created = []
        self.released = []
        self.value = 0

    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "fake",
                    "description": "fake",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def create(self, create_kwargs):
        self.created.append(dict(create_kwargs))
        return "session-1", {"ready": True}

    def execute(self, instance_id, parameters):
        self.value += parameters["value"]
        return {"value": self.value}, 0.75, {"instance_id": instance_id}

    def calc_reward(self, instance_id):
        return float(self.value)

    def release(self, instance_id):
        self.released.append(instance_id)


def test_tool_agent_loop_keeps_one_session_and_tracks_rewards():
    tool = FakeTool()
    output = asyncio.run(
        ToolAgentLoop(
            FakeBackend(), tool, max_assistant_turns=4, retain_session=False
        ).run(
            [{"role": "user", "content": "work"}],
            create_kwargs={"case_id": "case"},
        )
    )
    assert output.final_text == "done"
    assert output.session_id == "session-1"
    assert output.policy_logprobs == [-0.1, -0.2]
    assert output.tool_rewards == [0.75]
    assert output.reward_score == 3.0
    assert output.metrics.assistant_turns == 2
    assert output.metrics.tool_calls == 1
    assert tool.created == [{"case_id": "case"}]
    assert tool.released == ["session-1"]


def test_tool_agent_loop_executes_tool_call_on_last_assistant_turn():
    tool = FakeTool()
    output = asyncio.run(
        ToolAgentLoop(
            FakeBackend(), tool, max_assistant_turns=1, retain_session=False
        ).run(
            [{"role": "user", "content": "work"}],
            create_kwargs={"case_id": "case"},
        )
    )
    assert output.metrics.assistant_turns == 1
    assert output.metrics.tool_calls == 1
    assert output.tool_rewards == [0.75]
    assert output.reward_score == 3.0


def test_generation_history_compacts_large_write_payloads():
    source = "x" * 90000
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "work"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "write-1",
                    "type": "function",
                    "function": {
                        "name": "geak",
                        "arguments": json.dumps(
                            {
                                "action": "write_file",
                                "path": "kernel.py",
                                "content": source,
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "write-1", "content": '{"ok": true}'},
    ]
    prepared = _messages_for_generation(messages)
    arguments = json.loads(prepared[2]["tool_calls"][0]["function"]["arguments"])
    assert "omitted 90000 written characters" in arguments["content"]


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("%s Client Error" % self.status_code, response=self)

    def json(self):
        return self._body


class RecordingSession:
    def __init__(self):
        self.payloads = []

    def post(self, _url, *, json, headers, timeout):
        self.payloads.append(json)
        if len(self.payloads) == 1:
            return FakeResponse(
                400,
                {
                    "error": {
                        "message": (
                            "This model's maximum context length is 32000 tokens "
                            "and your request has 30381 input tokens"
                        )
                    }
                },
            )
        return FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "recovered"},
                        "logprobs": {"content": []},
                    }
                ],
                "usage": {},
            },
        )


def test_backend_retries_context_error_with_available_completion_budget():
    session = RecordingSession()
    turn = OpenAIModelBackend(
        "http://localhost/v1", "model", session=session
    ).generate([{"role": "user", "content": "work"}])
    assert turn.text == "recovered"
    assert [payload["max_tokens"] for payload in session.payloads] == [4096, 1555]

