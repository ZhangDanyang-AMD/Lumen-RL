"""Agent rollouts on the trainer side: replicas, masks, metrics, pipelines."""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from lumenrl.agent.chat import ChatTemplateTokenizer
from lumenrl.agent.deepseek_v4 import DeepSeekV4ChatTokenizer, DsmlToolParser
from lumenrl.agent.loop import ToolAgentLoop
from lumenrl.agent.trajectory import Trajectory
from lumenrl.core.config import LumenRLConfig
from lumenrl.engine.inference.atom_ray_server import ATOMRayServer
from lumenrl.trainer import rl_trainer
from lumenrl.trainer.agent_rollout import (
    AgentGroupRollout,
    agent_metrics,
    build_agent_rollout,
    mask_environment_tokens,
    num_turns,
    verl_agent_loop_metrics,
)
from lumenrl.trainer.rl_trainer import RLTrainer
from tests.unit.agent.fakes import IM_END, ChatMLTokenizer

PROMPT = [{"role": "user", "content": "q"}]


class AnsweringServer:
    """A replica whose every turn is a final answer; records what it was sent."""

    def __init__(self, tok: ChatMLTokenizer, delay_s: float = 0.0) -> None:
        self.tok = tok
        self.delay_s = delay_s
        self.prompts: list[list[int]] = []
        self.generate = SimpleNamespace(remote=self._generate)

    async def _generate(self, prompt_ids, sampling_params):
        self.prompts.append(list(prompt_ids))
        await asyncio.sleep(self.delay_s)
        ids = self.tok.encode("Answer: 1") + [IM_END]
        return {"token_ids": ids, "logprobs": [-0.1] * len(ids), "finish_reason": "stop"}


def make_rollout(servers, tok, **kwargs) -> AgentGroupRollout:
    agent = ToolAgentLoop(ChatTemplateTokenizer(tok), [])
    return AgentGroupRollout(agent, servers, max_response_tokens=64, **kwargs)


def test_a_group_runs_on_one_replica_the_least_loaded():
    tok = ChatMLTokenizer()
    slow, fast = AnsweringServer(tok, delay_s=0.2), AnsweringServer(tok)
    rollout = make_rollout([slow, fast], tok)
    try:
        first = rollout.submit(PROMPT, 3, {"max_tokens": 64})
        second = rollout.submit(PROMPT, 2, {"max_tokens": 64})
        assert len(first.result(5)) == 3
        assert len(second.result(5)) == 2
        assert (len(slow.prompts), len(fast.prompts)) == (3, 2)

        rollout.submit(PROMPT, 1, {"max_tokens": 64}).result(5)
        assert len(slow.prompts) == 4, "both idle again: the first replica wins"
    finally:
        rollout.close()


def test_run_returns_groups_in_prompt_order():
    tok = ChatMLTokenizer()
    rollout = make_rollout([AnsweringServer(tok)], tok)
    try:
        groups = rollout.run(
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "bb"}]],
            2, {"max_tokens": 64},
        )
    finally:
        rollout.close()

    assert [len(group) for group in groups] == [2, 2]
    first, second = (tok.decode(group[0].prompt_ids) for group in groups)
    assert "user\na<|im_end|>" in first
    assert "user\nbb<|im_end|>" in second


def test_the_system_prompt_is_prepended_once():
    tok = ChatMLTokenizer()
    server = AnsweringServer(tok)
    rollout = make_rollout([server], tok, system_prompt="Use tools.")
    try:
        rollout.submit(PROMPT, 1, {"max_tokens": 64}).result(5)
        rollout.submit(
            [{"role": "system", "content": "Own."}, *PROMPT], 1, {"max_tokens": 64},
        ).result(5)
    finally:
        rollout.close()

    with_default, with_own = (tok.decode(p) for p in server.prompts)
    assert with_default.startswith("<|im_start|>system\nUse tools.<|im_end|>")
    assert with_own.startswith("<|im_start|>system\nOwn.<|im_end|>")
    assert "Use tools." not in with_own


def test_config_builds_the_sandbox_tool():
    config = LumenRLConfig()
    mt = config.policy.generation.multi_turn
    mt.enabled = True
    mt.code_interpreter.url = "http://172.30.0.10:8080/run_code"
    mt.max_assistant_turns = 3

    rollout = build_agent_rollout(config, ChatMLTokenizer(), [object()])
    try:
        [schema] = rollout._agent.tool_schemas
        assert schema["function"]["name"] == "code_interpreter"
        [tool] = rollout._agent._tools.values()
        assert tool._url == "http://172.30.0.10:8080/run_code"
        assert rollout._agent._max_assistant_turns == 3
        assert rollout._max_response_tokens == config.policy.max_response_length
    finally:
        rollout.close()


def test_config_rejects_an_unknown_tool_call_format():
    config = LumenRLConfig()
    config.policy.generation.multi_turn.format = "json"

    with pytest.raises(ValueError, match="'hermes' .* 'dsml'"):
        build_agent_rollout(config, ChatMLTokenizer(), [object()])


def test_config_picks_deepseek_v4_encoding_for_dsml_calls():
    config = LumenRLConfig()
    mt = config.policy.generation.multi_turn
    mt.enabled = True
    mt.format = "dsml"
    mt.chat_template_kwargs = {"thinking_mode": "thinking"}

    rollout = build_agent_rollout(config, ChatMLTokenizer(), [object()])
    try:
        assert isinstance(rollout._agent._chat, DeepSeekV4ChatTokenizer)
        assert rollout._agent._chat._assistant_start.endswith("<think>")
        assert isinstance(rollout._agent._parser, DsmlToolParser)
    finally:
        rollout.close()


# -- masks --------------------------------------------------------------------


def trajectory(prompt, *segments) -> Trajectory:
    """``segments`` alternate model tokens and environment tokens."""
    t = Trajectory(prompt_ids=list(prompt))
    for i, tokens in enumerate(segments):
        if i % 2 == 0:
            t.add_model_turn(list(tokens), [-float(tok) for tok in tokens])
        else:
            t.add_environment(list(tokens))
    return t


def assembling_trainer() -> RLTrainer:
    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(generation=SimpleNamespace(
            vllm_cfg=SimpleNamespace(calculate_log_probs=True),
        )),
        moe=SimpleNamespace(r3=SimpleNamespace(enabled=False, rollout_replay=False)),
    )
    trainer._tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._device = torch.device("cpu")
    return trainer


MODEL_TOKENS = {10, 11, 12, 13, 14, 15}


def test_only_model_tokens_stay_in_the_response_mask():
    trainer = assembling_trainer()
    rows = [
        trajectory([1, 2, 3], [10, 11], [20, 21, 22], [12]),
        trajectory([4, 5], [13, 14, 15]),
    ]
    sequences, seq_mask, prompt_lengths, rollout_lp, ragged = trainer._assemble_ray_rollout(
        [row.to_rollout_result() for row in rows],
    )
    response_mask = trainer._build_response_mask(sequences, seq_mask, prompt_lengths)

    masked = mask_environment_tokens(
        response_mask, seq_mask, prompt_lengths, ragged["agent_response_mask"],
    )

    for i in range(len(rows)):
        kept = masked[i].nonzero(as_tuple=True)[0]
        scored_tokens = sequences[i, kept + 1].tolist()
        assert set(scored_tokens) <= MODEL_TOKENS
        assert scored_tokens == [t for t in rows[i].response_ids if t in MODEL_TOKENS]
        assert rollout_lp[i, kept].tolist() == [-float(t) for t in scored_tokens]
    assert masked.sum().item() == 6
    assert response_mask.sum().item() == 9, "the unmasked response keeps tool output"
    assert ragged["agent_stats"][0]["environment_tokens"] == 3


def test_mask_must_cover_every_row_and_fit_the_response():
    trainer = assembling_trainer()
    rows = [trajectory([1, 2], [10, 11])]
    sequences, seq_mask, prompt_lengths, _, _ = trainer._assemble_ray_rollout(
        [row.to_rollout_result() for row in rows],
    )
    response_mask = trainer._build_response_mask(sequences, seq_mask, prompt_lengths)

    with pytest.raises(ValueError, match="rows"):
        mask_environment_tokens(response_mask, seq_mask, prompt_lengths, [])
    with pytest.raises(ValueError, match="past the response end"):
        mask_environment_tokens(response_mask, seq_mask, prompt_lengths, [[1, 1, 1]])


def test_plain_rollouts_carry_no_agent_fields():
    trainer = assembling_trainer()
    *_, ragged = trainer._assemble_ray_rollout(
        [{"prompt_token_ids": [1], "token_ids": [2, 3], "logprobs": [-1.0, -1.0]}],
    )

    assert ragged == {}


def test_agent_metrics_summarise_the_trained_rows():
    stats = [
        trajectory([1], [10], [20], [11]).stats() | {"tool_calls": 1, "stop_reason": "answer"},
        trajectory([1], [10]).stats() | {"tool_calls": 0, "stop_reason": "length"},
    ]
    stats[0]["tool_errors"] = 1

    metrics = agent_metrics(stats)

    assert metrics["agent/assistant_turns_mean"] == 1.5
    assert metrics["agent/tool_call_frac"] == 0.5
    assert metrics["agent/tool_error_rate"] == 1.0
    assert metrics["agent/stop_answer_frac"] == 0.5
    assert metrics["agent/stop_length_frac"] == 0.5
    assert metrics["agent/environment_tokens_mean"] == 0.5
    assert agent_metrics([]) == {}


def test_verl_metrics_count_turns_and_time_per_rollout():
    one_round = trajectory([1], [10], [20], [11])
    one_round.generate_s, one_round.tool_s = 2.0, 3.0
    answer_only = trajectory([1], [10])
    answer_only.generate_s = 4.0

    metrics = verl_agent_loop_metrics(
        [one_round.stats(), answer_only.stats()], prompt_lengths=[7, 9], response_lengths=[3, 1],
    )

    assert num_turns(one_round.stats()) == 4, "prompt + 2 assistant turns + 1 tool round"
    assert (metrics["num_turns/max"], metrics["num_turns/min"]) == (4.0, 2.0)
    assert metrics["timing_s/agent_loop/tool_calls/mean"] == 1.5
    assert metrics["timing_s/agent_loop/generate_sequences/max"] == 4.0
    assert metrics["timing_s/agent_loop/slowest/tool_calls"] == 3.0
    assert metrics["timing_s/agent_loop/slowest/prompt_length"] == 7.0
    assert verl_agent_loop_metrics([], [], []) == {}


# -- pipelines ----------------------------------------------------------------


def test_fully_async_backend_scores_and_filters_agent_groups():
    trainer = assembling_trainer()
    trainer._ray_rollout_mgr = SimpleNamespace(servers=[object()])
    trainer._ray_sampling_params = lambda want_lp: {"max_tokens": 64}
    requested = []

    def get_prompts_range(start, count, *, messages=False):
        requested.append((start, messages))
        return [[{"role": "user", "content": f"q{start}"}]], [f"gt{start}"]

    trainer._get_prompts_range = get_prompts_range
    accs = {0: [1.0, 0.0], 1: [1.0, 1.0]}

    class Rollout:
        def submit(self, messages, n, sp):
            index = int(messages[0]["content"][1:])
            future = concurrent.futures.Future()
            future.set_result([trajectory([1, 2], [10 + index], [20], [12]) for _ in range(n)])
            future.index = index
            return future

    trainer._get_agent_rollout = lambda: Rollout()
    trainer._compute_rewards_full = lambda seqs, mask, plen, gts, verbose: (
        torch.zeros(2), ["a", "b"], accs[int(gts[0][2:])],
    )
    backend = rl_trainer._AgentRolloutGroups(trainer, num_generations=2, use_filter=True)

    handles = [backend.submit(0), backend.submit(1)]
    assert backend.wait(handles, timeout=1) == handles
    rows = backend.finish(handles[0])
    assert backend.finish(handles[1]) is None, "an all-correct group carries no signal"

    assert requested == [(0, True), (1, True)]
    assert rows[5] == ["gt0", "gt0"]
    assert len(rows[7]["agent_response_mask"]) == 2
    assert rows[7]["agent_response_mask"][0] == [1, 0, 1]


def test_fully_async_picks_the_agent_backend(monkeypatch):
    built = []
    monkeypatch.setattr(
        rl_trainer, "_AgentRolloutGroups",
        lambda trainer, n, use_filter: built.append(n) or SimpleNamespace(),
    )

    class Streaming:
        def __init__(self, backend, merge, **kwargs):
            self.backend = backend

        def start(self):
            pass

    monkeypatch.setattr("lumenrl.trainer.fully_async_rollout.StreamingRollout", Streaming)
    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        controller=SimpleNamespace(ray=SimpleNamespace(fully_async=SimpleNamespace(
            max_inflight_groups=4, trigger_parameter_sync_step=1,
            staleness_threshold=0.0, partial_rollout=False,
        ))),
        policy=SimpleNamespace(
            train_global_batch_size=8, gen_batch_size=0,
            generation=SimpleNamespace(multi_turn=SimpleNamespace(enabled=True)),
        ),
        algorithm=SimpleNamespace(name="grpo", dapo=SimpleNamespace()),
    )
    trainer._ray_rollout_mgr = SimpleNamespace(num_replicas=1)
    trainer._prompt_cursor = 0

    trainer._start_streaming_rollout(num_generations=4, rollout_version=0)

    assert built == [4]


def _multi_turn_trainer(*, vllm=True, atom=False, r3=False) -> RLTrainer:
    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        controller=SimpleNamespace(ray=SimpleNamespace(
            one_step_off_policy=False, fully_async=SimpleNamespace(enabled=False),
        )),
        policy=SimpleNamespace(generation=SimpleNamespace(multi_turn=SimpleNamespace(enabled=True))),
        moe=SimpleNamespace(r3=SimpleNamespace(enabled=r3, rollout_replay=False)),
    )
    trainer._ray_use_vllm = vllm
    trainer._ray_use_atom = atom
    return trainer


@pytest.mark.parametrize("kwargs", [{}, {"vllm": False, "atom": True}], ids=["vllm", "atom"])
def test_multi_turn_runs_on_a_ray_rollout(kwargs):
    assert _multi_turn_trainer(**kwargs)._resolve_rollout_pipeline(True) == "sync"


@pytest.mark.parametrize(
    ("use_ray_rollout", "kwargs", "message"),
    [
        (False, {}, "Ray controller with the vLLM or ATOM"),
        (True, {"vllm": False}, "Ray controller with the vLLM or ATOM"),
        (True, {"r3": True}, "moe.r3"),
    ],
)
def test_multi_turn_rejects_what_it_cannot_drive(use_ray_rollout, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _multi_turn_trainer(**kwargs)._resolve_rollout_pipeline(use_ray_rollout)


def test_prompts_come_as_chat_messages_for_the_agent_loop():
    trainer = object.__new__(RLTrainer)
    trainer._tokenizer = None
    trainer._dataset = [
        {"prompt": [{"role": "user", "content": "listed"}], "reward_model": {"ground_truth": "1"}},
        {"prompt": "plain", "reward_model": {"ground_truth": "2"}},
    ]

    prompts, gts = trainer._get_prompts_range(0, 2, messages=True)

    assert prompts == [
        [{"role": "user", "content": "listed"}],
        [{"role": "user", "content": "plain"}],
    ]
    assert gts == ["1", "2"]


def test_sync_collection_sends_chat_messages_to_the_agent_loop():
    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(
            train_global_batch_size=4, gen_batch_size=0,
            generation=SimpleNamespace(multi_turn=SimpleNamespace(enabled=True)),
        ),
        algorithm=SimpleNamespace(name="grpo"),
    )
    trainer._dataset = None
    trainer._prompt_cursor = 0
    seen = []

    def rollout_with_agent_loop(prompts, g):
        seen.append((prompts, g))
        return torch.zeros(4, 3), torch.ones(4, 3), [1] * 4, None, {"agent_stats": [{}] * 4}

    trainer._rollout_with_agent_loop = rollout_with_agent_loop
    trainer._compute_rewards_full = lambda seqs, mask, plen, gts: (torch.zeros(4), ["r"] * 4, [0.0] * 4)

    *_, ragged = trainer._collect_rollout_batch(step=0, num_generations=2)

    [(prompts, g)] = seen
    assert g == 2
    assert prompts == [
        [{"role": "user", "content": "What is 0 + 1?"}],
        [{"role": "user", "content": "What is 1 + 2?"}],
    ]
    assert len(ragged["agent_stats"]) == 4


# -- ATOM sampling noise --------------------------------------------------------


def _atom_trainer(*, multi_turn: bool = True) -> RLTrainer:
    trainer = object.__new__(RLTrainer)
    trainer.config = LumenRLConfig()
    trainer.config.policy.generation.multi_turn.enabled = multi_turn
    return trainer


def test_atom_agent_rollouts_draw_sampling_noise_of_their_own():
    kwargs: dict = {}

    _atom_trainer()._atom_agent_sampling(kwargs)

    assert kwargs == {"sampler_independent_noise": True}


def test_an_explicit_sampler_noise_setting_is_kept():
    kwargs = {"sampler_independent_noise": False}

    _atom_trainer()._atom_agent_sampling(kwargs)

    assert kwargs == {"sampler_independent_noise": False}


def test_single_turn_atom_rollouts_keep_the_default_noise():
    kwargs: dict = {}

    _atom_trainer(multi_turn=False)._atom_agent_sampling(kwargs)

    assert kwargs == {}


def test_the_atom_replicas_of_an_agent_run_get_independent_noise(monkeypatch):
    seen: dict = {}

    class Built(Exception):
        pass

    def manager(colocation_wg, model_name, engine_kwargs, **kwargs):
        seen.update(engine_kwargs)
        raise Built

    monkeypatch.setattr("lumenrl.engine.inference.atom_ray_server.ATOMReplicaManager", manager)
    trainer = _atom_trainer()
    trainer._actor_wg = object()
    trainer._rollout_wg = None
    generation = trainer.config.policy.generation

    with pytest.raises(Built):
        trainer._setup_ray_atom_rollout("model", generation.vllm_cfg, generation.atom_cfg)

    assert seen["sampler_independent_noise"] is True


def _launch_atom(monkeypatch, engine_kwargs: dict, *, independent_noise: bool) -> list[dict]:
    fields = [("model", str, dataclasses.field(default=""))]
    if independent_noise:
        fields.append(("sampler_independent_noise", bool, dataclasses.field(default=False)))
    built: list[dict] = []

    class AsyncLLMEngine:
        def __init__(self, **kwargs):
            built.append(kwargs)

    names = ("atom", "atom.config", "atom.rollout", "atom.rollout.async_engine")
    modules = {name: types.ModuleType(name) for name in names}
    modules["atom.config"].Config = dataclasses.make_dataclass("Config", fields)
    modules["atom.rollout.async_engine"].AsyncLLMEngine = AsyncLLMEngine
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(ATOMRayServer, "_get_node_ip", staticmethod(lambda: "127.0.0.1"))
    monkeypatch.setattr(ATOMRayServer, "_get_free_port", staticmethod(lambda: 1234))

    asyncio.run(ATOMRayServer("model", engine_kwargs, 0).launch())
    return built


def test_an_atom_that_cannot_sample_n1_requests_independently_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="sampler_independent_noise"):
        _launch_atom(monkeypatch, {"sampler_independent_noise": True}, independent_noise=False)


def test_the_independent_noise_setting_reaches_an_atom_that_has_it(monkeypatch):
    built = _launch_atom(monkeypatch, {"sampler_independent_noise": True}, independent_noise=True)

    assert built[0]["sampler_independent_noise"] is True


def test_an_atom_without_the_setting_still_serves_single_turn_runs(monkeypatch):
    built = _launch_atom(monkeypatch, {}, independent_noise=False)

    assert "sampler_independent_noise" not in built[0]
