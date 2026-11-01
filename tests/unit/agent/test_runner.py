"""AgentRunner: coroutines on a private loop, driven from sync code."""

from __future__ import annotations

import asyncio
import threading

import pytest

from lumenrl.agent.runner import AgentRunner


def test_submitted_coroutines_run_on_the_runner_thread():
    runner = AgentRunner()
    try:
        async def where():
            await asyncio.sleep(0)
            return threading.current_thread().name

        assert runner.submit(where()).result(5) == "lumenrl-agent-loop"
    finally:
        runner.close()


def test_close_runs_cleanup_and_cancels_what_is_still_running():
    runner = AgentRunner()
    cleaned = []
    started = threading.Event()

    async def forever():
        started.set()
        await asyncio.sleep(3600)

    async def cleanup():
        cleaned.append(True)

    future = runner.submit(forever())
    assert started.wait(5)
    runner.close(cleanup)

    assert cleaned == [True]
    assert future.cancelled()


def test_submit_after_close_is_an_error():
    runner = AgentRunner()
    runner.close()

    async def nothing():
        return None

    with pytest.raises(RuntimeError, match="closed"):
        runner.submit(nothing())
