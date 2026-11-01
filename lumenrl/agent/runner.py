"""An asyncio loop on its own thread, for running agent rollouts from sync code."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class AgentRunner:
    """Runs coroutines on a private event loop in a daemon thread.

    The trainer is synchronous; agent rollouts are coroutines that await the
    engine and the tools. ``submit`` hands one to the loop and returns a
    ``concurrent.futures.Future`` the caller can wait on or poll.
    """

    def __init__(self, name: str = "lumenrl-agent-loop") -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=name, daemon=True)
        self._thread.start()
        self._closed = False

    def submit(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        if self._closed:
            coro.close()
            raise RuntimeError("AgentRunner is closed")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self, cleanup: Callable[[], Awaitable[None]] | None = None, timeout: float = 30.0) -> None:
        """Run ``cleanup`` on the loop, cancel what is still running, stop the thread."""
        if self._closed:
            return
        self._closed = True

        async def shutdown() -> None:
            if cleanup is not None:
                await cleanup()
            current = asyncio.current_task()
            pending = [task for task in asyncio.all_tasks() if task is not current]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(timeout)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout)
            if not self._thread.is_alive():
                self._loop.close()
