"""One-step off-policy rollout: generate step s+1 while step s trains.

With the rollout on its own GPUs (``controller.ray.rollout.num_workers > 0``)
the trainer does not touch the rollout engine while it computes log-probs and
updates the actor, so the next batch can be generated in that window. The
batch trained at step s then comes from the weights synced after step s-1, one
update behind the trainer; the rollout-correction weights account for that gap.
Weights are synced only after the in-flight batch has finished, so the engine
never loads new weights in the middle of a request.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any


class OneStepOffPolicyRollout:
    """Keeps at most one rollout batch in flight on a background thread.

    ``collect(step)`` builds one step's batch; it must not use anything the
    training side mutates while it runs.
    """

    def __init__(self, collect: Callable[[int], Any], total_steps: int) -> None:
        self._collect = collect
        self._total_steps = total_steps
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lumenrl-rollout",
        )
        self._pending: Future | None = None
        self._pending_version = 0

    def submit(self, step: int, rollout_version: int) -> None:
        """Start ``step``'s batch on the weights of policy version ``rollout_version``."""
        if self._pending is not None:
            raise RuntimeError("a rollout batch is already in flight")
        self._pending_version = rollout_version
        self._pending = self._executor.submit(self._timed_collect, step)

    def _timed_collect(self, step: int) -> tuple[Any, float]:
        started = time.perf_counter()
        batch = self._collect(step)
        return batch, time.perf_counter() - started

    def take(self, step: int, rollout_version: int) -> tuple[Any, float, int, float]:
        """Return ``step``'s batch and start generating the next one.

        ``rollout_version`` is the policy version now loaded in the rollout; the
        next batch is generated on it. Returns ``(batch, gen_s, staleness,
        wait_s)``: how long the batch took to generate, how many updates the
        trainer (at version ``step``) has applied since the weights that
        generated it, and how long this call blocked.
        """
        if self._pending is None:
            raise RuntimeError(f"no rollout batch in flight for step {step}")
        started = time.perf_counter()
        pending, self._pending = self._pending, None
        batch, gen_s = pending.result()
        wait_s = time.perf_counter() - started
        staleness = step - self._pending_version
        if step + 1 < self._total_steps:
            self.submit(step + 1, rollout_version)
        return batch, gen_s, staleness, wait_s

    def drain(self) -> float:
        """Block until the in-flight batch finishes and return the seconds waited.

        A failure is not raised here; the next ``take`` raises it.
        """
        if self._pending is None:
            return 0.0
        started = time.perf_counter()
        wait([self._pending])
        return time.perf_counter() - started

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._pending = None
