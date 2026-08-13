"""LumenRL's ATOM model runner: trims two sources of waste in hidden-state capture.

On-policy training needs two sweeps over each batch:

1. **generate** — decode a response from the prompt. Only the tokens matter;
   hidden states from the prompt prefill would be dead weight.
2. **extract** — prefill the full prompt+response sequence and ship the
   auxiliary hidden states to Mooncake.

Two adjustments to the stock ``RLHFModelRunner``:

**Capture is parked for requests nobody will read.** ``configure_hidden_states()``
latches ``_extract_mode`` on for good, so the generate sweep would also run the
hook-capture forward path and concatenate five layers of activations that are
then dropped -- roughly 4.7 GB of pointless traffic per batch of B=64. ATOM keys
every Mooncake write on ``external_request_ids``, so a sweep that submits
requests *without* data ids already writes nothing; this runner takes the extra
step of skipping the capture forward path entirely for such batches, which is
what makes one engine able to serve both sweeps with no mode switch and no
restart.

**A tensor-parallel write guard.** Every TP rank runs ``_store_hidden_states``,
and under tensor parallelism the residual stream is replicated -- so all 8 ranks
write byte-identical payloads to the same Mooncake key. Only rank 0 writes,
matching how the vLLM connector behaved. Because the store is created lazily on
first use, this also means exactly one Mooncake client exists per engine rather
than eight competing ones.

The guard deliberately covers the *store* only, never the forward path: the
capture decision is derived from batch metadata, which is identical on every
rank, so all ranks take the same branch and TP collectives stay in lockstep.

The runner is selected by passing ``runner_qualname`` to ``AsyncLLMEngine``,
which only ever ``setdefault``s it — so no ATOM source needs patching.
"""

from __future__ import annotations

import logging

from atom.model_engine.model_runner import ModelRunner
from atom.rollout.model_runner_ext import RLHFModelRunner

logger = logging.getLogger("atom")


class LumenRLModelRunner(RLHFModelRunner):
    """ATOM RLHF runner that captures hidden states only when someone reads them."""

    @staticmethod
    def _batch_wants_capture(batch) -> bool:
        """True when any request in *batch* carries a Mooncake key.

        ATOM writes hidden states under ``external_request_id``, so a request
        without one has no destination. The generate sweep submits its prompts
        that way on purpose, which is what parks capture.

        Derived purely from batch metadata, so every TP rank agrees.
        """
        if batch is None:
            return True

        external_ids = getattr(batch, "external_request_ids", None)
        if not external_ids:
            return False
        return any(ext_id is not None for ext_id in external_ids)

    def run_model(self, input_ids, batch=None):
        """Take the stock forward path when nothing will consume the capture."""
        if self._extract_mode and not self._batch_wants_capture(batch):
            return ModelRunner.run_model(self, input_ids, batch)
        return super().run_model(input_ids, batch)

    def _store_hidden_states(self, batch):
        """Write captured hidden states to Mooncake, from TP rank 0 only."""
        if getattr(self, "rank", 0) != 0:
            return
        return super()._store_hidden_states(batch)


__all__ = ["LumenRLModelRunner"]
