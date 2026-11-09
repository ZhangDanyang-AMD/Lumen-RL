"""LumenRL-side MICake plugin: exposes MICake as an AgentEnvironment backend.

MICake keeps ownership of candidate validation, source materialization,
compilation, GPU correctness and measurement; this package only drives its
worker and maps results to observations, rewards and receipts. Importing it
registers the ``micake_gemm`` environment backend.
"""

from sandbox.micake.environment import BACKEND_NAME, MicakeGemmEnvironment, RewardWeights
from sandbox.micake.registry import register

register()

__all__ = ["BACKEND_NAME", "MicakeGemmEnvironment", "RewardWeights", "register"]
