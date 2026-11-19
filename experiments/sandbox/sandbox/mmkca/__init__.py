"""LumenRL-side MMKCA plugin: exposes MMKCA's gfx950 GEMM schedules as an AgentEnvironment backend.

MMKCA keeps ownership of schedule verification, MFMA lowering, compilation,
GPU correctness and timing; this package only drives its worker. Importing it
registers the ``mmkca_gemm`` environment backend.
"""

from sandbox.mmkca.environment import BACKEND_NAME, MmkcaGemmEnvironment
from sandbox.mmkca.registry import register

register()

__all__ = ["BACKEND_NAME", "MmkcaGemmEnvironment", "register"]
