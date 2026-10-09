"""GEAK sandbox backend — GPU Efficiency Assessment Kit.

Evaluates GPU kernels via the GEAK task_runner pipeline:
  compile -> correctness -> performance

Requires task directories with scripts/task_runner.py harness.
"""

from sandbox.registry import register_backend

from .backend import GEAKBackend

register_backend("geak", GEAKBackend)

__all__ = ["GEAKBackend"]
