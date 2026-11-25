"""Opt-in ATOM main adapters owned and launched by Lumen-RL.

This package must not edit the ATOM checkout. All opt-ins are explicit and
verified before the first rollout. The shared BF16 GEMM operation remains in
Lumen; new GPU KV packing kernels belong in AITER, not this package.
"""
