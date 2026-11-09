"""LumenRL native AgenticRL: environments, multi-turn rollout, credit assignment.

This package must stay backend-neutral; it never imports MICake, GEAK,
VerlTool or rLLM. Concrete environments register through
:func:`lumenrl.agentic.core.register_environment`.
"""
