"""Plugin contract for embodied RL.

LumenRL core owns the RL machinery (GRPO loss, FSDP2 engine, rollout loop,
world-model environment, trainer). A project supplies the models through a
module that exposes ``PLUGIN``, an :class:`EmbodiedPlugin`. The module is
named in the config as ``embodied.plugin``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class EmbodiedPlugin:
    """Model factories supplied by a project.

    build_policy(model_path, dtype=..., device=...) -> nn.Module
        Policy whose ``forward(input_ids, attention_mask, pixel_values,
        action_tokens=None, temperature=1.0, calculate_entropy=False)`` returns
        a dict with ``logits`` ``[B, A, V]`` and, when ``action_tokens`` is
        given, ``log_probs`` ``[B, A]``. It must expose
        ``gradient_checkpointing_enable()`` and ``fsdp_wrap_modules()``.
    build_processor(model_path, unnorm_key) -> processor
        Provides ``num_action_tokens``, ``build_text_batch(instructions)``,
        ``preprocess_images(uint8 [B,H,W,3])``, ``tokens_to_env_actions(tokens)``.
    build_world_model(model_dir, device, **world_model_kwargs) -> model
        Provides ``cfg.chunk_size``, ``device``, ``context_frame_step`` (usable
        context lengths are ``1 + k * context_frame_step``) and
        ``predict_frames(ctx_frames, ctx_actions, tgt_actions, generator)``.
    build_reward_model(path, device) -> nn.Module
        Maps frames scaled to ``[-1, 1]`` to one success logit per frame.
    """

    build_policy: Callable[..., Any]
    build_processor: Callable[..., Any]
    build_world_model: Callable[..., Any]
    build_reward_model: Callable[..., Any]


def load_embodied_plugin(module_path: str) -> EmbodiedPlugin:
    if not module_path:
        raise ValueError("embodied.plugin must name a module that exposes PLUGIN")
    module = importlib.import_module(module_path)
    plugin = getattr(module, "PLUGIN", None)
    if not isinstance(plugin, EmbodiedPlugin):
        raise TypeError(f"{module_path}.PLUGIN must be an EmbodiedPlugin")
    return plugin
