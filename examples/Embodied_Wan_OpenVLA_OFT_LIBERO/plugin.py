"""OpenVLA-OFT policy in the Wan action-conditioned world model (LIBERO-Spatial)."""

from __future__ import annotations

from lumenrl.core.embodied_plugin import EmbodiedPlugin
from examples.Embodied_Wan_OpenVLA_OFT_LIBERO.models.openvla_oft import (
    OpenVLAOFTProcessor,
    load_openvla_oft,
)
from examples.Embodied_Wan_OpenVLA_OFT_LIBERO.models.wan_world_model import (
    WanActionWorldModel,
    WanWorldModelConfig,
)
from examples.Embodied_Wan_OpenVLA_OFT_LIBERO.models.world_model_reward import load_reward_model


def _build_world_model(model_dir: str, device, **kwargs):
    return WanActionWorldModel(
        model_dir, WanWorldModelConfig(**kwargs), device=device,
    )


PLUGIN = EmbodiedPlugin(
    build_policy=load_openvla_oft,
    build_processor=OpenVLAOFTProcessor,
    build_world_model=_build_world_model,
    build_reward_model=load_reward_model,
)
