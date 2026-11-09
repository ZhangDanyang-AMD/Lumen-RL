"""OpenVLA-OFT policy in the OpenSora action-conditioned world model (LIBERO-Spatial).

The policy and the frame success classifier are shared with the Wan project.
"""

from __future__ import annotations

from lumenrl.core.embodied_plugin import EmbodiedPlugin
from examples.Embodied_OpenSora_OpenVLA_OFT_LIBERO.models.opensora_world_model import (
    OpenSoraActionWorldModel,
    OpenSoraWorldModelConfig,
)
from examples.Embodied_Wan_OpenVLA_OFT_LIBERO.models.openvla_oft import (
    OpenVLAOFTProcessor,
    load_openvla_oft,
)
from examples.Embodied_Wan_OpenVLA_OFT_LIBERO.models.world_model_reward import load_reward_model


def _build_world_model(model_dir: str, device, **kwargs):
    return OpenSoraActionWorldModel(model_dir, OpenSoraWorldModelConfig(**kwargs), device=device)


PLUGIN = EmbodiedPlugin(
    build_policy=load_openvla_oft,
    build_processor=OpenVLAOFTProcessor,
    build_world_model=_build_world_model,
    build_reward_model=load_reward_model,
)
