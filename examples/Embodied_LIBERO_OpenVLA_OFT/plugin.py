"""OpenVLA-OFT policy in the real LIBERO simulator (robosuite / MuJoCo, CPU OSMesa)."""

from lumenrl.core.embodied_plugin import EmbodiedPlugin
from examples.Embodied_Wan_OpenVLA_OFT_LIBERO.models.openvla_oft import (
    OpenVLAOFTProcessor,
    load_openvla_oft,
)

PLUGIN = EmbodiedPlugin(
    build_policy=load_openvla_oft,
    build_processor=OpenVLAOFTProcessor,
)
