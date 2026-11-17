"""OpenVLA-OFT policy for the LIBERO simulator.

``modeling_openvla_oft`` and ``processing_openvla_oft`` stay in
``Lumen/examples/openvla_oft``, which ``run_embodied_grpo.sh`` puts on
``PYTHONPATH``.
"""

from __future__ import annotations

from lumenrl.core.embodied_plugin import EmbodiedPlugin
from modeling_openvla_oft import load_openvla_oft
from processing_openvla_oft import OpenVLAOFTProcessor

PLUGIN = EmbodiedPlugin(
    build_policy=load_openvla_oft,
    build_processor=OpenVLAOFTProcessor,
)
