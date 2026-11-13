"""Stands in for the miles plugin of the same name.

The radixark/Megatron-LM PR #28 call sites import
``miles_plugins.models.deepseek_v4.ops.hyper_connection`` by name. This module
satisfies that import with the tile_kernels-free implementation in ``mhc_shim``,
so the DSv4 increment can be exercised on stock megatron-core without pulling in
miles, tile_kernels or tilelang.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

from mhc_shim import (  # noqa: E402,F401
    DeepSeekV4HyperConnectionUtil,
    HCHeadParams,
    TorchHyperConnectionUtil,
)
