"""miles' sparse-MQA entry point, backed by PyTorch instead of tilelang.

The LumenRL image deliberately ships without tilelang / tile_kernels. The real
kernel is replaced by ``sparse_mqa_torch.sparse_mqa_attention``, which was read
off the tilelang source and validated against it in probe 5.4: forward and all
three gradients agree, and each is at least as close to an fp64 reference as the
kernel is.

``sparse_attn_tilelang`` is a pure function of activations plus ``attn_sink`` and
declares no parameters, so this substitution leaves every checkpoint key intact.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
)

from sparse_mqa_torch import sparse_attn_tilelang, sparse_mqa_attention  # noqa: E402,F401
