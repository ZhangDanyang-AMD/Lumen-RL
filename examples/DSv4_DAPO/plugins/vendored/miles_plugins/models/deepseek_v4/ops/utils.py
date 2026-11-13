# Modified from radixark/miles@5400905d5 (Apache-2.0): PyTorch Hadamard fallback and
# a re-export of wrapped_precompute_freqs_cis. See ../../../../NOTICE.
import os
import sys

import torch

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    # Absent from the LumenRL image; fall back to the PyTorch butterfly, which
    # probe 5.6 checks against fast_hadamard_transform 1.0.4.
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
    )
    from hadamard_torch import hadamard_transform


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Scaled Hadamard transform used to redistribute activation energy before
    QAT. Consumed by both the attention compressor and the DSA indexer.
    """
    assert x.dtype == torch.bfloat16
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)


# megatron's patched dsa.py imports wrapped_precompute_freqs_cis from here, but
# miles defines it in ops/rope.py. Handoff §2.2 flagged this fork/miles skew;
# re-exporting keeps both import paths working.
from miles_plugins.models.deepseek_v4.ops.rope import (  # noqa: E402,F401
    wrapped_precompute_freqs_cis,
)
