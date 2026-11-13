"""miles' DSA-indexer scoring entry points, backed by PyTorch instead of tilelang.

Validated against the real kernel in probe 5.7. Both symbols are pure functions
of activations, so nothing about the checkpoint changes.

The alternative -- Megatron's dense ``DSAIndexer`` fallback -- is not usable:
the fork's dsv4 branch calls ``wrapped_precompute_freqs_cis`` with a signature
miles does not ship (handoff §2.2's fork/miles skew), so that path cannot even be
constructed. Keeping ``V4Indexer`` also keeps the checkpoint parameter names.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
)

from indexer_torch import (  # noqa: E402,F401
    _make_causal_cu_seqlens,
    batched_indexer_fwd,
)
