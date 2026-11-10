"""Walsh-Hadamard transform in pure PyTorch.

Stands in for ``fast_hadamard_transform.hadamard_transform``, which is not in the
LumenRL image. DSv4 uses it as ``rotate_activation`` to redistribute activation
energy before QAT, in the attention compressor and the DSA indexer.

Sylvester-order H_n for power-of-two n, computed with the usual in-place
butterfly: at stride h, combine the pairs (j, j+h) inside each block of 2h as
(a+b, a-b). Validated against fast_hadamard_transform 1.0.4 in probe 5.6.
"""

import torch
from torch import Tensor


def hadamard_transform(x: Tensor, scale: float = 1.0) -> Tensor:
    """Transform along the last axis, which must be a power of two."""
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f'hadamard_transform needs a power-of-two last dim, got {n}')

    shape = x.shape
    out = x.reshape(-1, n).float()

    h = 1
    while h < n:
        out = out.view(-1, n // (2 * h), 2, h)
        a, b = out[:, :, 0, :], out[:, :, 1, :]
        out = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2

    return (out * scale).reshape(shape).to(x.dtype)
