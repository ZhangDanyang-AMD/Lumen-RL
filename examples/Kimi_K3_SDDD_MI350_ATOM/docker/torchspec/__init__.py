"""Compatibility shim mapping ``torchspec`` onto LumenRL's transfer layer.

ATOM's ``RLHFModelRunner.configure_hidden_states()`` imports its Mooncake
plumbing from ``torchspec``:

    from torchspec.config.mooncake_config import MooncakeConfig
    from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

That package is not shipped in ``rocm/atom-dev``. LumenRL already carries
equivalent implementations under ``lumenrl.transfer``, and both sides of the
wire must agree on the layout byte-for-byte anyway — the teacher writes with
this class and the trainer reads with ``lumenrl.transfer``'s. So rather than
vendor a second copy, this shim points ``torchspec`` at the LumenRL classes,
leaving exactly one implementation of the format.

Installed into site-packages at image build time; ATOM itself is unpatched.
"""

__all__ = ["config", "transfer"]
