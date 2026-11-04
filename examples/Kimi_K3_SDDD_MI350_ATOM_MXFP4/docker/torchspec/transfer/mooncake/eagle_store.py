"""ATOM adapter for LumenRL's segmented Mooncake producer."""

from lumenrl.transfer.eagle_mooncake_store import SegmentedEagleMooncakeStore


class EagleMooncakeStore(SegmentedEagleMooncakeStore):
    """Use the reusable producer pool under ATOM's expected class name."""


__all__ = ["EagleMooncakeStore"]
