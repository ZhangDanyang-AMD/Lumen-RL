"""Draft model architectures for speculative decoding distillation."""

from __future__ import annotations

from lumenrl.models.eagle3 import Eagle3Model
from lumenrl.models.dflash import DFlashModel
from lumenrl.models.dspark import DSparkModel

__all__ = [
    "DFlashModel",
    "DSparkModel",
    "Eagle3Model",
]
