"""Load an exported DSpark draft (ATOM tensor names) back into DSparkModel.

This is the exact inverse of ``output/export_dspark_hf.py``. It exists because
continuing training from a released draft is otherwise a silent no-op: the
generic ``draft.resume_from`` path in the trainer renames ``midlayer.`` to
``layers.0.`` and ``norm`` to ``out_norm``, which is Eagle3's layout, so every
DSpark tensor lands as an unexpected key and ``strict=False`` keeps the random
initialisation. Nothing raises, loss starts near a cold-start value, and the run
looks like it merely converges slowly.

The same failure already cost this project once in the other direction (P14: the
step-640 export wrote LumenRL names where ATOM reads ``context_proj`` /
``context_norm`` / ``final_norm``, and the served draft accepted 0.00% of tokens
across 3060 forward steps while every log line looked healthy). So the contract
here is checked, not assumed: every tensor ATOM emits must be consumed, and every
draft parameter except the frozen embedding must be filled.
"""

from __future__ import annotations

import glob
import logging
import os

import torch

logger = logging.getLogger(__name__)

# ATOM name -> DSparkModel name. Everything else (layers.*, markov_head.*,
# confidence_head.*) is already identical on both sides.
_RENAMES = {
    "context_proj.weight": "fc.weight",
    "context_norm.weight": "hidden_norm.weight",
    "final_norm.weight": "norm.weight",
}

# The teacher's frozen embedding travels with the export so ATOM can serve the
# draft standalone. The trainer gets it from the teacher instead, and DSparkModel
# has no parameter for it.
_DROP = {"embed_tokens.weight", "lm_head.weight"}


def load_dspark_from_hf(model: torch.nn.Module, path: str) -> int:
    """Load ``path``'s safetensors into ``model``. Returns tensors loaded.

    Raises if any draft parameter would be left at its initial value, or if the
    export carries a tensor this loader does not know how to place.
    """
    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {path!r}")

    raw: dict[str, torch.Tensor] = {}
    for shard in shards:
        raw.update(load_file(shard, device="cpu"))

    mapped: dict[str, torch.Tensor] = {}
    unknown: list[str] = []
    for key, tensor in raw.items():
        if key in _DROP:
            continue
        target = _RENAMES.get(key, key)
        if target not in dict(model.named_parameters()) and target not in dict(
            model.named_buffers()
        ):
            unknown.append(key)
            continue
        mapped[target] = tensor
    if unknown:
        raise RuntimeError(
            f"{path}: {len(unknown)} exported tensors have no home in "
            f"DSparkModel: {sorted(unknown)[:8]}. Loading anyway would train a "
            "draft that silently kept part of its random initialisation."
        )

    own = {name for name, _ in model.named_parameters()}
    missing = sorted(own - set(mapped))
    if missing:
        raise RuntimeError(
            f"{path}: {len(missing)} draft parameters are not in the export: "
            f"{missing[:8]}. This is the P14 failure mode -- a checkpoint that "
            "loads cleanly but leaves whole subnetworks untrained."
        )

    for name, param in model.named_parameters():
        loaded = mapped[name]
        if tuple(loaded.shape) != tuple(param.shape):
            raise RuntimeError(
                f"{path}: shape mismatch for {name}: export {tuple(loaded.shape)} "
                f"vs model {tuple(param.shape)}"
            )

    info = model.load_state_dict(mapped, strict=False)
    if info.unexpected_keys:
        raise RuntimeError(
            f"{path}: load_state_dict rejected {info.unexpected_keys[:8]}"
        )
    logger.info(
        "Loaded DSpark draft from %s: %d tensors, %d parameters",
        path,
        len(mapped),
        sum(t.numel() for t in mapped.values()),
    )
    return len(mapped)


__all__ = ["load_dspark_from_hf"]
