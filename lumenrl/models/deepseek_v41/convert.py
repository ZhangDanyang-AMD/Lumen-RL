"""Load the released DeepSeek-V4.1 checkpoint into `DSV41Model` (or a slice of it).

The checkpoint already uses the module paths of the reference implementation, which this
package keeps, so the only work is dtype: FP8 weights (E4M3, one E8M0 scale per 32x32 block)
and FP4 routed experts (E2M1 packed two per int8, one E8M0 scale per 32 inputs) dequantize to
BF16 -- exactly, since a 3-bit or 1-bit mantissa times a power of two fits BF16 -- and BF16
tensors the model keeps in fp32 are upcast. The Engram tables stay FP8 with their scales.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import torch
from torch import Tensor

FP8_BLOCK = 32
FP4_BLOCK = 32
_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)
_SKIP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")


def dequant_fp8_block(weight: Tensor, scale: Tensor, block: int = FP8_BLOCK) -> Tensor:
    """[out, in] E4M3 with [ceil(out/block), ceil(in/block)] scales -> BF16."""
    out_dim, in_dim = weight.shape
    s = scale.float().repeat_interleave(block, 0)[:out_dim].repeat_interleave(block, 1)[:, :in_dim]
    return (weight.float() * s).to(torch.bfloat16)


def dequant_fp4(weight: Tensor, scale: Tensor, block: int = FP4_BLOCK) -> Tensor:
    """[out, in/2] int8 holding two E2M1 values each (low nibble first) with [out, in/block]
    scales -> BF16 [out, in]."""
    x = weight.view(torch.uint8)
    table = _FP4_TABLE.to(weight.device)
    values = torch.stack([table[(x & 0x0F).long()], table[((x >> 4) & 0x0F).long()]], dim=-1).flatten(-2)
    return (values * scale.float().repeat_interleave(block, -1)).to(torch.bfloat16)


def checkpoint_name(model_key: str, original_layer_ids: tuple[int, ...]) -> str:
    """The checkpoint name of one of the model's state-dict keys (layer index -> released id)."""
    m = re.match(r"layers\.(\d+)\.(.*)", model_key)
    if not m:
        return model_key
    return f"layers.{original_layer_ids[int(m.group(1))]}.{m.group(2)}"


def _convert(name: str, tensor: Tensor, scales: dict[str, Tensor], target: Tensor) -> Tensor:
    if target.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        if tensor.dtype != target.dtype:
            raise TypeError(f"{name}: kept as {target.dtype} but stored as {tensor.dtype}")
        return tensor
    if tensor.dtype == torch.float8_e4m3fn:
        tensor = dequant_fp8_block(tensor, scales[name])
    elif tensor.dtype == torch.int8:
        tensor = dequant_fp4(tensor, scales[name])
    if tensor.shape != target.shape:
        raise ValueError(f"{name}: checkpoint shape {tuple(tensor.shape)} != model {tuple(target.shape)}")
    return tensor.to(target.dtype)


@torch.no_grad()
def load_released(model, ckpt_dir: str, device: str | torch.device = "cpu") -> dict:
    """Fill `model` from the released checkpoint at `ckpt_dir`. Returns counts and the names it
    left unused (other layers, MTP, vision). Raises if a model tensor has no source."""
    from safetensors import safe_open

    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    state = model.state_dict()
    wanted = {checkpoint_name(k, model.cfg.original_layer_ids): k for k in state}
    missing = [src for src in wanted if src not in index]
    if missing:
        raise KeyError(f"{len(missing)} model tensors have no checkpoint source, e.g. {missing[:3]}")
    # a weight's scale is consumed by dequantization, unless the model keeps it (Engram tables)
    dequant_scale = {}
    for src in wanted:
        sibling = src[: -len("weight")] + "scale" if src.endswith(".weight") else None
        if sibling in index and sibling not in wanted:
            dequant_scale[src] = sibling
    needed = set(wanted) | set(dequant_scale.values())
    by_file = defaultdict(list)
    for src in sorted(needed):
        by_file[index[src]].append(src)

    raw: dict[str, Tensor] = {}
    loaded = 0
    for file, names in sorted(by_file.items()):
        with safe_open(os.path.join(ckpt_dir, file), framework="pt", device=str(device)) as f:
            for src in names:
                raw[src] = f.get_tensor(src)
        # convert what is complete so far, so the raw FP8/FP4 tensors do not pile up
        for src in [s for s in raw if s in wanted]:
            scale_name = dequant_scale.get(src)
            if scale_name is not None and scale_name not in raw:
                continue
            scales = {src: raw.pop(scale_name)} if scale_name is not None else {}
            key = wanted[src]
            state[key].copy_(_convert(src, raw.pop(src), scales, state[key]))
            loaded += 1
    if loaded != len(wanted):
        raise RuntimeError(f"loaded {loaded} of {len(wanted)} tensors; left: {sorted(raw)[:5]}")
    unused = len(index) - len(needed)
    skipped_kinds = sorted({n.split(".")[0] for n in index if n.startswith(_SKIP_PREFIXES)})
    return {"loaded": loaded, "unused": unused, "skipped_kinds": skipped_kinds}
