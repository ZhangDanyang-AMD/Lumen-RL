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


def _dequant(name: str, tensor: Tensor, scale: Tensor | None, dtype: torch.dtype) -> Tensor:
    if dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        if tensor.dtype != dtype:
            raise TypeError(f"{name}: kept as {dtype} but stored as {tensor.dtype}")
        return tensor
    if tensor.dtype == torch.float8_e4m3fn:
        tensor = dequant_fp8_block(tensor, scale)
    elif tensor.dtype == torch.int8:
        tensor = dequant_fp4(tensor, scale)
    return tensor.to(dtype)


def _sources(model) -> dict[str, list[str]]:
    """Each state-dict key of `model` -> the reference keys it is made of, concatenated in that
    order along dim 0. A module laid out differently from the reference (`MegatronMoE`) says how
    with `reference_names(key)`; a key it maps to nothing is not loaded."""
    sources = {k: [k] for k in model.state_dict()}
    for name, module in model.named_modules():
        remap = getattr(module, "reference_names", None)
        if not name or remap is None:
            continue
        prefix = name + "."
        for key in [k for k in sources if k.startswith(prefix)]:
            refs = remap(key[len(prefix) :])
            if refs:
                sources[key] = [prefix + r for r in refs]
            else:
                del sources[key]
    return sources


_SAFETENSORS_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8, "U8": torch.uint8,
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
}


def mapped_tensor(path: str, name: str) -> Tensor:
    """Tensor `name` of the safetensors file `path` as a read-only CPU tensor over the file's
    pages: every process that maps the same file shares one copy in the page cache, and nothing
    is read until a row is."""
    import struct
    import warnings

    import numpy as np

    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(size))[name]
    begin, end = meta["data_offsets"]
    pages = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size + begin, shape=(end - begin,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch warns that the array is read-only; the table is frozen
        raw = torch.from_numpy(pages)
    return raw.view(_SAFETENSORS_DTYPES[meta["dtype"]]).view(meta["shape"])


def _attach_host_tables(model, ckpt_dir: str, index: dict[str, str]) -> int:
    from lumenrl.models.deepseek_v41.engram import EngramTable

    attached = 0
    for name, module in model.named_modules():
        if isinstance(module, EngramTable) and module.on_host:
            src = checkpoint_name(name, model.cfg.original_layer_ids)
            weight, scale = f"{src}.weight", f"{src}.scale"
            for t in (weight, scale):
                if t not in index:
                    raise KeyError(f"host engram table {name}: no checkpoint source {t}")
            module.attach(
                mapped_tensor(os.path.join(ckpt_dir, index[weight]), weight),
                mapped_tensor(os.path.join(ckpt_dir, index[scale]), scale),
            )
            attached += 1
    return attached


@torch.no_grad()
def load_released(model, ckpt_dir: str, device: str | torch.device = "cpu") -> dict:
    """Fill `model` from the released checkpoint at `ckpt_dir`; Engram tables kept on the host
    are mapped from the file instead. Returns counts and the names it left unused (other layers,
    other ranks' experts, MTP, vision). Raises if a model tensor has no source."""
    from safetensors import safe_open

    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    state = model.state_dict()
    ids = model.cfg.original_layer_ids
    parts = {key: [checkpoint_name(r, ids) for r in refs] for key, refs in _sources(model).items()}
    wanted = {src for srcs in parts.values() for src in srcs}
    missing = sorted(src for src in wanted if src not in index)
    if missing:
        raise KeyError(f"{len(missing)} model tensors have no checkpoint source, e.g. {missing[:3]}")
    # a weight's scale is consumed by dequantization, unless the model keeps it (Engram tables)
    dequant_scale = {}
    for src in wanted:
        sibling = src[: -len("weight")] + "scale" if src.endswith(".weight") else None
        if sibling in index and sibling not in wanted:
            dequant_scale[src] = sibling
    needed = wanted | set(dequant_scale.values())
    by_file = defaultdict(list)
    for src in sorted(needed):
        by_file[index[src]].append(src)

    raw: dict[str, Tensor] = {}
    pending = dict(parts)
    for file, names in sorted(by_file.items()):
        with safe_open(os.path.join(ckpt_dir, file), framework="pt", device=str(device)) as f:
            for src in names:
                raw[src] = f.get_tensor(src)
        # convert what is complete so far, so the raw FP8/FP4 tensors do not pile up
        for key, srcs in list(pending.items()):
            if not all(s in raw and dequant_scale.get(s, s) in raw for s in srcs):
                continue
            target = state[key]
            tensor = torch.cat(
                [_dequant(s, raw.pop(s), raw.pop(dequant_scale[s]) if s in dequant_scale else None, target.dtype)
                 for s in srcs]
            )
            if tensor.shape != target.shape:
                raise ValueError(f"{key}: checkpoint {srcs} gives {tuple(tensor.shape)}, model has {tuple(target.shape)}")
            target.copy_(tensor)
            del pending[key]
    if pending:
        raise RuntimeError(f"loaded {len(parts) - len(pending)} of {len(parts)} tensors; left: {sorted(pending)[:5]}")
    mapped = _attach_host_tables(model, ckpt_dir, index)
    unused = len(index) - len(needed) - 2 * mapped
    skipped_kinds = sorted({n.split(".")[0] for n in index if n.startswith(_SKIP_PREFIXES)})
    return {"loaded": len(parts), "mapped": mapped, "unused": unused, "skipped_kinds": skipped_kinds}


def sliced_hf_config(raw: dict, keep: list[int]) -> dict:
    """The released config.json for the stack of layers `keep`, renumbered 0..len(keep)-1, with
    no draft (DSpark) layers. Refuses a slice that drops a source a kept layer reads, and one
    that moves an Engram layer: ATOM seeds an Engram layer's hash with its position and draws its
    primes in list order, so a kept Engram layer must keep its place and the kept ones must be a
    prefix of the released list."""
    from lumenrl.models.deepseek_v41.config import DSV41Config

    keep = list(keep)
    DSV41Config.from_hf(raw).slice(keep)
    text = raw.get("text_config", raw)
    pos = {orig: k for k, orig in enumerate(keep)}
    engram_ids = list(text.get("engram_layer_ids", ()))
    kept = [i for i in engram_ids if i in pos]
    if kept != engram_ids[: len(kept)] or any(pos[i] != i for i in kept):
        raise ValueError(f"the slice moves Engram layers {kept} of {engram_ids}; ATOM would hash them differently")
    cand = text.get("candidate_source_layer_id")
    new = {
        **text,
        "num_hidden_layers": len(keep),
        "compress_ratios": [text["compress_ratios"][i] for i in keep],
        "kv_source_layer_ids": [pos[i] for i in text.get("kv_source_layer_ids", ()) if i in pos],
        "index_source_layer_ids": [pos[i] for i in text.get("index_source_layer_ids", ()) if i in pos],
        "candidate_source_layer_id": pos.get(cand, -1) if cand is not None and cand >= 0 else -1,
        "engram_layer_ids": kept,
        "engram_num_embeddings": list(text.get("engram_num_embeddings", ()))[: len(kept)],
        "num_nextn_predict_layers": 0,
    }
    if "dspark_target_layer_ids" in text:
        new["dspark_target_layer_ids"] = []
    return {**raw, "text_config": new} if "text_config" in raw else new


def write_slice_checkpoint(src_dir: str, dst_dir: str, keep: list[int]) -> dict:
    """Write the released checkpoint of the stack `keep` to `dst_dir` in the released format
    (what ATOM loads): kept layers renamed to their new positions, other layers and the draft
    layers left out, everything else (embedding, head, vision) as it is. A shard none of whose
    tensors changes name is symlinked, not copied -- for slice B that is the 98 GB Engram table
    of layer 1. Other non-weight files (tokenizer, ...) are symlinked too."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = os.path.abspath(src_dir), os.path.abspath(dst_dir)
    with open(os.path.join(src, "config.json")) as f:
        raw = json.load(f)
    config = sliced_hf_config(raw, keep)
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    pos = {orig: k for k, orig in enumerate(keep)}

    def target(name: str) -> str | None:
        m = re.match(r"layers\.(\d+)\.(.*)", name)
        if m:
            i = int(m.group(1))
            return f"layers.{pos[i]}.{m.group(2)}" if i in pos else None
        return None if name.startswith("mtp.") else name

    os.makedirs(dst)
    by_shard = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)
    new_map, linked, written = {}, 0, 0
    for shard, names in sorted(by_shard.items()):
        renamed = {n: t for n in names if (t := target(n)) is not None}
        if not renamed:
            continue
        if len(renamed) == len(names) and all(n == t for n, t in renamed.items()):
            os.symlink(os.path.realpath(os.path.join(src, shard)), os.path.join(dst, shard))
            linked += 1
        else:
            with safe_open(os.path.join(src, shard), framework="pt") as f:
                tensors = {t: f.get_tensor(n) for n, t in renamed.items()}
                metadata = f.metadata() or {"format": "pt"}
            save_file(tensors, os.path.join(dst, shard), metadata=metadata)
            del tensors
            written += 1
        new_map.update({t: shard for t in renamed.values()})
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": dict(sorted(new_map.items()))}, f, indent=2)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    rewritten = {"config.json", "model.safetensors.index.json", *by_shard}
    for entry in sorted(os.listdir(src)):
        path = os.path.join(src, entry)
        if os.path.isfile(path) and entry not in rewritten:
            os.symlink(os.path.realpath(path), os.path.join(dst, entry))
    return {"tensors": len(new_map), "linked_shards": linked, "written_shards": written}
