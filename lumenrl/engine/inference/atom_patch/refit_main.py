"""Selected MXFP4-2D refit over the pinned ATOM main weight-updater ABI.

This is an explicit, default-off worker-process adapter. ATOM owns IPC/SHM
transport, reader fences, KV clearing, failure fencing and global FINISH.
The adapter owns the BF16 source to packed weight+scale translation and refuses
an incomplete version *before* a worker can acknowledge the last bucket.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
from collections.abc import Mapping

_TRUE = frozenset(("1", "true", "yes", "on"))


def _selected(module, param_name):
    return getattr(module, "mxfp4_2d", False) is True and param_name == "weight"


def install() -> bool:
    """Install only on the worker, before its first weight-update RPC."""
    if os.environ.get("LUMEN_ATOM_MXFP4_2D", "0").strip().lower() not in _TRUE:
        return False
    import torch
    from atom.rollout.weight_updater import WeightUpdaterMixin
    from lumenrl.engine.inference.atom_patch.quant_main import quantize_mxfp4_2d_weight

    cls = WeightUpdaterMixin
    if getattr(cls, "_lumen_mxfp4_2d_refit_installed", False):
        return True
    required = (
        "_get_param_to_module_mapping", "_resolve_packed_name",
        "_get_packed_shard_order", "_sync_target_model", "_copy_into_param",
        "_apply_unmatched_weight", "_apply_packed_weight",
        "_discard_failed_weight_sync", "update_weights",
        "update_weights_from_shm", "update_weights_from_ipc",
    )
    for name in required:
        if not callable(getattr(cls, name, None)):
            raise RuntimeError(f"ATOM main MXFP4-2D refit lacks {name}")
    if tuple(inspect.signature(cls.update_weights_from_ipc).parameters)[1:4] != (
        "ipc_handle", "bucket_meta", "is_last"
    ):
        raise RuntimeError("ATOM main IPC updater signature changed")

    old_mapping = cls._get_param_to_module_mapping
    old_packed = cls._apply_packed_weight
    old_discard = cls._discard_failed_weight_sync
    old_direct = cls.update_weights
    old_shm = cls.update_weights_from_shm
    old_ipc = cls.update_weights_from_ipc
    old_copy = cls._copy_into_param
    old_load = cls._load_into_param
    old_is_fp8 = cls._is_fp8_param

    def _state(self):
        state = self.__dict__.get("_lumen_mxfp4_refit_state")
        if state is None:
            state = {"seen": set(), "generation": 0,
                     "committed": None, "expected": None}
            self.__dict__["_lumen_mxfp4_refit_state"] = state
        return state

    def _targets(self):
        mapping = old_mapping(self)
        targets = {
            id(module): (name, module, param)
            for name, (module, pname, param) in mapping.items()
            if _selected(module, pname)
        }
        if not targets:
            raise RuntimeError("MXFP4-2D selected but model has no 2D dense targets")
        if self.world_size != 1:
            raise ValueError("MXFP4-2D refit supports TP1 only")
        return mapping, targets

    def _begin(self):
        state = _state(self)
        if state["expected"] is None:
            _, targets = _targets(self)
            state["expected"] = set(targets)
            state["seen"].clear()
        return state

    def _requantize(self, module, param, tensor):
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != tuple(
            module._mxfp4_2d_shape
        ):
            raise ValueError("MXFP4-2D refit needs complete BF16 logical weight")
        state = _begin(self)
        key = id(module)
        if key in state["seen"]:
            raise ValueError("duplicate MXFP4-2D target in one weight sync")
        weight, scale = quantize_mxfp4_2d_weight(tensor.to(self.device))
        if weight.shape != param.shape or scale.shape != module.weight_scale.shape:
            raise ValueError("MXFP4-2D refit changes live parameter storage")
        # Keep main's per-parameter reader fence and captured address.
        old_copy(self, param, weight.view(param.dtype))
        old_copy(self, module.weight_scale, scale.view(module.weight_scale.dtype))
        state["seen"].add(key)

    @functools.wraps(old_packed)
    def packed(self, name, tensor, mapping):
        resolved = self._resolve_packed_name(name, mapping)
        if resolved is None:
            return old_packed(self, name, tensor, mapping)
        atom_name, shard_id, target_suffix = resolved
        module, pname, param = mapping[atom_name]
        if not _selected(module, pname):
            return old_packed(self, name, tensor, mapping)
        if tensor.dtype != torch.bfloat16:
            raise ValueError("MXFP4-2D packed source shard must be BF16")
        loader = getattr(module, "weight_loader", None)
        order = self._get_packed_shard_order().get(target_suffix, [])
        if not callable(loader) or not order or shard_id not in order:
            raise RuntimeError("MXFP4-2D packed shard has no validated loader/order")
        _begin(self)
        accumulator = self.__dict__.setdefault("_packed_weight_accum", {})
        shards = accumulator.setdefault(atom_name, {"shards": {}})["shards"]
        if shard_id in shards:
            raise ValueError(f"duplicate MXFP4-2D packed shard {name}")
        shards[shard_id] = tensor.clone()
        if set(shards) != set(order):
            return "accumulated"
        # Historical recipe uses FP32 assembly then casts the complete matrix
        # back to BF16. Preserve this exact quantization source.
        buf = torch.nn.Parameter(
            torch.zeros(module._mxfp4_2d_shape, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        wlp = getattr(param, "weight_loader_process", None)
        if wlp is not None:
            buf.weight_loader_process = wlp
        for sid in order:
            loader(buf, shards[sid].to(device=self.device, dtype=torch.float32), sid)
        _requantize(self, module, param, buf.data.to(torch.bfloat16))
        del accumulator[atom_name]
        return "updated"

    @functools.wraps(old_discard)
    def discard(self):
        state = self.__dict__.get("_lumen_mxfp4_refit_state")
        if state is not None:
            state["expected"] = None
            state["seen"].clear()
        return old_discard(self)

    def _complete(self):
        state = _begin(self)
        if state["seen"] != state["expected"]:
            raise RuntimeError(
                f"incomplete MXFP4-2D sync: {len(state['seen'])}/{len(state['expected'])}"
            )
        if self.__dict__.get("_packed_weight_accum"):
            raise RuntimeError("incomplete packed weight shards at final bucket")
        state["generation"] += 1
        state["committed"] = state["generation"]
        state["expected"] = None
        state["seen"].clear()
        if os.environ.get("ATOM_ALIGNED_KV_PACK", "0").strip().lower() in _TRUE:
            from lumenrl.engine.inference.atom_patch.aligned_kv_pack import reset_plan_cache
            reset_plan_cache()
        return state["committed"]

    def _intercept_main_writes(self, mapping):
        # Main branches may pass the original BF16 tensor to either its plain
        # copy or weight_loader fallback. Requantize at that *exact* write seam
        # rather than changing main transport or hiding its completed counts.
        selected = {
            id(param): (module, param)
            for module, pname, param in mapping.values() if _selected(module, pname)
        }
        if not selected:
            raise RuntimeError("missing MXFP4-2D targets")
        return selected

    def wrap_update(name, original):
        @functools.wraps(original)
        def update(self, *args, **kwargs):
            mapping, _ = _targets(self)
            if name == "update_weights":
                incoming = args[0] if args else kwargs.get("named_tensors", ())
                keys = [key for key, _ in incoming]
            else:
                meta = args[1] if len(args) > 1 else kwargs.get("bucket_meta", {})
                keys = list(meta)
            direct_targets = {
                k for k in keys if k in mapping and
                _selected(mapping[k][0], mapping[k][1])
            }
            if len(direct_targets) != sum(k in direct_targets for k in keys):
                raise ValueError("duplicate MXFP4-2D direct weight in one bucket")
            # Fail before entering main's broad weight_loader exception handler:
            # it logs and counts a failed direct 2D load as 'skipped'.
            for key in direct_targets:
                module, _, _ = mapping[key]
                if name == "update_weights":
                    source = next(value for n, value in incoming if n == key)
                    dtype, shape = source.dtype, tuple(source.shape)
                else:
                    info = meta[key]
                    dtype = getattr(torch, str(info["dtype"]).removeprefix("torch."), None)
                    shape = tuple(info["shape"])
                if dtype != torch.bfloat16 or shape != tuple(module._mxfp4_2d_shape):
                    raise ValueError(f"MXFP4-2D BF16 source mismatch: {key}")
            state = _begin(self)
            if any(id(mapping[k][0]) in state["seen"] for k in direct_targets):
                raise ValueError("repeated MXFP4-2D parameter in one sync")
            selected = _intercept_main_writes(self, mapping)
            # Per-call routing is confined to this ModelRunner. Main's read
            # and dispatch still run, including its IPC buffer lifetime.
            direct_seen = set()

            def intercept_copy(param, tensor):
                target = selected.get(id(param))
                if target is None:
                    return old_copy(self, param, tensor)
                module, actual = target
                if id(module) in direct_seen:
                    raise ValueError("duplicate direct MXFP4-2D write")
                direct_seen.add(id(module))
                _requantize(self, module, actual, tensor)

            def intercept_loader(param, loader, tensor, *load_args, **load_kw):
                target = selected.get(id(param))
                if target is None:
                    return old_load(self, param, loader, tensor, *load_args, **load_kw)
                module, actual = target
                if id(module) in direct_seen:
                    raise ValueError("duplicate direct MXFP4-2D loader")
                direct_seen.add(id(module))
                if tuple(tensor.shape) != tuple(module._mxfp4_2d_shape):
                    raise ValueError("MXFP4-2D loader source shape mismatch")
                _requantize(self, module, actual, tensor.to(torch.bfloat16))

            # Bind only on the current worker instance; all other workers and
            # quantized non-2D parameters still call the original methods.
            had_copy = "_copy_into_param" in self.__dict__
            had_load = "_load_into_param" in self.__dict__
            copy_bound = self.__dict__.get("_copy_into_param")
            load_bound = self.__dict__.get("_load_into_param")
            self._copy_into_param = intercept_copy
            self._load_into_param = intercept_loader
            try:
                result = original(self, *args, **kwargs)
                is_last = (
                    (args[1] if len(args) > 1 else kwargs.get("clear_kv_cache", True))
                    if name == "update_weights"
                    else (args[2] if len(args) > 2 else kwargs.get("is_last", True))
                )
                if is_last:
                    _complete(self)
                return result
            except Exception:
                discard(self)
                raise
            finally:
                if had_copy:
                    self._copy_into_param = copy_bound
                else:
                    self.__dict__.pop("_copy_into_param", None)
                if had_load:
                    self._load_into_param = load_bound
                else:
                    self.__dict__.pop("_load_into_param", None)
        return update

    def get_lumen_refit_receipt(self, expect_committed=True, audit_name=None):
        state = _state(self)
        if state["expected"] is not None:
            raise RuntimeError("weight refit receipt requested during partial sync")
        if expect_committed and state["committed"] is None:
            raise RuntimeError("no complete MXFP4-2D refit has been committed")
        from lumenrl.engine.inference.atom_patch import alignment_main

        counters = getattr(alignment_main, "_lumen_c1_counters", None)
        if os.environ.get("ATOM_ALIGNED_KV_PACK", "0").strip().lower() in _TRUE:
            if not isinstance(counters, dict):
                raise RuntimeError("selected C1 counters missing on worker")
        _, targets = _targets(self)
        receipt = {
            "rank": self.rank,
            "mxfp4_2d_target_count": len(targets),
            "refit_generation": state["committed"],
            "c1_hit": 0 if counters is None else counters["hit"],
            "c1_fallback": 0 if counters is None else counters["fallback"],
            "refit_installed": True,
        }
        if os.environ.get("LUMENRL_SMCI_GPU_IDENTITY_MANIFEST"):
            from uuid import UUID

            if torch.cuda.device_count() != 1:
                raise RuntimeError("SMCI ModelRunner must see exactly one GPU")
            prop = torch.cuda.get_device_properties(0)
            unique_id = UUID(str(prop.uuid)).bytes.decode("ascii").lower()
            pci = (f"{prop.pci_domain_id:04X}:{prop.pci_bus_id:02X}:"
                   f"{prop.pci_device_id:02X}.0")
            if (unique_id != os.environ.get("LUMENRL_EXPECTED_GPU_UUID") or
                    pci != os.environ.get("LUMENRL_EXPECTED_GPU_PCI")):
                raise RuntimeError(f"SMCI ModelRunner hardware mismatch: {unique_id} {pci}")
            receipt["physical_gpu_uuid"] = unique_id
            receipt["physical_gpu_pci"] = pci
        if audit_name == "*":
            receipt["audit_target_names"] = sorted(
                name for name, (module, pname, _) in
                self._get_param_to_module_mapping().items()
                if _selected(module, pname)
            )[:12]
            return receipt
        if audit_name is not None:
            entry = self._get_param_to_module_mapping().get(audit_name)
            if entry is None or not _selected(entry[0], entry[1]):
                raise ValueError(f"unknown MXFP4-2D audit target: {audit_name}")
            module, _, weight = entry

            def digest(tensor):
                raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                return hashlib.sha256(raw).hexdigest()

            receipt["audit"] = {
                "name": audit_name,
                "weight_sha256": digest(weight),
                "scale_sha256": digest(module.weight_scale),
                "weight_shape": tuple(weight.shape),
                "scale_shape": tuple(module.weight_scale.shape),
                "generation": state["committed"],
            }
        return receipt

    def is_fp8_param(module, param):
        if getattr(module, "mxfp4_2d", False) is True and param is module.weight:
            return False
        return old_is_fp8(module, param)

    cls._is_fp8_param = staticmethod(is_fp8_param)
    cls.get_lumen_refit_receipt = get_lumen_refit_receipt
    cls._apply_packed_weight = packed
    cls._discard_failed_weight_sync = discard
    for name, original in (
        ("update_weights", old_direct), ("update_weights_from_shm", old_shm),
        ("update_weights_from_ipc", old_ipc),
    ):
        setattr(cls, name, wrap_update(name, original))
    cls._lumen_mxfp4_2d_refit_installed = True
    return True
