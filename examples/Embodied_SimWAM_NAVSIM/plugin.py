"""Lumen kernel switch for SimWAM FlowGRPO.

Called after ``configure_grpo`` so the video expert, the LoRA-wrapped action
expert, the frozen reference expert and the VAE are all in the module tree.
Quantization stays off.

* ``hf_attn_patch`` — ``F.scaled_dot_product_attention``. A bool keep-mask
  (True = attend) is converted to an additive bias inside the patch.
* ``lumen_linear`` — every ``nn.Linear``. LoRA A/B are bare parameters, so
  the same ``lumen.ops.bf16_gemm.linear`` is installed on ``LoRALinear.forward``.
* ``aiter_conv`` — ``nn.Conv2d`` and ``nn.Conv3d`` go through AITER.
  ``CausalConv3d.forward`` is left in place; only ``_conv_forward`` changes.

Each patch is applied on its own, so one kernel failing to load does not turn
the others off.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_OFF = {"", "0", "false", "none", "off"}
# counters of the verify modes, by patch
_STATS = {}
_PATCHES = ("hf_attn_patch", "lumen_linear", "aiter_conv")


def _log_coverage(model) -> None:
    import torch.nn as nn

    n_linear = n_conv2d = n_conv3d = 0
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            n_conv3d += 1
        elif isinstance(module, nn.Conv2d):
            n_conv2d += 1
        elif isinstance(module, nn.Linear):
            n_linear += 1
    parts = []
    for name in ("video_expert", "action_expert", "ref_action_expert", "vae"):
        child = getattr(model, name, None)
        if child is None:
            continue
        nparam = sum(p.numel() for p in child.parameters()) / 1e9
        parts.append(f"{name}={nparam:.2f}B")
    logger.info(
        "SimWAM coverage: %s | nn.Linear=%d nn.Conv2d=%d nn.Conv3d=%d",
        ", ".join(parts) or "unpacked", n_linear, n_conv2d, n_conv3d,
    )


def _patch_lora_gemm() -> None:
    """Route LoRA A/B through the same bf16 GEMM as ``nn.Linear``.

    ``lora_A`` / ``lora_B`` are bare parameters, so ``lumen_linear`` does not
    see them. Patching the class covers adapters created before or after.
    """
    from simwam.models.wan22.lora import LoRALinear

    if getattr(LoRALinear.forward, "_lumen_bf16", False):
        return

    def forward(self, x):
        from lumen.ops.bf16_gemm import linear

        out = self.base(x)
        dropped = self.lora_dropout(x)
        mid = linear(dropped, self.lora_A)
        lora = linear(mid, self.lora_B)
        return out + self.scaling * lora

    forward._lumen_bf16 = True
    LoRALinear.forward = forward


def install_grpo_hook() -> None:
    """Apply Lumen after ``configure_grpo`` without editing SimWAM sources.

    ``configure_grpo`` copies the frozen reference expert and wraps LoRA on
    the action expert. Patching any earlier would miss those modules.
    """
    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin

    if getattr(ActionGRPOMixin.configure_grpo, "_lumen_hook", False):
        return
    original = ActionGRPOMixin.configure_grpo

    def configure_grpo(self, grpo_cfg):
        original(self, grpo_cfg)
        spec = os.environ.get("LUMEN_ENABLE", "").strip()
        if spec:
            enable_lumen(self, spec)

    configure_grpo._lumen_hook = True
    ActionGRPOMixin.configure_grpo = configure_grpo


def install_lora_init_seed() -> None:
    """Seed every RNG right before ``configure_grpo``, i.e. before LoRA A is initialised.

    On the NAVSIM path SimWAM seeds only in the trainer, after ``configure_grpo`` has drawn
    LoRA A with ``kaiming_uniform_``, so two runs of the same config start from different
    adapters. This calls SimWAM's ``set_global_seed`` (which adds the rank) first; the trainer
    re-seeds later exactly as before, so rollout noise is unchanged.
    ``SIMWAM_GRPO_LORA_SEED`` sets the seed (default 42, the config's ``seed``); ``off`` disables.
    """
    value = os.environ.get("SIMWAM_GRPO_LORA_SEED", "42").strip()
    if value.lower() in _OFF:
        return
    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin
    from simwam.utils.pytorch_utils import set_global_seed

    if getattr(ActionGRPOMixin.configure_grpo, "_lora_seed", False):
        return
    seed = int(value)
    original = ActionGRPOMixin.configure_grpo

    def configure_grpo(self, grpo_cfg):
        set_global_seed(seed)
        original(self, grpo_cfg)

    configure_grpo._lora_seed = True
    ActionGRPOMixin.configure_grpo = configure_grpo
    logger.info("RNGs are seeded with %d (+rank) before LoRA initialisation", seed)


def install_ref_logprob_cache() -> None:
    """Score each rollout with the frozen reference expert once, not once per inner epoch.

    ``_policy_loss`` runs ``action_chain_logprobs(..., use_ref=True)`` under ``no_grad`` in
    every inner epoch, on the same rollout (``cond_bg``, chain, timesteps, deltas, stoch_idx)
    and with the frozen ``ref_action_expert``, so all epochs get the same result. The first
    result is kept on the rollout dict and returned to the later epochs.
    ``SIMWAM_GRPO_REF_CACHE=0`` turns this off.
    """
    if os.environ.get("SIMWAM_GRPO_REF_CACHE", "1").strip().lower() in _OFF:
        return
    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin
    from simwam.trainer_grpo import SimWAMGRPOTrainer

    if getattr(SimWAMGRPOTrainer._policy_loss, "_ref_cache", False):
        return
    current = {"rollout": None}
    policy_loss = SimWAMGRPOTrainer._policy_loss
    logprobs = ActionGRPOMixin.action_chain_logprobs

    def _policy_loss(self, model, rollout, *args, **kwargs):
        current["rollout"] = rollout
        try:
            return policy_loss(self, model, rollout, *args, **kwargs)
        finally:
            current["rollout"] = None

    def action_chain_logprobs(self, *args, **kwargs):
        rollout = current["rollout"]
        if not kwargs.get("use_ref") or rollout is None:
            return logprobs(self, *args, **kwargs)
        cached = rollout.get("_ref_logprobs")
        if cached is None:
            cached = logprobs(self, *args, **kwargs)
            rollout["_ref_logprobs"] = cached
        return cached

    _policy_loss._ref_cache = True
    SimWAMGRPOTrainer._policy_loss = _policy_loss
    ActionGRPOMixin.action_chain_logprobs = action_chain_logprobs
    logger.info("Reference log-probs are computed once per rollout")


_LEAN_ATTN = {"on": False, "flag": None}


def install_lean_lumen_attention() -> None:
    """Make ``hf_attn_patch``'s masked no-grad attention capturable, and launch it once per call.

    ``hf_attn_patch`` sends masked no-grad calls to Lumen's Triton ``attn_fwd``, whose
    dispatcher copies unit scales from the host; under CUDA graph capture that copy raises and
    ``hf_attn_patch`` silently falls back to torch SDPA, so a captured graph would record a
    different kernel. Both modes call the same Triton forward with resident scales instead.
    * ``1`` (default): inside ``action_velocity`` the masks (the action rows of the MoT mask and
      the context mask) are all True, so ``hf_attn_patch``'s bias is all zeros. The same zero
      bias is kept resident and broadcast over the batch with stride 0, so one launch serves
      all samples instead of one launch per sample, and the bias is not rebuilt per call.
      Whether the masks are all True is checked once per condition, outside capture; other
      calls go to ``hf_attn_patch`` and are never captured.
    * ``loop``: ``hf_attn_patch``'s path op for op (bias built from the mask, one launch per
      sample when the mask differs per sample) for every masked no-grad call.
    Needs ``hf_attn_patch`` in ``LUMEN_ENABLE``. ``SIMWAM_GRPO_LEAN_ATTN=0`` turns this off;
    ``SIMWAM_GRPO_LEAN_ATTN_VERIFY=1`` also runs ``hf_attn_patch`` on every uncaptured call and
    counts outputs that are not bit-identical.
    """
    mode = os.environ.get("SIMWAM_GRPO_LEAN_ATTN", "1").strip().lower()
    if mode in _OFF:
        return
    import torch
    import torch.nn.functional as F
    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin

    if getattr(ActionGRPOMixin.action_velocity, "_lean_attn", False):
        return

    def attention_triton_forward_impl(*args):
        from lumen.kernels.attention.attention_impl import attention_triton_forward_impl as impl

        return impl(*args)
    verify = os.environ.get("SIMWAM_GRPO_LEAN_ATTN_VERIFY", "0") == "1"
    stats = {"lean": 0, "mismatch": 0, "max_diff": 0.0}
    inner = {"sdpa": None, "ones": {}, "zeros": {}}

    def mask_flag(cond, action_len):
        key = f"_lean_attn_all_true_{action_len}"
        flag = cond.get(key)
        if flag is None:
            vs = int(cond["video_seq_len"])
            flag = bool(cond["attention_mask"][vs:vs + action_len, :vs + action_len].all()) and bool(
                cond["context_mask"].all())
            cond[key] = flag
        return flag

    def triton(q, k, v, bias, one):
        return attention_triton_forward_impl(q, k, v, 1.0, one, one, one, 0.0, q.shape[-1] ** (-0.5), False, -1, -1,
                                             bias, None, False, False)[0]

    def replica(query, key, value, attn_mask, q, k, v, one):
        # hf_attn_patch's masked no-grad path, op for op
        b, h, sq, _ = query.shape
        target = (b, h, sq, key.shape[-2])
        bias = torch.zeros(attn_mask.shape, dtype=query.dtype, device=attn_mask.device)
        bias = bias.masked_fill(~attn_mask, torch.finfo(query.dtype).min)
        if tuple(bias.shape) != target:
            while bias.dim() < 4:
                bias = bias.unsqueeze(0)
            bias = bias.expand(target)
        if q.shape[0] > 1 and bias.stride(0) != 0:
            return torch.cat([triton(q[i:i + 1], k[i:i + 1], v[i:i + 1], bias[i:i + 1], one)
                              for i in range(q.shape[0])], dim=0)
        return triton(q, k, v, bias, one)

    def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if ((mode != "loop" and not _LEAN_ATTN["on"]) or attn_mask is None or attn_mask.dtype != torch.bool
                or kwargs or dropout_p or is_causal or scale is not None
                or (torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad))):
            return inner["sdpa"](query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                                 scale=scale, **kwargs)
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        # Lumen's dispatcher builds its unused unit scales with torch.tensor([1.0], device=...), a host
        # copy that cannot be captured; pass resident ones instead (made on the first, eager, call:
        # the graphs warm up eagerly before capturing)
        one = inner["ones"].get(q.device)
        if one is None:
            one = inner["ones"][q.device] = torch.ones(1, device=q.device)
        if mode == "loop":
            out = replica(query, key, value, attn_mask, q, k, v, one).transpose(1, 2)
            stats["lean"] += 1
            if verify and not torch.cuda.is_current_stream_capturing():
                ref = inner["sdpa"](query, key, value, attn_mask=attn_mask)
                if not torch.equal(ref, out):
                    stats["mismatch"] += 1
                    stats["max_diff"] = max(stats["max_diff"], float((ref.float() - out.float()).abs().max()))
            return out
        b, h, sq, sk = query.shape[0], query.shape[1], query.shape[2], key.shape[2]
        zkey = (q.device, q.dtype, sq, sk)
        zero = inner["zeros"].get(zkey)
        if zero is None:
            zero = inner["zeros"][zkey] = torch.zeros((1, 1, sq, sk), dtype=q.dtype, device=q.device)
        # the same zero bias hf_attn_patch builds, broadcast over batch (stride 0) so one launch serves
        # every sample; the kernel ignores the bias batch stride, which is why hf_attn_patch loops
        out = triton(q, k, v, zero.expand(b, h, sq, sk), one).transpose(1, 2)
        stats["lean"] += 1
        if verify and not torch.cuda.is_current_stream_capturing():
            ref = inner["sdpa"](query, key, value, attn_mask=attn_mask)
            if not torch.equal(ref, out):
                stats["mismatch"] += 1
                stats["max_diff"] = max(stats["max_diff"], float((ref.float() - out.float()).abs().max()))
        return out

    original = ActionGRPOMixin.configure_grpo

    def configure_grpo(self, grpo_cfg):
        original(self, grpo_cfg)
        from lumen.ops.attention import hf_patch

        if hf_patch._original_sdpa is not None and inner["sdpa"] is None:
            inner["sdpa"] = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = sdpa
            logger.info("hf_attn_patch: all-True masks inside action_velocity take one bias-free Triton launch "
                        "(verify=%s)", verify)

    velocity = ActionGRPOMixin.action_velocity

    def action_velocity(self, latents_action, timestep_action, cond, use_ref=False):
        if inner["sdpa"] is None:
            return velocity(self, latents_action, timestep_action, cond, use_ref=use_ref)
        previous = _LEAN_ATTN["on"]
        _LEAN_ATTN["on"] = mask_flag(cond, int(latents_action.shape[1]))
        try:
            return velocity(self, latents_action, timestep_action, cond, use_ref=use_ref)
        finally:
            _LEAN_ATTN["on"] = previous

    action_velocity._lean_attn = True
    action_velocity.stats = stats
    _STATS["lean_attn"] = stats
    ActionGRPOMixin.configure_grpo = configure_grpo
    ActionGRPOMixin.action_velocity = action_velocity
    if mode == "loop":
        # the replica is capture-safe for any mask
        _LEAN_ATTN["flag"] = lambda cond, action_len: "loop" if inner["sdpa"] is not None else None
    else:
        _LEAN_ATTN["flag"] = lambda cond, action_len: mask_flag(cond, action_len) if inner["sdpa"] is not None else None


def install_action_velocity_graphs() -> None:
    """Replay ``action_velocity`` from CUDA graphs when no gradient is needed.

    The action expert runs on 8 action tokens per sample, so its forward is a few
    thousand small kernels and is bound by launch overhead. Rollout denoising, the
    rollout's old log-probs and the reference log-probs all call ``action_velocity``
    under ``no_grad`` with fixed shapes. One graph is captured per (expert, shapes);
    a call copies its inputs into the graph's static tensors and replays. The KV cache
    of a condition is copied once per condition, not once per call.
    ``SIMWAM_GRPO_CUDA_GRAPH=0`` turns this off; ``SIMWAM_GRPO_CUDA_GRAPH_VERIFY=1``
    also runs eager on every call and counts outputs that are not bit-identical.
    """
    if os.environ.get("SIMWAM_GRPO_CUDA_GRAPH", "1").strip().lower() in _OFF:
        return
    import torch
    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin

    if getattr(ActionGRPOMixin.action_velocity, "_cuda_graph", False):
        return
    eager = ActionGRPOMixin.action_velocity
    verify = os.environ.get("SIMWAM_GRPO_CUDA_GRAPH_VERIFY", "0") == "1"
    graphs = {}
    stats = {"replay": 0, "eager": 0, "mismatch": 0, "loads": 0}
    serial = [0]

    def cond_tensors(cond):
        kv = [t for layer in cond["video_kv_cache"] for t in (layer["k"], layer["v"])]
        return kv + [cond["context"], cond["context_mask"], cond["attention_mask"]]

    def static_cond(cond, tensors):
        n = len(cond["video_kv_cache"])
        kv = [{"k": tensors[2 * i], "v": tensors[2 * i + 1]} for i in range(n)]
        out = dict(cond)
        out.update(video_kv_cache=kv, context=tensors[-3], context_mask=tensors[-2], attention_mask=tensors[-1])
        return out

    def capture(self, latents, timestep, cond, use_ref, tensors):
        # ActionDiT keeps its RoPE table as a plain CPU attribute and copies it to the
        # device on every forward; a pageable host copy cannot be captured.
        for expert in (self.action_expert, getattr(self, "ref_action_expert", None)):
            freqs = getattr(expert, "freqs", None)
            if torch.is_tensor(freqs) and freqs.device != latents.device:
                expert.freqs = freqs.to(latents.device)
        # Keep the caller's strides: rollout log-probs pass ``chain[:, k]`` slices, and a
        # contiguous copy takes a different GEMM path that differs in the last bf16 bit.
        s_lat = torch.empty_strided(latents.shape, latents.stride(), dtype=latents.dtype, device=latents.device)
        s_lat.copy_(latents)
        s_t = timestep.clone()
        s_tensors = [t.clone() for t in tensors]
        s_cond = static_cond(cond, s_tensors)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                eager(self, s_lat, s_t, s_cond, use_ref=use_ref)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = eager(self, s_lat, s_t, s_cond, use_ref=use_ref)
        return {"graph": graph, "lat": s_lat, "t": s_t, "tensors": s_tensors, "out": out, "loaded": None}

    def action_velocity(self, latents_action, timestep_action, cond, use_ref=False):
        if torch.is_grad_enabled() or not latents_action.is_cuda:
            stats["eager"] += 1
            return eager(self, latents_action, timestep_action, cond, use_ref=use_ref)
        tensors = cond_tensors(cond)
        # the lean attention path bakes "masks are all True" into the graph
        lean = _LEAN_ATTN["flag"](cond, int(latents_action.shape[1])) if _LEAN_ATTN["flag"] else None
        key = (bool(use_ref), tuple(latents_action.shape), tuple(latents_action.stride()), tuple(timestep_action.shape),
               int(cond["video_seq_len"]), tuple((tuple(t.shape), t.dtype) for t in tensors), lean)
        g = graphs.get(key)
        if g is None and lean is False:
            # hf_attn_patch falls back to torch SDPA when a host copy fails under capture, so a
            # masked call outside the lean path would be recorded with a different kernel
            g = graphs[key] = False
        if g is None:
            try:
                g = capture(self, latents_action, timestep_action, cond, use_ref, tensors)
            except Exception as exc:
                logger.warning("CUDA graph capture of action_velocity failed (%s); using eager", exc)
                g = False
            graphs[key] = g
        if g is False:
            stats["eager"] += 1
            return eager(self, latents_action, timestep_action, cond, use_ref=use_ref)
        tag = cond.get("_cuda_graph_tag")
        if tag is None:
            serial[0] += 1
            tag = cond["_cuda_graph_tag"] = serial[0]
        if g["loaded"] != tag:
            torch._foreach_copy_(g["tensors"], tensors)
            g["loaded"] = tag
            stats["loads"] += 1
        g["lat"].copy_(latents_action)
        g["t"].copy_(timestep_action)
        g["graph"].replay()
        out = g["out"].clone()
        stats["replay"] += 1
        if verify:
            ref = eager(self, latents_action, timestep_action, cond, use_ref=use_ref)
            ref2 = eager(self, latents_action, timestep_action, cond, use_ref=use_ref)
            name = f"{'ref' if use_ref else 'policy'}_b{latents_action.shape[0]}"
            by = stats.setdefault("by", {}).setdefault(name, [0, 0, 0.0, 0])
            by[0] += 1
            if not torch.equal(ref, out):
                stats["mismatch"] += 1
                by[1] += 1
                by[2] = max(by[2], float((ref.float() - out.float()).abs().max()))
            if not torch.equal(ref, ref2):
                by[3] += 1
        return out

    action_velocity._cuda_graph = True
    action_velocity.stats = stats
    _STATS["cuda_graph"] = stats
    action_velocity.eager = eager
    ActionGRPOMixin.action_velocity = action_velocity
    logger.info("action_velocity replays CUDA graphs under no_grad (verify=%s)", verify)


_CHUNK = {"sizes": None, "views": None, "ops": frozenset(), "lumen": None}


def _chunk_active(x) -> bool:
    sizes = _CHUNK["sizes"]
    return sizes is not None and x.dim() >= 2 and x.shape[0] == sum(sizes)


def _install_chunked_ops() -> None:
    """Run the GEMMs of a batched call once per original step.

    While ``_CHUNK["sizes"]`` holds the per-step batch sizes, every ``nn.Linear``, LoRA matmul and
    Lumen bf16 GEMM whose input has that total batch runs once per step on its slice and the
    outputs are concatenated. The GEMMs then have the shapes and strides of the unbatched path,
    so they pick the same kernels, and autograd accumulates each LoRA weight gradient from the
    same per-step GEMMs in the same order. ``_CHUNK["views"]`` maps ``id(weight)`` to per-step
    inputs that replace the slices: the action encoder is fed ``chain[:, k]``, a non-contiguous
    view, which ``F.linear`` computes as a matmul plus a separate bias add.
    ``_CHUNK["ops"]`` can add ``rmsnorm`` and ``attn`` (SDPA) to the per-step ops.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from simwam.models.wan22 import wan_video_dit as wvd
    from simwam.models.wan22.lora import LoRALinear

    if getattr(nn.Linear.forward, "_chunked", False):
        return

    def pieces(x, weight=None):
        views = _CHUNK["views"]
        if views is not None and weight is not None and id(weight) in views:
            return views[id(weight)]
        return torch.split(x, _CHUNK["sizes"], dim=0)

    linear_forward = nn.Linear.forward

    def linear(self, x):
        if not _chunk_active(x):
            return linear_forward(self, x)
        return torch.cat([linear_forward(self, xc) for xc in pieces(x, self.weight)], dim=0)

    lora_forward = LoRALinear.forward

    def lora(self, x):
        if not _chunk_active(x):
            return lora_forward(self, x)
        out = self.base(x)
        parts = [(self.lora_dropout(xc) @ self.lora_A.t()) @ self.lora_B.t() for xc in pieces(x)]
        return out + self.scaling * torch.cat(parts, dim=0)

    rms_forward = wvd.RMSNorm.forward

    def rms(self, x):
        if "rmsnorm" not in _CHUNK["ops"] or not _chunk_active(x):
            return rms_forward(self, x)
        return torch.cat([rms_forward(self, xc) for xc in pieces(x)], dim=0)

    sdpa = F.scaled_dot_product_attention

    def chunked_sdpa(query, key, value, attn_mask=None, *args, **kwargs):
        if "attn" not in _CHUNK["ops"] or not _chunk_active(query):
            return sdpa(query, key, value, attn_mask, *args, **kwargs)
        sizes = _CHUNK["sizes"]
        qs, ks, vs = (torch.split(t, sizes, dim=0) for t in (query, key, value))
        if attn_mask is not None and attn_mask.dim() == 4 and attn_mask.shape[0] == query.shape[0]:
            ms = torch.split(attn_mask, sizes, dim=0)
        else:
            ms = [attn_mask] * len(sizes)
        return torch.cat([sdpa(q, k, v, m, *args, **kwargs) for q, k, v, m in zip(qs, ks, vs, ms)], dim=0)

    linear._chunked = True
    nn.Linear.forward = linear
    LoRALinear.forward = lora
    wvd.RMSNorm.forward = rms
    F.scaled_dot_product_attention = chunked_sdpa

    def chunk_lumen_gemm():
        # Lumen is imported by LumenConfig inside configure_grpo; importing it here, before the
        # trainer sets the device, would open a HIP context on GPU 0 in every rank
        import sys

        bg = sys.modules.get("lumen.ops.bf16_gemm")
        if bg is None or getattr(bg.linear, "_chunked", False):
            return
        bg_linear = bg.linear

        def lumen_linear(x, weight, bias=None):
            if not _chunk_active(x):
                return bg_linear(x, weight, bias)
            return torch.cat([bg_linear(xc, weight, bias) for xc in pieces(x, weight)], dim=0)

        lumen_linear._chunked = True
        bg.linear = lumen_linear

    _CHUNK["lumen"] = chunk_lumen_gemm


def install_batched_logprobs() -> None:
    """Score all stochastic steps of a chain with one grad-enabled ``action_velocity`` call.

    ``action_chain_logprobs`` calls the action expert once per stochastic step (3 by default),
    each on the same ``B*G`` samples. With gradients on, the steps are stacked along the batch
    and the expert runs once on ``W*B*G`` samples, so the element-wise kernels of its forward
    and backward launch once instead of ``W`` times. The GEMMs still run once per step (see
    ``_install_chunked_ops``), which keeps the result bit-identical to the unbatched path: a
    single GEMM over ``W*B*G`` rows picks other kernels, and its LoRA weight gradient would be
    one reduction instead of ``W`` GEMMs summed in bf16. The per-step SDE mean, std, clamping and
    reduction are unchanged. No-grad calls (old and reference log-probs) keep the original path,
    which the CUDA graphs already make cheap. Only the rigorous-SDE, unmerged path is batched.
    ``SIMWAM_GRPO_BATCHED_LOGP=0`` turns this off; ``SIMWAM_GRPO_BATCHED_CHUNK`` adds ``rmsnorm``
    and/or ``attn`` to the per-step ops; ``SIMWAM_GRPO_BATCHED_LOGP_VERIFY=1`` also runs the
    unbatched path on every call and compares outputs and LoRA gradients bit for bit.
    """
    if os.environ.get("SIMWAM_GRPO_BATCHED_LOGP", "1").strip().lower() in _OFF:
        return
    import math

    import torch
    from torch.distributions import Normal
    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin

    if getattr(ActionGRPOMixin.action_chain_logprobs, "_batched", False):
        return
    _install_chunked_ops()
    _CHUNK["ops"] = frozenset(
        x.strip() for x in os.environ.get("SIMWAM_GRPO_BATCHED_CHUNK", "").split(",") if x.strip())
    verify = os.environ.get("SIMWAM_GRPO_BATCHED_LOGP_VERIFY", "0") == "1"
    stats = {"batched": 0, "fwd_mismatch": 0, "grad_mismatch": 0, "grad_checked": 0, "max_grad_diff": 0.0}
    original = ActionGRPOMixin.action_chain_logprobs

    def repeat_cond(cond, w):
        key = f"_batched_logp_x{w}"
        rep = cond.get(key)
        if rep is None:
            rep = dict(cond)
            rep.pop("_cuda_graph_tag", None)
            rep["video_kv_cache"] = [
                {"k": torch.cat([layer["k"]] * w, dim=0), "v": torch.cat([layer["v"]] * w, dim=0)}
                for layer in cond["video_kv_cache"]
            ]
            rep["context"] = torch.cat([cond["context"]] * w, dim=0)
            rep["context_mask"] = torch.cat([cond["context_mask"]] * w, dim=0)
            cond[key] = rep
        return rep

    def batched(self, cond, chain, timesteps, deltas, steps, use_ref, return_terms):
        sched = self.infer_action_scheduler
        device = self.device
        dtype = getattr(self, "grpo_action_dtype", self.torch_dtype)
        batch_size = int(chain.shape[0])
        expert = self.ref_action_expert if use_ref else self.action_expert
        x_views = [chain[:, k] for k in steps]
        x_all = torch.cat(x_views, dim=0)
        t_all = torch.cat([
            timesteps[k].to(device=device, dtype=dtype).reshape(1).expand(batch_size) for k in steps
        ])
        _CHUNK["lumen"]()
        _CHUNK["sizes"] = [batch_size] * len(steps)
        _CHUNK["views"] = {id(expert.action_encoder.weight): x_views}
        try:
            v_all = self.action_velocity(x_all, t_all, repeat_cond(cond, len(steps)), use_ref=use_ref)
        finally:
            _CHUNK["sizes"] = None
            _CHUNK["views"] = None
        logps, means, vels, stds = [], [], [], []
        for i, k in enumerate(steps):
            x_k = chain[:, k]
            v = v_all[i * batch_size:(i + 1) * batch_size]
            x_next = chain[:, k + 1]
            sigma_k = float(timesteps[k].item()) / float(sched.num_train_timesteps)
            mean_f32, std = self._sde_mean_std(x_k, v, sigma_k, float(deltas[k].item()))
            mean = mean_f32.to(dtype)
            dist = Normal(mean.float(), torch.tensor(std, device=device, dtype=torch.float32))
            if self.grpo_logprob_clamp_mode == "relative":
                peak = -math.log(std) - 0.5 * math.log(2.0 * math.pi)
                lo, hi = peak - self.grpo_logprob_clamp_width, peak
            else:
                lo, hi = -5.0, 2.0
            logps.append(dist.log_prob(x_next.float()).clamp(min=lo, max=hi).mean(dim=(1, 2)))
            if return_terms:
                means.append(mean_f32)
                vels.append(v.float())
                stds.append(std)
        logp = torch.stack(logps, dim=1)
        if not return_terms:
            return logp
        return logp, {
            "mean": torch.stack(means, dim=1),
            "vel": torch.stack(vels, dim=1),
            "std": torch.tensor(stds, device=device, dtype=torch.float32),
        }

    def probe(out, gen):
        logp, terms = out
        total = (logp * torch.randn(logp.shape, generator=gen, device=logp.device)).sum()
        for key in ("mean", "vel"):
            t = terms[key]
            total = total + (t * torch.randn(t.shape, generator=gen, device=t.device)).sum()
        return total

    def check(self, args, steps, use_ref, new):
        expert = self.ref_action_expert if use_ref else self.action_expert
        params = [p for p in expert.parameters() if p.requires_grad]
        cond, chain, timesteps, deltas = args
        ref = original(self, cond, chain, timesteps, deltas, stoch_idx=steps, use_ref=use_ref, return_terms=True)
        fwd_ok = torch.equal(ref[0], new[0]) and all(torch.equal(ref[1][k], new[1][k]) for k in ref[1])
        stats["fwd_mismatch"] += int(not fwd_ok)
        if params:
            dev = new[0].device
            g_ref = torch.autograd.grad(probe(ref, torch.Generator(device=dev).manual_seed(1234)), params,
                                        allow_unused=True)
            g_new = torch.autograd.grad(probe(new, torch.Generator(device=dev).manual_seed(1234)), params,
                                        allow_unused=True, retain_graph=True)
            bad, worst = 0, 0.0
            for a, b in zip(g_ref, g_new):
                if a is None or b is None:
                    bad += int((a is None) != (b is None))
                    continue
                if not torch.equal(a, b):
                    bad += 1
                    worst = max(worst, float((a.float() - b.float()).abs().max()))
            stats["grad_checked"] += 1
            stats["grad_mismatch"] += int(bad > 0)
            stats["max_grad_diff"] = max(stats["max_grad_diff"], worst)
            logger.info("batched logp verify: forward bit-identical=%s, LoRA grads differing=%d/%d (max %.3g)",
                        fwd_ok, bad, len(params), worst)

    def action_chain_logprobs(self, cond, chain, timesteps, deltas, stoch_idx=None, use_ref=False,
                              return_terms=False):
        steps = self._normalize_stoch_idx(stoch_idx, int(timesteps.shape[0]))
        expert = self.ref_action_expert if use_ref else self.action_expert
        if (not torch.is_grad_enabled() or getattr(self, "grpo_merge_sde_window", False)
                or self.grpo_sde_mode != "rigorous" or len(steps) < 2
                or (self.training and getattr(expert, "use_gradient_checkpointing", False))):
            return original(self, cond, chain, timesteps, deltas, stoch_idx=stoch_idx, use_ref=use_ref,
                            return_terms=return_terms)
        stats["batched"] += 1
        if verify:
            new = batched(self, cond, chain, timesteps, deltas, steps, use_ref, True)
            check(self, (cond, chain, timesteps, deltas), steps, use_ref, new)
            return new if return_terms else new[0]
        return batched(self, cond, chain, timesteps, deltas, steps, use_ref, return_terms)

    action_chain_logprobs._batched = True
    action_chain_logprobs.original = original
    action_chain_logprobs.stats = stats
    _STATS["batched_logp"] = stats
    ActionGRPOMixin.action_chain_logprobs = action_chain_logprobs
    logger.info("action_chain_logprobs runs the grad-enabled steps in one action_velocity call (per-step ops: %s, "
                "verify=%s)", sorted(_CHUNK["ops"] | {"gemm"}), verify)


_PDM_WORKER = {}


def _pdm_worker_init(cls, args, kwargs):
    logging.getLogger().setLevel(logging.WARNING)
    _PDM_WORKER["reward"] = cls(*args, **kwargs)
    _PDM_WORKER["cache"] = {}


def _pdm_worker_score(task):
    token, poses = task
    reward = _PDM_WORKER["reward"]
    if not reward.available(token):
        return ("missing",)
    cache = _PDM_WORKER["cache"]
    metric_cache = cache.get(token)
    if metric_cache is None:
        metric_cache = cache[token] = reward.metric_cache_loader.get_from_token(token)
    reward._true_pdms, reward._submetric_rows = [], []
    try:
        value = reward.score(poses, metric_cache)
    except Exception as exc:  # same fallback as score_batch
        return ("failed", str(exc))
    return ("ok", value, reward._true_pdms[-1], reward._submetric_rows[-1])


def install_parallel_pdm_reward() -> None:
    """Score the trajectories of one ``score_batch`` call in a process pool.

    ``NavSimPDMReward.score_batch`` runs the closed-loop PDM simulation for each of the
    ``B*G`` trajectories one after another on the trainer's CPU thread. The scores are
    independent and deterministic, so they are computed by a persistent pool of spawned
    workers, each holding its own reward object built with the same arguments, and are
    reassembled in order together with the failure counters and per-sample sub-metrics.
    ``SIMWAM_GRPO_PDM_WORKERS`` sets the pool size (default 8, ``0`` turns this off).
    """
    workers = int(os.environ.get("SIMWAM_GRPO_PDM_WORKERS", "8") or 0)
    if workers <= 0:
        return
    import multiprocessing as mp

    import numpy as np
    import torch
    from simwam.datasets.navsim.pdm_reward import NavSimPDMReward, _SUBMETRIC_COLUMNS

    if getattr(NavSimPDMReward.score_batch, "_parallel", False):
        return
    init = NavSimPDMReward.__init__
    serial = NavSimPDMReward.score_batch

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._pool_init_args = (args, kwargs)

    def score_batch(self, abs_poses, tokens, metric_cache_map=None, device=None):
        if metric_cache_map is not None or len(tokens) < 2 or not hasattr(self, "_pool_init_args"):
            return serial(self, abs_poses, tokens, metric_cache_map=metric_cache_map, device=device)
        pool = self.__dict__.get("_pool")
        if pool is None:
            args, kwargs = self._pool_init_args
            pool = self._pool = mp.get_context("spawn").Pool(
                workers, initializer=_pdm_worker_init, initargs=(type(self), args, kwargs))
        if isinstance(abs_poses, torch.Tensor):
            if device is None:
                device = abs_poses.device
            poses_np = abs_poses.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            poses_np = np.asarray(abs_poses, dtype=np.float32)
        if device is None:
            device = torch.device("cpu")
        results = pool.map(_pdm_worker_score, [(t, poses_np[i]) for i, t in enumerate(tokens)])
        self.last_num_total = len(tokens)
        self.last_num_missing = 0
        self.last_num_failed = 0
        self._true_pdms, self._submetric_rows = [], []
        nan_row = {name: float("nan") for name in _SUBMETRIC_COLUMNS}
        rewards = []
        for token, res in zip(tokens, results):
            if res[0] == "ok":
                self._true_pdms.append(res[2])
                self._submetric_rows.append(res[3])
                rewards.append(res[1])
                continue
            if res[0] == "missing":
                self.last_num_missing += 1
            else:
                self.last_num_failed += 1
                logger.warning("PDM scoring failed for token %s: %s", token, res[1])
            rewards.append(self.failure_reward)
            self._submetric_rows.append(dict(nan_row))
        return torch.tensor(rewards, device=device, dtype=torch.float32)

    __init__._parallel = True
    score_batch._parallel = True
    score_batch.serial = serial
    NavSimPDMReward.__init__ = __init__
    NavSimPDMReward.score_batch = score_batch
    logger.info("PDM rewards are scored by %d worker processes", workers)


def _emulate_eager_casts() -> None:
    # inductor keeps fused intermediates in fp32 by default; emulate the bf16 roundings eager does
    import torch._inductor.config as inductor_config

    if hasattr(inductor_config, "emulate_precision_casts"):
        inductor_config.emulate_precision_casts = True


def install_compiled_rmsnorm() -> None:
    """Compile SimWAM's own ``RMSNorm.forward`` into fused kernels (forward and backward).

    Only ``simwam.models.wan22.wan_video_dit.RMSNorm`` is replaced; Lumen's class-name based
    ``lumen_norm`` is not used, because it would also swap the affine-free ``LayerNorm``s.
    ``SIMWAM_GRPO_COMPILE_RMSNORM=0`` turns it off.
    """
    if os.environ.get("SIMWAM_GRPO_COMPILE_RMSNORM", "1").strip().lower() in _OFF:
        return
    import torch
    from simwam.models.wan22.wan_video_dit import RMSNorm

    if getattr(RMSNorm.forward, "_compiled", False):
        return
    _emulate_eager_casts()

    @torch.compile(dynamic=None)
    def rmsnorm(x, weight, eps: float):
        dtype = x.dtype
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(dtype) * weight

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)

    forward._compiled = True
    RMSNorm.forward = forward
    logger.info("SimWAM RMSNorm runs through torch.compile")


def enable_lumen(model, spec: str):
    """Apply ``LumenConfig.enable`` once per requested patch."""
    requested = []
    for key in str(spec).split(","):
        key = key.strip()
        if not key or key.lower() in _OFF:
            continue
        requested.append(key)
    if not requested:
        return model

    from lumen.config import LumenConfig

    applied = []
    for key in requested:
        if key not in _PATCHES:
            logger.warning("Unknown LUMEN_ENABLE flag %r; ignored", key)
            continue
        try:
            _, model = LumenConfig(scaling="none", **{key: True}).enable(model)
            applied.append(key)
        except Exception as exc:
            logger.warning("%s failed (%s); that op stays on torch", key, exc)
    if "lumen_linear" in applied:
        try:
            _patch_lora_gemm()
        except Exception as exc:
            logger.warning("LoRA GEMM patch failed (%s); LoRA A/B stay on torch", exc)
    _log_coverage(model)
    logger.info("Lumen enabled on SimWAM: %s", applied)
    return model
