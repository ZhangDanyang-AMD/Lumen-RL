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
* ``flydsl_conv`` — the VAE's convolutions only, on aiter's FlyDSL implicit-GEMM
  kernel, with Lumen PR #32's ``examples/wan2.2/flydsl_conv.py`` (see
  :func:`install_vae_flydsl_conv`). Applied after the flags above, so with
  ``aiter_conv`` as well the VAE takes FlyDSL and every other conv keeps AITER.

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
# applied by this plugin, not by LumenConfig
_PLUGIN_PATCHES = ("flydsl_conv",)


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


_DROP = {"on": False}


def _masks_all_true(cond) -> bool:
    flag = cond.get("_masks_all_true")
    if flag is None:
        vs, al = int(cond["video_seq_len"]), int(cond["action_horizon"])
        flag = bool(cond["attention_mask"][vs:vs + al, :vs + al].all()) and bool(cond["context_mask"].all())
        cond["_masks_all_true"] = flag
    return flag


def _prefill_masks_all_true(video_context_payload, video_attention_mask) -> bool:
    ctx = (video_context_payload or {}).get("mask")
    return bool(video_attention_mask.all()) and (ctx is None or bool(ctx.all()))


def install_drop_true_masks() -> None:
    """Call SDPA without the attention masks when they are all True.

    In this example every mask the action expert and the video prefill pass to SDPA is all True:
    the GRPO condition is a single frame (so ``first_frame_causal`` masks nothing), the action rows
    of the MoT mask see every video and action token, and the NAVSIM dataset sets the text mask to
    ones (``context_mask = torch.ones_like(...)`` after zeroing the padding). A bool mask still
    sends SDPA to the masked kernels (``hf_attn_patch``'s Lumen Triton for no-grad calls, aotriton
    with a bias for grad-enabled ones); without it torch takes its flash path for both, so
    rollout and training run the same attention forward. The masks are checked once per
    condition (``build_action_condition`` / ``expand_condition``, outside any CUDA graph) and per
    video prefill; a condition with a False entry keeps the masked path. Not bit-identical to
    the masked kernels. ``SIMWAM_GRPO_DROP_MASKS=0`` turns it off.
    With ``SIMWAM_GRPO_AITER_FMHA_GRAD=1`` the grad-enabled calls without a mask go to AITER's
    Triton ``flash_attn_func`` (forward and backward) instead of torch's flash path; no-grad calls
    stay on torch, whose flash forward is AITER's CK kernel already. ``=all`` sends the no-grad
    calls to the same Triton kernel too, so rollout and training attention match again.
    """
    if os.environ.get("SIMWAM_GRPO_DROP_MASKS", "1").strip().lower() in _OFF:
        return
    import torch
    import torch.nn.functional as F
    from simwam.models.wan22.mot import MoT
    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin

    if getattr(ActionGRPOMixin.action_velocity, "_drop_masks", False):
        return
    torch_sdpa = torch._C._nn.scaled_dot_product_attention
    fmha_mode = os.environ.get("SIMWAM_GRPO_AITER_FMHA_GRAD", "0").strip().lower()
    fmha_grad = fmha_mode not in _OFF
    fmha_all = fmha_mode == "all"
    stats = {"dropped": 0, "masked": 0, "aiter_fmha": 0}
    state = {"fa": None}

    def aiter_fmha(query, key, value, args, kwargs):
        # SDPA's positional order after attn_mask: dropout_p, is_causal, scale
        opts = dict(zip(("dropout_p", "is_causal", "scale"), args))
        opts.update(kwargs)
        grad = torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad)
        if (set(opts) - {"dropout_p", "is_causal", "scale"} or opts.get("dropout_p") or opts.get("is_causal")
                or not (grad or fmha_all)):
            return None
        if state["fa"] is None:
            from aiter.ops.triton.mha import flash_attn_func

            state["fa"] = flash_attn_func
        scale = opts.get("scale") or query.shape[-1] ** (-0.5)
        stats["aiter_fmha"] += 1
        out = state["fa"](query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0,
                          softmax_scale=scale, causal=False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.transpose(1, 2)

    def wrap_sdpa(prev):
        def sdpa(query, key, value, attn_mask=None, *args, **kwargs):
            if _DROP["on"] and attn_mask is not None and attn_mask.dtype == torch.bool:
                stats["dropped"] += 1
                if fmha_grad:
                    out = aiter_fmha(query, key, value, args, kwargs)
                    if out is not None:
                        return out
                return torch_sdpa(query, key, value, None, *args, **kwargs)
            if attn_mask is not None:
                stats["masked"] += 1
            return prev(query, key, value, attn_mask, *args, **kwargs)

        sdpa._drop_masks = True
        return sdpa

    original_cfg = ActionGRPOMixin.configure_grpo

    def configure_grpo(self, grpo_cfg):
        # after hf_attn_patch and the lean attention have replaced SDPA, so this runs first
        original_cfg(self, grpo_cfg)
        if not getattr(F.scaled_dot_product_attention, "_drop_masks", False):
            F.scaled_dot_product_attention = wrap_sdpa(F.scaled_dot_product_attention)
            logger.info("all-True attention masks are dropped (torch SDPA without a mask)")

    build = ActionGRPOMixin.build_action_condition
    expand = ActionGRPOMixin.expand_condition

    def build_action_condition(self, *args, **kwargs):
        cond = build(self, *args, **kwargs)
        _masks_all_true(cond)
        return cond

    def expand_condition(self, cond, *args, **kwargs):
        out = expand(self, cond, *args, **kwargs)
        _masks_all_true(out)
        return out

    velocity = ActionGRPOMixin.action_velocity

    def action_velocity(self, latents_action, timestep_action, cond, use_ref=False):
        flag = cond.get("_masks_all_true")
        if flag is None:
            flag = False if torch.cuda.is_current_stream_capturing() else _masks_all_true(cond)
        previous = _DROP["on"]
        _DROP["on"] = flag
        try:
            return velocity(self, latents_action, timestep_action, cond, use_ref=use_ref)
        finally:
            _DROP["on"] = previous

    prefill = MoT.prefill_video_cache

    def prefill_video_cache(self, video_tokens, video_freqs, video_t_mod, video_context_payload,
                            video_attention_mask):
        previous = _DROP["on"]
        if torch.cuda.is_current_stream_capturing():
            # install_prefill_graph checks the masks before capturing
            _DROP["on"] = bool(_DROP.get("prefill_flag"))
        else:
            _DROP["on"] = _prefill_masks_all_true(video_context_payload, video_attention_mask)
        try:
            return prefill(self, video_tokens, video_freqs, video_t_mod, video_context_payload, video_attention_mask)
        finally:
            _DROP["on"] = previous

    action_velocity._drop_masks = True
    _STATS["drop_masks"] = stats
    ActionGRPOMixin.configure_grpo = configure_grpo
    ActionGRPOMixin.build_action_condition = build_action_condition
    ActionGRPOMixin.expand_condition = expand_condition
    ActionGRPOMixin.action_velocity = action_velocity
    MoT.prefill_video_cache = prefill_video_cache


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


def install_numa_affinity() -> None:
    """Pin each rank's threads to the CPUs of its GPU's NUMA node.

    torchrun leaves every rank free to run on any CPU of both sockets, so a rank's host work
    (eager condition building, launches) can land on the socket away from its GPU and run a
    little slower than the others', which then wait for it in the gradient all-reduce. After
    ``configure_grpo`` (the device is set by then) the NUMA node of the current GPU is read from
    sysfs and every thread of the process is restricted to that node's CPUs, intersected with
    the CPUs it is already allowed. Processes started later (PDM pool, data loader workers)
    inherit it. Nothing changes when sysfs gives no node. ``SIMWAM_GRPO_NUMA_AFFINITY=1`` turns
    it on.
    """
    if os.environ.get("SIMWAM_GRPO_NUMA_AFFINITY", "0").strip().lower() in _OFF:
        return
    import glob

    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin

    if getattr(ActionGRPOMixin.configure_grpo, "_numa", False):
        return
    original = ActionGRPOMixin.configure_grpo

    def parse_cpulist(text):
        cpus = set()
        for part in text.strip().split(","):
            if "-" in part:
                a, b = part.split("-")
                cpus.update(range(int(a), int(b) + 1))
            elif part:
                cpus.add(int(part))
        return cpus

    def pin():
        import torch

        bus = int(torch.cuda.get_device_properties(torch.cuda.current_device()).pci_bus_id)
        paths = glob.glob(f"/sys/bus/pci/devices/*:{bus:02x}:00.0/numa_node")
        if len(paths) != 1:
            return None
        node = int(open(paths[0]).read().strip())
        if node < 0:
            return None
        cpus = parse_cpulist(open(f"/sys/devices/system/node/node{node}/cpulist").read()) & os.sched_getaffinity(0)
        if not cpus:
            return None
        for tid in os.listdir("/proc/self/task"):
            try:
                os.sched_setaffinity(int(tid), cpus)
            except OSError:
                pass
        return node, len(cpus)

    def configure_grpo(self, grpo_cfg):
        original(self, grpo_cfg)
        try:
            res = pin()
        except Exception as exc:
            res = None
            logger.warning("NUMA affinity not set (%s)", exc)
        if res is not None:
            logger.info("rank threads pinned to NUMA node %d (%d CPUs)", *res)

    configure_grpo._numa = True
    ActionGRPOMixin.configure_grpo = configure_grpo


def install_prefill_graph() -> None:
    """Replay the video prefill of ``build_action_condition`` from a CUDA graph.

    The video expert is frozen and its prefill runs once per rollout condition with fixed
    shapes (one frame, the same context length): about 30 layers of eager ops whose launch cost
    the GPU does not hide, so the host spends far longer on it than the GPU does. One graph is
    captured per input signature (shapes, strides, dtypes, and whether the masks are all True);
    a call copies its inputs into the graph's static tensors, replays, and returns a copy of the
    per-layer K/V. Grad-enabled calls, inputs not all on the GPU and failed captures run eagerly.
    ``SIMWAM_GRPO_PREFILL_GRAPH=1`` turns it on.
    """
    if os.environ.get("SIMWAM_GRPO_PREFILL_GRAPH", "0").strip().lower() in _OFF:
        return
    import torch
    from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
    from simwam.models.wan22.mot import MoT

    if getattr(MoT.prefill_video_cache, "_graph", False):
        return
    eager = MoT.prefill_video_cache
    graphs = {}
    stats = {"replay": 0, "eager": 0, "captured": 0}

    def capture(self, leaves, spec, flag):
        static, static_leaves = [], []
        for x in leaves:
            if torch.is_tensor(x):
                s = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
                s.copy_(x)
                static.append(s)
                x = s
            static_leaves.append(x)
        args = tree_unflatten(static_leaves, spec)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                eager(self, *args)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        _DROP["prefill_flag"] = flag
        with torch.cuda.graph(graph):
            out = eager(self, *args)
        return {"graph": graph, "static": static, "out": out}

    def prefill_video_cache(self, video_tokens, video_freqs, video_t_mod, video_context_payload,
                            video_attention_mask):
        args = (video_tokens, video_freqs, video_t_mod, video_context_payload, video_attention_mask)
        leaves, spec = tree_flatten(args)
        tensors = [x for x in leaves if torch.is_tensor(x)]
        if (torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing() or not tensors
                or not all(t.is_cuda for t in tensors)):
            stats["eager"] += 1
            return eager(self, *args)
        flag = _prefill_masks_all_true(video_context_payload, video_attention_mask)
        key = (id(self), str(spec), flag, tuple((tuple(x.shape), tuple(x.stride()), x.dtype) if torch.is_tensor(x)
                                               else repr(x) for x in leaves))
        g = graphs.get(key)
        if g is None:
            try:
                g = capture(self, leaves, spec, flag)
                stats["captured"] += 1
            except Exception as exc:
                logger.warning("CUDA graph capture of the video prefill failed (%s); using eager", exc)
                g = False
            graphs[key] = g
        if g is False:
            stats["eager"] += 1
            return eager(self, *args)
        for s, t in zip(g["static"], tensors):
            s.copy_(t)
        g["graph"].replay()
        stats["replay"] += 1
        return tree_map(lambda t: t.clone() if torch.is_tensor(t) else t, g["out"])

    prefill_video_cache._graph = True
    _STATS["prefill_graph"] = stats
    MoT.prefill_video_cache = prefill_video_cache
    logger.info("the video prefill replays a CUDA graph")


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
    Not bit-identical, on by default: ``SIMWAM_GRPO_BATCHED_GEMM`` runs each GEMM once over all
    steps (no per-step split); ``SIMWAM_GRPO_BATCHED_LOGP_NOGRAD`` batches the no-grad calls (the
    rollout's old and reference log-probs) the same way, so they take the training path and the
    rollout's log-probs match the first inner update's. Set both to ``0`` for the per-step path.
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
    one_gemm = os.environ.get("SIMWAM_GRPO_BATCHED_GEMM", "1").strip().lower() not in _OFF
    nograd = os.environ.get("SIMWAM_GRPO_BATCHED_LOGP_NOGRAD", "1").strip().lower() not in _OFF
    stats = {"batched": 0, "batched_nograd": 0, "fwd_mismatch": 0, "grad_mismatch": 0, "grad_checked": 0,
             "max_grad_diff": 0.0}
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
        if not one_gemm:
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
        grad = torch.is_grad_enabled()
        if ((not grad and not nograd) or getattr(self, "grpo_merge_sde_window", False)
                or self.grpo_sde_mode != "rigorous" or len(steps) < 2
                or (self.training and getattr(expert, "use_gradient_checkpointing", False))):
            return original(self, cond, chain, timesteps, deltas, stoch_idx=stoch_idx, use_ref=use_ref,
                            return_terms=return_terms)
        stats["batched" if grad else "batched_nograd"] += 1
        if not grad:
            return batched(self, cond, chain, timesteps, deltas, steps, use_ref, return_terms)
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
                "verify=%s, no-grad calls batched=%s)",
                sorted(_CHUNK["ops"] | (set() if one_gemm else {"gemm"})), verify, nograd)


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


def install_train_graphs() -> None:
    """Replay the policy's grad-enabled ``action_velocity`` forward and backward from CUDA graphs.

    With ``install_batched_logprobs`` each inner epoch makes one grad-enabled ``action_velocity``
    call followed by one backward. A forward graph and a backward graph are captured per input
    shape/stride and wrapped in an autograd Function that takes the trainable (LoRA) parameters
    as explicit inputs, so their gradients leave the backward graph as ordinary outputs and are
    accumulated, reduced and stepped outside the graph exactly as in eager.

    Unlike ``torch.cuda.make_graphed_callables``:
    * the parameter gradients and the output are cloned out of the graph's memory pool, so
      nothing outside the graph holds memory that the next replay overwrites;
    * the captured output is detached afterwards, which frees the capture-time autograd graph.
      That graph otherwise keeps the parameters' AccumulateGrad nodes alive with the capture
      stream recorded on them;
    * warmup uses ``autograd.grad`` and never touches ``.grad``; each shape gets its own pool.

    The batched call still runs its GEMMs once per step (``_install_chunked_ops``); the per-step
    views of the caller's chain that the action encoder reads are part of the graph's static
    inputs, and the lean-attention mask check runs before capture, as in the no_grad graphs.

    Only without DeepSpeed (``install_manual_data_parallel``): with DeepSpeed ZeRO-1 the backward
    capture segfaults in ``capture_end`` (DeepSpeed 0.19.8, ROCm 7.14), so nothing is installed.
    Install after ``install_action_velocity_graphs`` (no_grad calls go to that path).
    ``SIMWAM_GRPO_TRAIN_GRAPH=0`` turns this off; ``SIMWAM_GRPO_TRAIN_GRAPH_VERIFY=1`` also runs
    eager forward and backward before each replay and counts outputs and LoRA gradients that
    are not bit-identical.
    """
    if os.environ.get("SIMWAM_GRPO_TRAIN_GRAPH", "1").strip().lower() in _OFF:
        return
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").strip().lower() not in _OFF:
        logger.info("Training CUDA graphs need SIMWAM_GRPO_MANUAL_DP=1 (not DeepSpeed); not installed")
        return
    import torch
    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin

    if getattr(ActionGRPOMixin.action_velocity, "_train_graph", False):
        return
    inner = ActionGRPOMixin.action_velocity
    eager = getattr(inner, "eager", None) or inner
    verify = os.environ.get("SIMWAM_GRPO_TRAIN_GRAPH_VERIFY", "0") == "1"
    warmup = int(os.environ.get("SIMWAM_GRPO_TRAIN_GRAPH_WARMUP", "3"))
    graphs = {}
    stats = {"graphed": 0, "eager": 0, "captures": 0, "out_mismatch": 0, "grad_mismatch": 0, "checks": 0,
             "out_maxabs": 0.0, "grad_maxabs": 0.0}
    _COND = ("video_kv_cache", "context", "context_mask", "attention_mask")

    def flat(cond):
        kv = [t for layer in cond["video_kv_cache"] for t in (layer["k"], layer["v"])]
        return kv + [cond["context"], cond["context_mask"], cond["attention_mask"]]

    def unflat(rest, layers, tensors):
        cond = dict(rest)
        cond.update(
            video_kv_cache=[{"k": tensors[2 * i], "v": tensors[2 * i + 1]} for i in range(layers)],
            context=tensors[2 * layers], context_mask=tensors[2 * layers + 1],
            attention_mask=tensors[2 * layers + 2],
        )
        return cond

    class _Replay(torch.autograd.Function):
        @staticmethod
        def forward(ctx, g, *inputs):
            for static, value in zip(g["inputs"], inputs[:g["n_in"]]):
                if static.data_ptr() != value.data_ptr():
                    static.copy_(value)
            g["fwd"].replay()
            ctx.g = g
            return g["out"].clone()

        @staticmethod
        def backward(ctx, grad_out):
            g = ctx.g
            g["grad_out"].copy_(grad_out)
            g["bwd"].replay()
            grads = tuple(None if x is None else x.clone() for x in g["grads"])
            return (None,) + (None,) * g["n_in"] + grads

    def chunk_views():
        # install_batched_logprobs feeds some weights per-step views of the caller's chain instead
        # of slices of the batched input; the graph must read them from a static copy of that base
        views = _CHUNK["views"]
        if not views:
            return [], (), None
        bases, geometry = [], []
        for wid, vlist in sorted(views.items()):
            base = vlist[0]._base if vlist[0]._base is not None else vlist[0]
            if any((v._base if v._base is not None else v) is not base for v in vlist):
                raise ValueError("per-step views do not share one base")
            bases.append(base)
            geometry.append((wid, tuple(tuple((tuple(v.shape), tuple(v.stride()), v.storage_offset() - base.storage_offset()))
                                        for v in vlist)))
        return bases, tuple(geometry), tuple(_CHUNK["sizes"] or ())

    def capture(self, args, cond, params, n_cond, geometry):
        for expert in (self.action_expert, getattr(self, "ref_action_expert", None)):
            freqs = getattr(expert, "freqs", None)
            if torch.is_tensor(freqs) and freqs.device != args[0].device:
                expert.freqs = freqs.to(args[0].device)
        layers = len(cond["video_kv_cache"])
        rest = {k: v for k, v in cond.items() if k not in _COND and k != "_cuda_graph_tag"}
        inputs = [torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device) for t in args]
        torch._foreach_copy_(inputs, list(args))
        static_views = {}
        for base, (wid, geo) in zip(inputs[2 + n_cond:], geometry):
            static_views[wid] = [base.as_strided(shape, stride, base.storage_offset() + offset)
                                 for shape, stride, offset in geo]

        def run():
            saved = _CHUNK["views"]
            if static_views:
                _CHUNK["views"] = static_views
            try:
                return eager(self, inputs[0], inputs[1], unflat(rest, layers, inputs[2:2 + n_cond]), use_ref=False)
            finally:
                _CHUNK["views"] = saved

        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                out = run()
                torch.autograd.grad(out, params, grad_outputs=torch.zeros_like(out), allow_unused=True)
                del out
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        # Capture on the warmup stream: hipBLASLt fails ("operation not permitted when stream is
        # capturing") the first time it sees a new stream inside a capture.
        fwd, bwd = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(fwd, stream=side):
            out = run()
        grad_out = torch.zeros_like(out)
        with torch.cuda.graph(bwd, pool=fwd.pool(), stream=side):
            grads = torch.autograd.grad(out, params, grad_outputs=grad_out, allow_unused=True)
        torch.cuda.synchronize()
        stats["captures"] += 1
        return {"fwd": fwd, "bwd": bwd, "inputs": inputs, "n_in": len(inputs), "out": out.detach(),
                "grad_out": grad_out, "grads": grads}

    def action_velocity(self, latents_action, timestep_action, cond, use_ref=False):
        if use_ref or not torch.is_grad_enabled() or not latents_action.is_cuda:
            return inner(self, latents_action, timestep_action, cond, use_ref=use_ref)
        params = [p for p in self.action_expert.parameters() if p.requires_grad]
        # the lean attention path checks the masks once per condition (a host sync); do it here,
        # outside capture, so the cached flag travels with the condition into the graph
        lean = _LEAN_ATTN["flag"](cond, int(latents_action.shape[1])) if _LEAN_ATTN["flag"] else None
        cond_tensors = flat(cond)
        bases, geometry, sizes = chunk_views()
        args = (latents_action, timestep_action, *cond_tensors, *bases)
        key = (tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in args), len(params),
               tuple(k for k in cond if k not in _COND), int(cond.get("video_seq_len", -1)), lean, geometry, sizes)
        g = graphs.get(key)
        if g is None and lean is False:
            g = graphs[key] = False
        if g is None:
            try:
                g = capture(self, args, cond, params, len(cond_tensors), geometry)
            except Exception as exc:
                logger.warning("CUDA graph capture of the training action_velocity failed (%s); using eager", exc)
                g = False
            graphs[key] = g
        if g is False:
            stats["eager"] += 1
            return eager(self, latents_action, timestep_action, cond, use_ref=False)
        if verify:
            ref = eager(self, latents_action, timestep_action, cond, use_ref=False)
            # a private generator, so verification does not advance the default CUDA RNG the rollout samples from
            w = torch.randn(ref.shape, dtype=ref.dtype, device=ref.device,
                            generator=torch.Generator(device=ref.device).manual_seed(1234))
            g_ref = torch.autograd.grad(ref, params, grad_outputs=w, allow_unused=True)
            out = _Replay.apply(g, *args, *params)
            g_out = torch.autograd.grad(out, params, grad_outputs=w, allow_unused=True)
            stats["checks"] += 1
            if not torch.equal(ref, out):
                stats["out_mismatch"] += 1
                stats["out_maxabs"] = max(stats["out_maxabs"], float((ref.float() - out.float()).abs().max()))
            for a, b in zip(g_ref, g_out):
                if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                    stats["grad_mismatch"] += 1
                    if a is not None and b is not None:
                        stats["grad_maxabs"] = max(stats["grad_maxabs"], float((a.float() - b.float()).abs().max()))
                    break
        stats["graphed"] += 1
        return _Replay.apply(g, *args, *params)

    action_velocity._train_graph = True
    action_velocity._cuda_graph = getattr(inner, "_cuda_graph", False)
    action_velocity.stats = getattr(inner, "stats", stats)
    action_velocity.train_stats = stats
    action_velocity.eager = eager
    ActionGRPOMixin.action_velocity = action_velocity
    logger.info("grad-enabled action_velocity replays CUDA graphs (verify=%s)", verify)


class _MasterAdamW:
    """fp32 master copy of the trainable parameters, averaged gradients and AdamW.

    Mirrors what DeepSpeed ZeRO-1 (0.19.8, bf16, ``overlap_comm`` and ``contiguous_gradients``
    off) does for this model, without partitioning: the bf16 gradients are flattened into one
    fp32 buffer, all-reduced and averaged, the global norm is taken (and clipped only when the
    trainer asked DeepSpeed to clip), AdamW steps the flat fp32 master and the result is
    rounded back into the bf16 parameters in place.

    With ``round_avg`` (the default) the averaged gradient is rounded to the parameter dtype
    before the step and the norm is accumulated the way ZeRO-1 does it (fp64 norm per gradient
    slice of each rank's partition, fp32 sum per partition, summed over partitions), so the
    update and the reported norm match DeepSpeed bit for bit. DeepSpeed's ``allreduce_bucket``
    writes the fp32-reduced gradient back into the bf16 gradient tensors.
    ``SIMWAM_GRPO_MANUAL_DP_GRAD_AVG=fp32`` keeps the averaged gradient in fp32 instead
    (slightly more precise, not bit-identical to DeepSpeed).
    """

    def __init__(self, shell, clip, round_avg=True):
        import torch
        import torch.distributed as dist

        if len(shell.param_groups) != 1:
            raise ValueError("manual data parallel expects one optimizer param group")
        self.shell = shell
        self.clip = clip
        self.round_avg = round_avg
        self.params = list(shell.param_groups[0]["params"])
        self.world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if self.world > 1:
            # DeepSpeed broadcasts the model from rank 0 at init; LoRA A is randomly initialised per rank
            for p in self.params:
                dist.broadcast(p.data, 0)
        flat = torch.cat([p.detach().reshape(-1).float() for p in self.params])
        self.master = torch.nn.Parameter(flat, requires_grad=False)
        self.master.grad = torch.zeros_like(flat)
        self.low = torch.empty_like(flat, dtype=self.params[0].dtype)
        sizes = [p.numel() for p in self.params]
        self.low_views = [v.view_as(p) for v, p in zip(self.low.split(sizes), self.params)]
        # (start, stop) of every gradient slice inside each rank's ZeRO-1 partition, in flat order
        total, part = flat.numel(), flat.numel() // self.world
        self.norm_slices = []
        for r in range(self.world):
            lo, hi = r * part, (total if r == self.world - 1 else (r + 1) * part)
            off, slices = 0, []
            for n in sizes:
                a, b = off, off + n
                off = b
                if a < hi and b > lo:
                    slices.append((max(a, lo), min(b, hi)))
            self.norm_slices.append(slices)
        hp = shell.defaults
        self.adam = torch.optim.AdamW([self.master], lr=hp["lr"], betas=hp["betas"], eps=hp["eps"],
                                      weight_decay=hp["weight_decay"], amsgrad=hp.get("amsgrad", False))
        self.loaded = False

    def reset_from_params(self):
        """Restart the master from the current bf16 parameters (e.g. after a weights-only load)."""
        import torch

        self.master.data.copy_(torch.cat([p.detach().reshape(-1).float() for p in self.params]))

    def step(self, grads=None, write_back=True):
        import torch
        import torch.distributed as dist

        if grads is None:
            grads = [p.grad for p in self.params]
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.params)]
        torch.cat([g.reshape(-1) for g in grads], out=self.low)
        buf = self.master.grad
        buf.copy_(self.low)
        if self.world > 1:
            dist.all_reduce(buf)
            buf.div_(self.world)
        if self.round_avg:
            self.low.copy_(buf)
            buf.copy_(self.low)
            norm = self._zero1_norm(buf)
        else:
            norm = torch.linalg.vector_norm(buf, 2.0)
        if self.clip is not None:
            buf.mul_((self.clip / (norm + 1e-6)).clamp(max=1.0))
        for group in self.adam.param_groups:
            group["lr"] = self.shell.param_groups[0]["lr"]
        self.adam.step()
        self.low.copy_(self.master.detach())
        if write_back:
            torch._foreach_copy_([p.data for p in self.params], self.low_views)
        return norm

    def _zero1_norm(self, buf):
        import torch

        flat64 = buf.double()
        total = None
        for slices in self.norm_slices:
            norms = torch._foreach_norm([flat64[a:b] for a, b in slices])
            s = torch.stack(norms).pow(2).sum().float()
            total = s if total is None else total + s
        return torch.linalg.vector_norm(total.pow(0.5).reshape(1), 2.0)

    def state_dict(self):
        return {"master": self.master.detach().clone(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state):
        self.master.data.copy_(state["master"])
        self.adam.load_state_dict(state["adam"])
        self.low.copy_(self.master.detach())
        import torch

        torch._foreach_copy_([p.data for p in self.params], self.low_views)
        self.loaded = True


def install_manual_data_parallel() -> None:
    """Train on several GPUs without DeepSpeed: plain gradient all-reduce and an fp32-master AdamW.

    The action expert has 15.7 M trainable (LoRA) parameters, so ZeRO-1 partitioning saves
    about 94 MB per GPU, while its engine sits in every backward and step. This keeps
    Accelerate (data sharding, scheduler, checkpoint files) but in its non-DeepSpeed mode:
    * ``SimWAMGRPOTrainer.__init__`` refuses multi-GPU without DeepSpeed; it is rebuilt from
      its source with only that check removed (installation is skipped if the check is not
      found exactly once);
    * the model is not wrapped in DDP and Accelerate's autocast is a no-op: the weights are
      already bf16, as under DeepSpeed;
    * ``_optimizer_step`` runs ``_MasterAdamW`` (registered for checkpointing) and steps the
      scheduler; gradients are clipped only if ``deepspeed_clip_from_max_grad_norm`` is set,
      as DeepSpeed does.
    On by default: with it the grad-enabled forward and backward replay from CUDA graphs
    (``install_train_graphs``), and a run stays bit-identical to DeepSpeed ZeRO-1.
    ``SIMWAM_GRPO_MANUAL_DP=0`` turns it off (DeepSpeed). Call before the trainer is built.
    """
    if os.environ.get("SIMWAM_GRPO_MANUAL_DP", "1").strip().lower() in _OFF:
        return
    import __future__
    import ast
    import contextlib
    import inspect
    import textwrap

    import accelerate
    import simwam.trainer_grpo as tg

    T = tg.SimWAMGRPOTrainer
    if getattr(T.__init__, "_manual_dp", False):
        return
    original = T.__init__
    if "__class__" in original.__code__.co_freevars:
        logger.warning("SimWAMGRPOTrainer.__init__ uses super(); manual data parallel not installed")
        return
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    body = [n for n in fn.body if not (isinstance(n, ast.If) and "DEEPSPEED" in ast.unparse(n.test))]
    if len(body) != len(fn.body) - 1:
        logger.warning("DeepSpeed check not found in SimWAMGRPOTrainer.__init__; manual data parallel not installed")
        return
    fn.body = body
    ast.increment_lineno(tree, original.__code__.co_firstlineno - 1)
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), "exec", flags=__future__.annotations.compiler_flag,
                 dont_inherit=True), tg.__dict__, namespace)
    rebuilt = namespace["__init__"]
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"

    prepare_model = accelerate.Accelerator.prepare_model

    def _prepare_model(self, model, device_placement=None, evaluation_mode=False):
        native = self.native_amp
        self.native_amp = False
        try:
            return prepare_model(self, model, device_placement=device_placement, evaluation_mode=True)
        finally:
            self.native_amp = native

    accelerate.Accelerator.prepare_model = _prepare_model
    accelerate.Accelerator.autocast = lambda self, *args, **kwargs: contextlib.nullcontext()

    def _register_master(self):
        clip = self.max_grad_norm if self.cfg.get("deepspeed_clip_from_max_grad_norm", False) else None
        round_avg = os.environ.get("SIMWAM_GRPO_MANUAL_DP_GRAD_AVG", "bf16").strip().lower() != "fp32"
        self._master_adamw = _MasterAdamW(self.optimizer.optimizer, clip, round_avg=round_avg)
        self.accelerator.register_for_checkpointing(self._master_adamw)

    resume = T._resume_or_load_checkpoint

    def _resume_or_load_checkpoint(self):
        # Accelerate's load_state (called from here, inside __init__) restores registered objects
        # by index, so the master must be registered before it runs
        _register_master(self)
        resume(self)
        if not self._master_adamw.loaded:
            self._master_adamw.reset_from_params()

    def __init__(self, *args, **kwargs):
        rebuilt(self, *args, **kwargs)
        if getattr(self, "_master_adamw", None) is None:
            _register_master(self)
        logger.info("Manual data parallel: world=%d, %d trainable tensors, clip=%s, resumed=%s",
                    self._master_adamw.world, len(self._master_adamw.params), self._master_adamw.clip,
                    self._master_adamw.loaded)

    def _optimizer_step(self):
        grad_norm = self._master_adamw.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    __init__._manual_dp = True
    T.__init__ = __init__
    T._resume_or_load_checkpoint = _resume_or_load_checkpoint
    T._optimizer_step = _optimizer_step
    logger.info("SimWAM GRPO trains without DeepSpeed (manual gradient all-reduce, fp32-master AdamW)")


def _emulate_eager_casts() -> None:
    # inductor keeps fused intermediates in fp32 by default; emulate the bf16 roundings eager does
    import torch._inductor.config as inductor_config

    if hasattr(inductor_config, "emulate_precision_casts"):
        inductor_config.emulate_precision_casts = True


def install_aiter_rmsnorm() -> None:
    """SimWAM ``RMSNorm`` on AITER's CK ``rms_norm``: one fused launch instead of the eager chain.

    The kernel normalises and multiplies by the weight in one pass, without the bf16 rounding
    eager applies before the weight multiply, so it is not bit-identical. It has no backward:
    grad-enabled calls go through an autograd Function whose backward computes the input
    gradient from the RMSNorm formula in fp32 (the norm weights are frozen here; a weight
    gradient is computed too if one is needed). Only bf16 with the widths checked below.
    ``SIMWAM_GRPO_AITER_RMSNORM=0`` turns it off. ``SIMWAM_GRPO_TRITON_RMSNORM_BWD=1`` computes the
    input gradient in one Triton kernel (``fused_kernels.rmsnorm_backward``) instead of a chain
    of fp32 element-wise and reduction kernels.
    """
    if os.environ.get("SIMWAM_GRPO_AITER_RMSNORM", "1").strip().lower() in _OFF:
        return
    triton_bwd = os.environ.get("SIMWAM_GRPO_TRITON_RMSNORM_BWD", "0").strip().lower() not in _OFF
    import torch
    from simwam.models.wan22 import wan_video_dit as wvd

    if getattr(wvd.RMSNorm.forward, "_aiter", False):
        return
    previous = wvd.RMSNorm.forward
    widths = {3072}
    state = {"rms": None}

    class _RMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            return state["rms"](x, weight, eps)

        @staticmethod
        def backward(ctx, grad):
            x, weight = ctx.saved_tensors
            if triton_bwd and not ctx.needs_input_grad[1]:
                from examples.Embodied_SimWAM_NAVSIM.fused_kernels import rmsnorm_backward

                return rmsnorm_backward(x, grad, weight, ctx.eps), None, None
            xf = x.float()
            gw = grad.float() * weight.float()
            r = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + ctx.eps)
            dx = r * gw - xf * r.pow(3) * (gw * xf).mean(dim=-1, keepdim=True)
            dw = None
            if ctx.needs_input_grad[1]:
                dw = (grad.float() * xf * r).reshape(-1, x.shape[-1]).sum(dim=0).to(weight.dtype)
            return dx.to(x.dtype), dw, None

    def forward(self, x):
        if x.dtype != torch.bfloat16 or self.weight.dtype != torch.bfloat16 or x.shape[-1] not in widths:
            return previous(self, x)
        if state["rms"] is None:
            # imported on first use, after the trainer has set the device
            from aiter.ops.rmsnorm import rms_norm

            state["rms"] = rms_norm
        x = x.contiguous()
        if torch.is_grad_enabled() and (x.requires_grad or self.weight.requires_grad):
            return _RMSNorm.apply(x, self.weight, float(self.eps))
        return state["rms"](x, self.weight, float(self.eps))

    forward._aiter = True
    wvd.RMSNorm.forward = forward
    logger.info("SimWAM RMSNorm runs on AITER's CK rms_norm")


def install_triton_rope() -> None:
    """SimWAM's ``rope_apply`` as one Triton kernel in fp32, forward and backward.

    SimWAM rotates q and k in float64 complex arithmetic: a cast to float64, a complex multiply
    and a cast back, each a separate kernel, and the same again in the backward. The kernel in
    ``fused_kernels.rope`` reads the bf16 input and the (cos, sin) table, rotates in fp32 and
    writes bf16 in one launch; the backward rotates the gradient by the conjugate. Not
    bit-identical to the float64 path. Calls that are not bf16 on the GPU with a complex
    ``[S or 1, 1, D/2]`` table keep SimWAM's function. ``SIMWAM_GRPO_TRITON_ROPE=1`` turns it on.
    """
    if os.environ.get("SIMWAM_GRPO_TRITON_ROPE", "0").strip().lower() in _OFF:
        return
    import torch
    import simwam.models.wan22.mot as mot
    import simwam.models.wan22.wan_video_dit as wvd

    if getattr(wvd.rope_apply, "_triton", False):
        return
    original = wvd.rope_apply
    stats = {"triton": 0, "fallback": 0}

    class _Rope(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, freqs_real):
            from examples.Embodied_SimWAM_NAVSIM.fused_kernels import rope

            ctx.save_for_backward(freqs_real)
            return rope(x, freqs_real)

        @staticmethod
        def backward(ctx, grad):
            from examples.Embodied_SimWAM_NAVSIM.fused_kernels import rope

            (freqs_real,) = ctx.saved_tensors
            return rope(grad, freqs_real, inverse=True), None

    def rope_apply(x, freqs, num_heads):
        if (x.dtype != torch.bfloat16 or not x.is_cuda or x.dim() != 3 or not torch.is_complex(freqs)
                or not freqs.is_cuda or freqs.dim() != 3 or freqs.shape[1] != 1
                or freqs.shape[0] not in (1, x.shape[1]) or x.shape[-1] != num_heads * 2 * freqs.shape[-1]):
            stats["fallback"] += 1
            return original(x, freqs, num_heads)
        fr = torch.view_as_real(freqs)
        if fr.shape[0] == 1 and x.shape[1] > 1:
            fr = fr.expand(x.shape[1], *fr.shape[1:])
        stats["triton"] += 1
        return _Rope.apply(x, fr)

    rope_apply._triton = True
    _STATS["triton_rope"] = stats
    wvd.rope_apply = rope_apply
    mot.rope_apply = rope_apply
    logger.info("rope_apply runs as one Triton kernel (fp32)")


def install_compiled_rmsnorm() -> None:
    """Compile SimWAM's own ``RMSNorm.forward`` into fused kernels (forward and backward).

    Only ``simwam.models.wan22.wan_video_dit.RMSNorm`` is replaced; Lumen's class-name based
    ``lumen_norm`` is not used, because it would also swap the affine-free ``LayerNorm``s.
    The fused kernels are not bit-identical to eager (log-probs differ by ~0.02-0.03%), so this
    is off by default; ``SIMWAM_GRPO_COMPILE_RMSNORM=1`` turns it on. With ``rmsnorm`` in
    ``SIMWAM_GRPO_BATCHED_CHUNK`` the compiled function runs once per original step.
    """
    if os.environ.get("SIMWAM_GRPO_COMPILE_RMSNORM", "0").strip().lower() in _OFF:
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
        if "rmsnorm" in _CHUNK["ops"] and _chunk_active(x):
            return torch.cat([rmsnorm(xc, self.weight, self.eps) for xc in torch.split(x, _CHUNK["sizes"], dim=0)], dim=0)
        return rmsnorm(x, self.weight, self.eps)

    forward._compiled = True
    RMSNorm.forward = forward
    logger.info("SimWAM RMSNorm runs through torch.compile")


def install_fused_lora_add() -> None:
    """``LoRALinear.forward`` adds the scaled adapter output with one ``torch.add(alpha=)``.

    SimWAM computes ``out + self.scaling * lora`` as a multiply and an add. With the
    default ``alpha / r = 2.0`` the scale is a power of two, so ``torch.add(out, lora,
    alpha=scaling)`` rounds the same way. Inside a batched log-prob call the LoRA matmuls
    still run once per original step (as in ``install_batched_logprobs``), so the weight
    gradients come from the same per-step GEMMs. ``SIMWAM_GRPO_FUSED_LORA=0`` turns it off.
    """
    if os.environ.get("SIMWAM_GRPO_FUSED_LORA", "1").strip().lower() in _OFF:
        return
    import torch
    from simwam.models.wan22.lora import LoRALinear

    if getattr(LoRALinear.forward, "_fused_add", False):
        return

    def forward(self, x):
        out = self.base(x)
        if _chunk_active(x):
            lora = torch.cat([(self.lora_dropout(xc) @ self.lora_A.t()) @ self.lora_B.t()
                              for xc in torch.split(x, _CHUNK["sizes"], dim=0)], dim=0)
        else:
            lora = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return torch.add(out, lora, alpha=self.scaling)

    forward._fused_add = True
    LoRALinear.forward = forward
    logger.info("LoRALinear adds the adapter with torch.add(alpha=scaling)")


def install_fused_modulate() -> None:
    """``modulate`` and ``GateModule`` with ``torch.addcmul``: one kernel fewer per call.

    SimWAM computes ``x * (1 + scale) + shift`` (three kernels) and ``x + gate * residual``
    (two). ``torch.addcmul(shift, x, 1 + scale)`` and ``torch.addcmul(x, gate, residual)`` do the
    multiply-add in one pass and round once, so they are not bit-identical; rollout and training
    use the same op, so they still agree with each other. ``SIMWAM_GRPO_FUSED_MODULATE=1`` turns
    it on.
    """
    if os.environ.get("SIMWAM_GRPO_FUSED_MODULATE", "0").strip().lower() in _OFF:
        return
    import torch
    import simwam.models.wan22.mot as mot
    import simwam.models.wan22.wan_video_dit as wvd

    if getattr(wvd.modulate, "_addcmul", False):
        return

    def modulate(x, shift, scale):
        return torch.addcmul(shift, x, 1 + scale)

    def gate_forward(self, x, gate, residual):
        return torch.addcmul(x, gate, residual)

    modulate._addcmul = True
    wvd.modulate = modulate
    mot.modulate = modulate
    wvd.GateModule.forward = gate_forward
    logger.info("modulate and GateModule use torch.addcmul")


def install_overlap_reward_logprobs() -> None:
    """Score PDM rewards in the worker pool while the rollout's log-probs run on the GPU.

    ``_rollout`` scores the trajectories and only then computes the old log-probs, which do
    not depend on the rewards. With the pool from ``install_parallel_pdm_reward`` the
    rewards are submitted asynchronously right after sampling; the old log-probs and the
    reference log-probs of the first inner epoch (stored where ``install_ref_logprob_cache``
    looks) are computed while the workers simulate, and the rewards are collected before the
    advantages. Everything else follows ``SimWAMGRPOTrainer._rollout``.
    Needs both of those patches; ``SIMWAM_GRPO_OVERLAP_REWARD=0`` turns it off.
    """
    if os.environ.get("SIMWAM_GRPO_OVERLAP_REWARD", "1").strip().lower() in _OFF:
        return
    import multiprocessing as mp

    import numpy as np
    import torch
    from simwam.datasets.navsim.pdm_reward import NavSimPDMReward, _SUBMETRIC_COLUMNS
    from simwam.trainer_grpo import SimWAMGRPOTrainer

    if getattr(SimWAMGRPOTrainer._rollout, "_overlap", False):
        return
    if not getattr(NavSimPDMReward.score_batch, "_parallel", False):
        logger.warning("install_overlap_reward_logprobs needs install_parallel_pdm_reward; skipped")
        return
    rollout_orig = SimWAMGRPOTrainer._rollout
    workers = int(os.environ.get("SIMWAM_GRPO_PDM_WORKERS", "8") or 8)

    def submit(reward, abs_poses, tokens, device):
        pool = reward.__dict__.get("_pool")
        if pool is None:
            args, kwargs = reward._pool_init_args
            pool = reward._pool = mp.get_context("spawn").Pool(
                workers, initializer=_pdm_worker_init, initargs=(type(reward), args, kwargs))
        poses_np = abs_poses.detach().to(device="cpu", dtype=torch.float32).numpy()
        return pool.map_async(_pdm_worker_score, [(t, poses_np[i]) for i, t in enumerate(tokens)])

    def collect(reward, pending, tokens, device):
        results = pending.get()
        reward.last_num_total = len(tokens)
        reward.last_num_missing = 0
        reward.last_num_failed = 0
        reward._true_pdms, reward._submetric_rows = [], []
        nan_row = {name: float("nan") for name in _SUBMETRIC_COLUMNS}
        rewards = []
        for token, res in zip(tokens, results):
            if res[0] == "ok":
                reward._true_pdms.append(res[2])
                reward._submetric_rows.append(res[3])
                rewards.append(res[1])
                continue
            if res[0] == "missing":
                reward.last_num_missing += 1
            else:
                reward.last_num_failed += 1
                logger.warning("PDM scoring failed for token %s: %s", token, res[1])
            rewards.append(reward.failure_reward)
            reward._submetric_rows.append(dict(nan_row))
        return torch.tensor(rewards, device=device, dtype=torch.float32)

    @torch.no_grad()
    def _rollout(self, model, batch):
        reward = self.action_reward
        if (not isinstance(reward, NavSimPDMReward) or not hasattr(reward, "_pool_init_args")
                or not self.ppo_enabled or self.kl_gate_enabled):
            return rollout_orig(self, model, batch)
        input_image, action_horizon, context, context_mask, proprio, tokens = self._batch_condition_inputs(batch)
        cond_b = model.build_action_condition(input_image, action_horizon, context, context_mask, proprio)
        cond_bg = model.expand_condition(cond_b, self.group_size)
        chain, timesteps, deltas, stoch_idx = model.sample_action_chain(cond_bg, action_horizon, deterministic=False)
        abs_poses = self.train_dataset.denormalize_action(chain[:, -1])
        tokens_bg = [tok for tok in tokens for _ in range(self.group_size)]
        pending = submit(reward, abs_poses, tokens_bg, self.accelerator.device)
        il_abs = None
        steps_ahead = max(self.num_inner_epochs * self.rollout_buffer_batches, 1)
        will_visualize = (
            self.vis_enabled and self.accelerator.is_main_process and self.vis_every > 0
            and any((self.global_step + 1 + j) % self.vis_every == 0 for j in range(steps_ahead))
        )
        if will_visualize:
            det_chain, _, _, _ = model.sample_action_chain(cond_b, action_horizon, deterministic=True)
            il_abs = self.train_dataset.denormalize_action(det_chain[:, -1])
        old_logp = model.action_chain_logprobs(cond_bg, chain, timesteps, deltas, stoch_idx=stoch_idx).detach()
        ref = model.action_chain_logprobs(
            cond_bg, chain, timesteps, deltas, stoch_idx=stoch_idx, use_ref=True, return_terms=True)
        rewards = collect(reward, pending, tokens_bg, self.accelerator.device)
        pdm_fail_frac = float(getattr(reward, "last_fail_frac", 0.0))
        true_pdms = float(getattr(reward, "last_true_pdms", float("nan")))
        reward_submetrics = getattr(reward, "last_submetrics", None)
        bsz = len(tokens)
        rewards_mat = rewards.view(bsz, self.group_size)
        advantage_info = self._compute_advantages(rewards_mat)
        num_unavailable = int(sum((0 if reward.available(t) else 1 for t in tokens)))
        return {
            "cond_b": cond_b, "cond_bg": cond_bg, "chain": chain, "timesteps": timesteps, "deltas": deltas,
            "stoch_idx": stoch_idx, "num_sde_steps": int(stoch_idx.numel()), "pdm_fail_frac": pdm_fail_frac,
            "true_pdms": true_pdms, "reward_submetrics": reward_submetrics, "rewards": rewards.detach(),
            **advantage_info, "num_unavailable": num_unavailable, "tokens": tokens,
            "abs_group": abs_poses.reshape(bsz, self.group_size, abs_poses.shape[1], abs_poses.shape[2]),
            "il_abs": il_abs, "old_logp": old_logp, "old_mean": None, "old_std": None, "_ref_logprobs": ref,
        }

    _rollout._overlap = True
    SimWAMGRPOTrainer._rollout = _rollout
    logger.info("PDM rewards are scored while the rollout's old/reference log-probs run")


def install_foreach_grad_checks() -> None:
    """Run DeepSpeed ZeRO-1/2's overflow check and L2 gradient norm with foreach kernels.

    ``DeepSpeedZeroOptimizer`` checks every gradient for inf/NaN with an ``amax``/``amin``
    pair and computes the norm with one float64 ``vector_norm`` per gradient. With 480 LoRA
    tensors that is several thousand small launches per optimizer step on the main thread.
    By default only the overflow check is replaced: the flag becomes "some per-tensor float32
    L2 norm from ``torch._foreach_norm`` is not finite" (inf and NaN propagate; a finite
    gradient whose norm exceeds the float32 range, i.e. an element above ~1.8e19, also counts
    as overflow). The flag is a boolean, so the run stays bit-identical; the norm keeps
    DeepSpeed's per-tensor float64 code. ``SIMWAM_GRPO_FOREACH_GRAD_CHECKS=norm`` also computes
    the global norm from the foreach float32 norms (squares summed in float64), which removes
    the remaining per-tensor launches but changes the norm by ~1e-7 relative, so it is not
    bit-identical. The parameter selection, the all-reduces and the NaN masking follow
    ``get_grad_norm_direct``; other norm types, sparse gradients and CPU offload keep
    DeepSpeed's code. ``SIMWAM_GRPO_FOREACH_GRAD_CHECKS=0`` turns it off. Nothing is installed when
    ``install_manual_data_parallel`` has replaced DeepSpeed (``SIMWAM_GRPO_MANUAL_DP=1``).
    """
    mode = os.environ.get("SIMWAM_GRPO_FOREACH_GRAD_CHECKS", "1").strip().lower()
    if mode in _OFF:
        return
    from simwam.trainer_grpo import SimWAMGRPOTrainer

    if getattr(SimWAMGRPOTrainer.__init__, "_manual_dp", False):
        logger.info("Training without DeepSpeed (SIMWAM_GRPO_MANUAL_DP=1); foreach gradient checks not installed")
        return
    import torch
    import deepspeed.runtime.zero.stage_1_and_2 as s12

    Z = s12.DeepSpeedZeroOptimizer
    if getattr(Z.has_overflow_partitioned_grads_serial, "_foreach", False):
        return
    norm_orig = Z.get_grad_norm_direct
    overflow_orig = Z.has_overflow_partitioned_grads_serial

    def has_overflow_partitioned_grads_serial(self):
        grads = [g for i in range(len(self.bit16_groups)) for g in self.averaged_gradients[i]
                 if g is not None and g.numel() > 0]
        if not grads or any(g.is_sparse for g in grads):
            return overflow_orig(self)
        norms = torch.stack(torch._foreach_norm(grads, 2, dtype=torch.float32))
        return torch.isfinite(norms).all().logical_not().reshape(1)

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        if float(norm_type) != 2.0 or any(g.is_sparse for g in gradients):
            return norm_orig(self, gradients, params, norm_type)
        selected = []
        param_gradients = (g for g in gradients if not getattr(g, "_zero_padding", False))
        for g, p in zip(param_gradients, params):
            if hasattr(p, s12.PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue
            if s12.is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                selected.append(g.data.detach())
        if selected:
            norms = torch.stack(torch._foreach_norm(selected, 2, dtype=torch.float32))
            total_norm = norms.double().pow(2).sum().float()
        else:
            total_norm = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        s12.dist.all_reduce(total_norm, op=s12.dist.ReduceOp.SUM, group=self.dp_process_group)
        self._model_parallel_all_reduce(tensor=total_norm, op=s12.dist.ReduceOp.SUM)
        total_norm = total_norm.pow(0.5)
        s12.mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)
        return total_norm

    has_overflow_partitioned_grads_serial._foreach = True
    has_overflow_partitioned_grads_serial.orig = overflow_orig
    Z.has_overflow_partitioned_grads_serial = has_overflow_partitioned_grads_serial
    if mode == "norm":
        get_grad_norm_direct._foreach = True
        get_grad_norm_direct.orig = norm_orig
        Z.get_grad_norm_direct = get_grad_norm_direct
    logger.info("DeepSpeed ZeRO overflow check%s uses foreach kernels", " and gradient norm" if mode == "norm" else "")


def _flydsl_conv_module():
    """Lumen PR #32's ``flydsl_conv`` with a FlyDSL kernel behind it, or None.

    ``SIMWAM_GRPO_FLYDSL_DIR`` is the directory holding ``flydsl_conv.py``
    (``examples/wan2.2`` of a Lumen checkout of PR #32); default ``$LUMEN_DIR/examples/wan2.2``.
    The kernel is loaded by Lumen's ``examples/perf-veomni/veomni_patches/aiter_conv.py`` when it
    is there (``SIMWAM_GRPO_FLYDSL_LOADER_DIR``, default ``$LUMEN_DIR/examples/perf-veomni``):
    on an aiter that includes ROCm/aiter#5370 it is a plain import, and on an older one (the
    image's, after ``overlay_aiter.py --apply`` has added the kernel files) it also supplies the
    tuned-table path and works around the older ``aiter.ops.flydsl/__init__.py``; without the
    path every lookup would silently take the heuristic tile. Otherwise ``flydsl_conv.load()``
    imports ``aiter.ops.flydsl.conv_kernels`` directly.
    """
    import importlib
    import sys

    lumen_dir = os.environ.get("LUMEN_DIR", "")
    here = os.environ.get("SIMWAM_GRPO_FLYDSL_DIR") or os.path.join(lumen_dir, "examples", "wan2.2")
    if not os.path.isfile(os.path.join(here, "flydsl_conv.py")):
        logger.warning("flydsl_conv: no flydsl_conv.py in %s (set SIMWAM_GRPO_FLYDSL_DIR)", here)
        return None
    if here not in sys.path:
        sys.path.insert(0, here)
    fc = importlib.import_module("flydsl_conv")

    loader_dir = os.environ.get("SIMWAM_GRPO_FLYDSL_LOADER_DIR") or os.path.join(lumen_dir, "examples", "perf-veomni")
    if os.path.isfile(os.path.join(loader_dir, "veomni_patches", "aiter_conv.py")):
        if loader_dir not in sys.path:
            sys.path.insert(0, loader_dir)
        from veomni_patches import aiter_conv as loader

        if loader.load() is not None:
            fc.load = loader.load
            fc.lookup_stats = loader.tuned_lookup_stats
            logger.info("flydsl_conv: kernel through %s/veomni_patches/aiter_conv.py", loader_dir)
            return fc
        logger.warning(
            "flydsl_conv: FlyDSL conv unavailable on this aiter; run %s/overlay_aiter.py --apply "
            "in this container, or use an aiter with ROCm/aiter#5370", loader_dir,
        )  # fmt: skip
    # No perf-veomni loader (or it failed): the plain import, for an aiter that includes the kernel.
    if fc.load() is not None:
        logger.info("flydsl_conv: kernel from the installed aiter")
        return fc
    return None


def install_vae_flydsl_conv(model) -> bool:
    """Route every convolution of the VAE (``model.vae``, a ``WanVideoVAE38``) to aiter's FlyDSL kernel.

    The rewrites are PR #32's ``flydsl_conv.patch_vae``: a ``CausalConv3d`` called with one frame
    and no feature cache (every call of the rollout's single-frame encode) is the exact 2-D
    convolution with the last temporal tap of its kernel; every other call keeps the module's
    causal padding and only the convolution under it moves. The kernel is forward-only and
    BF16-only; any other call, and any call that needs a gradient, stays on torch and is counted.

    Logs the FlyDSL / torch calls of the first ``SIMWAM_GRPO_FLYDSL_LOG`` encodes (default 2;
    ``all`` for every encode). Returns whether the VAE was patched.
    """
    vae = getattr(model, "vae", None)
    if vae is None or getattr(vae, "_flydsl_conv", False):
        return False
    fc = _flydsl_conv_module()
    if fc is None:
        return False
    n_c3, n_c2, n_t1 = fc.patch_vae(vae)
    vae._flydsl_conv = True
    logger.info(
        "flydsl_conv: VAE %d CausalConv3d (%d with the T=1 -> conv2d rewrite) and %d Conv2d on FlyDSL",
        n_c3, n_t1, n_c2,
    )  # fmt: skip
    _STATS["flydsl_conv"] = {"encodes": 0}

    limit = os.environ.get("SIMWAM_GRPO_FLYDSL_LOG", "2").strip().lower()
    encode = vae.encode

    def logged_encode(*args, **kwargs):
        before = fc.conv_calls()
        out = encode(*args, **kwargs)
        after = fc.conv_calls()
        stats = _STATS["flydsl_conv"]
        stats["encodes"] += 1
        for k in after:
            stats[k] = stats.get(k, 0) + after[k] - before[k]
        if limit == "all" or stats["encodes"] <= int(limit or 0):
            logger.info(
                "flydsl_conv: VAE encode %d: conv calls %s, tuned-table lookups %s",
                stats["encodes"], {k: after[k] - before[k] for k in after}, fc.lookup_stats()["calls"],
            )  # fmt: skip
        return out

    vae.encode = logged_encode
    return True


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

    applied = []
    for key in requested:
        if key in _PLUGIN_PATCHES:
            continue
        if key not in _PATCHES:
            logger.warning("Unknown LUMEN_ENABLE flag %r; ignored", key)
            continue
        from lumen.config import LumenConfig

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
    # After LumenConfig, so that it rebinds the VAE convolutions aiter_conv bound.
    if "flydsl_conv" in requested:
        try:
            if install_vae_flydsl_conv(model):
                applied.append("flydsl_conv")
        except Exception as exc:
            logger.warning("flydsl_conv failed (%s); the VAE convolutions keep their previous path", exc)
    _log_coverage(model)
    logger.info("Lumen enabled on SimWAM: %s", applied)
    return model
