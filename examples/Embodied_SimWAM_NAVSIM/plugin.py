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
        key = (bool(use_ref), tuple(latents_action.shape), tuple(latents_action.stride()), tuple(timestep_action.shape),
               int(cond["video_seq_len"]), tuple((tuple(t.shape), t.dtype) for t in tensors))
        g = graphs.get(key)
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
    action_velocity.eager = eager
    ActionGRPOMixin.action_velocity = action_velocity
    logger.info("action_velocity replays CUDA graphs under no_grad (verify=%s)", verify)


def install_batched_logprobs() -> None:
    """Score all stochastic steps of a chain with one ``action_velocity`` call.

    ``action_chain_logprobs`` calls the action expert once per stochastic step (3 by
    default), each on the same ``B*G`` samples. Stacking the steps along the batch runs
    the expert once on ``W*B*G`` samples, so its forward and backward launch a third of
    the kernels. The per-step SDE mean, std, clamping and reduction are unchanged; the
    condition (video K/V cache, context) is repeated once per condition and reused.
    Only the rigorous-SDE, unmerged path is batched; anything else uses the original.
    ``SIMWAM_GRPO_BATCHED_LOGP=0`` turns this off.
    """
    if os.environ.get("SIMWAM_GRPO_BATCHED_LOGP", "1").strip().lower() in _OFF:
        return
    import math

    import torch
    from torch.distributions import Normal
    from simwam.models.wan22.simwam_grpo import ActionGRPOMixin

    if getattr(ActionGRPOMixin.action_chain_logprobs, "_batched", False):
        return
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

    def action_chain_logprobs(self, cond, chain, timesteps, deltas, stoch_idx=None, use_ref=False,
                              return_terms=False):
        if getattr(self, "grpo_merge_sde_window", False) or self.grpo_sde_mode != "rigorous":
            return original(self, cond, chain, timesteps, deltas, stoch_idx=stoch_idx, use_ref=use_ref,
                            return_terms=return_terms)
        sched = self.infer_action_scheduler
        device = self.device
        dtype = getattr(self, "grpo_action_dtype", self.torch_dtype)
        batch_size = int(chain.shape[0])
        steps = self._normalize_stoch_idx(stoch_idx, int(timesteps.shape[0]))
        if len(steps) < 2:
            return original(self, cond, chain, timesteps, deltas, stoch_idx=stoch_idx, use_ref=use_ref,
                            return_terms=return_terms)
        x_all = torch.cat([chain[:, k] for k in steps], dim=0)
        t_all = torch.cat([
            timesteps[k].to(device=device, dtype=dtype).reshape(1).expand(batch_size) for k in steps
        ])
        v_all = self.action_velocity(x_all, t_all, repeat_cond(cond, len(steps)), use_ref=use_ref)
        logps, means, vels, stds = [], [], [], []
        for i, k in enumerate(steps):
            x_k = x_all[i * batch_size:(i + 1) * batch_size]
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

    action_chain_logprobs._batched = True
    action_chain_logprobs.original = original
    ActionGRPOMixin.action_chain_logprobs = action_chain_logprobs
    logger.info("action_chain_logprobs scores all stochastic steps in one action_velocity call")


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
