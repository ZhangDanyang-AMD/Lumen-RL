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


def install_fused_lora_add() -> None:
    """``LoRALinear.forward`` adds the scaled adapter output with one ``torch.add(alpha=)``.

    SimWAM computes ``out + self.scaling * lora`` as a multiply and an add. With the
    default ``alpha / r = 2.0`` the scale is a power of two, so ``torch.add(out, lora,
    alpha=scaling)`` is expected to round the same way. ``SIMWAM_GRPO_FUSED_LORA=0``
    turns it off.
    """
    if os.environ.get("SIMWAM_GRPO_FUSED_LORA", "1").strip().lower() in _OFF:
        return
    import torch
    from simwam.models.wan22.lora import LoRALinear

    if getattr(LoRALinear.forward, "_fused_add", False):
        return

    def forward(self, x):
        out = self.base(x)
        lora = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return torch.add(out, lora, alpha=self.scaling)

    forward._fused_add = True
    LoRALinear.forward = forward
    logger.info("LoRALinear adds the adapter with torch.add(alpha=scaling)")


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
    Here both come from ``torch._foreach_norm`` (float32 per tensor): the overflow flag is
    "some per-tensor L2 norm is not finite" (inf and NaN propagate; a finite gradient whose
    norm exceeds the float32 range also counts as overflow), and the global norm sums the
    squared per-tensor norms in float64. The parameter selection, the all-reduces and the
    NaN masking follow ``get_grad_norm_direct``; other norm types, sparse gradients and CPU
    offload keep DeepSpeed's code. ``SIMWAM_GRPO_FOREACH_GRAD_CHECKS=0`` turns it off.
    """
    if os.environ.get("SIMWAM_GRPO_FOREACH_GRAD_CHECKS", "1").strip().lower() in _OFF:
        return
    import torch
    import deepspeed.runtime.zero.stage_1_and_2 as s12

    Z = s12.DeepSpeedZeroOptimizer
    if getattr(Z.get_grad_norm_direct, "_foreach", False):
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

    get_grad_norm_direct._foreach = True
    get_grad_norm_direct.orig = norm_orig
    has_overflow_partitioned_grads_serial.orig = overflow_orig
    Z.get_grad_norm_direct = get_grad_norm_direct
    Z.has_overflow_partitioned_grads_serial = has_overflow_partitioned_grads_serial
    logger.info("DeepSpeed ZeRO overflow check and gradient norm use foreach kernels")


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
