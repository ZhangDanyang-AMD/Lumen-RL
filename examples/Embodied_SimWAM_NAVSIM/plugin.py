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

    def capture(self, args, cond, params):
        for expert in (self.action_expert, getattr(self, "ref_action_expert", None)):
            freqs = getattr(expert, "freqs", None)
            if torch.is_tensor(freqs) and freqs.device != args[0].device:
                expert.freqs = freqs.to(args[0].device)
        layers = len(cond["video_kv_cache"])
        rest = {k: v for k, v in cond.items() if k not in _COND and k != "_cuda_graph_tag"}
        inputs = [torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device) for t in args]
        torch._foreach_copy_(inputs, list(args))

        def run():
            return eager(self, inputs[0], inputs[1], unflat(rest, layers, inputs[2:]), use_ref=False)

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
        args = (latents_action, timestep_action, *flat(cond))
        key = (tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in args), len(params),
               tuple(k for k in cond if k not in _COND), int(cond.get("video_seq_len", -1)))
        g = graphs.get(key)
        if g is None:
            try:
                g = capture(self, args, cond, params)
            except Exception as exc:
                logger.warning("CUDA graph capture of the training action_velocity failed (%s); using eager", exc)
                g = False
            graphs[key] = g
        if g is False:
            stats["eager"] += 1
            return eager(self, latents_action, timestep_action, cond, use_ref=False)
        if verify:
            ref = eager(self, latents_action, timestep_action, cond, use_ref=False)
            w = torch.randn_like(ref)
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
    ``SIMWAM_GRPO_MANUAL_DP=1`` turns this on (off by default). Call before the trainer is built.
    """
    if os.environ.get("SIMWAM_GRPO_MANUAL_DP", "0").strip().lower() in _OFF:
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
