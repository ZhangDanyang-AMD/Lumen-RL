"""Fused-MoE weight sync between a transformers-5.x trainer and vLLM.

transformers 5.x stores Qwen3-MoE experts as fused 3D tensors, so the training
side's ``state_dict()`` -- which is exactly what the IPC weight sync sends --
emits::

    model.layers.N.mlp.experts.gate_up_proj    (E, 2*I, H)
    model.layers.N.mlp.experts.down_proj       (E, H, I)

vLLM calls the same buffers ``experts.w13_weight`` / ``experts.w2_weight`` and
its ``expert_params_mapping`` only recognises the *per-expert* checkpoint names
``experts.{id}.{gate,up,down}_proj.weight``. A fused name therefore matches no
mapping and falls through ``Qwen3MoeForCausalLM.load_weights``'s silent
``if is_expert_weight: continue`` branch: no exception, no loaded param, and
93% of the policy's parameters never reach the rollout engine.

This module closes that gap on the receiving side. ``FusedMoEWeightRouter``
intercepts the fused tensors and feeds them to vLLM's own
``FusedMoE.weight_loader`` through its 3D "full load" path (one call per
shard instead of 128), so TP sharding, hidden-dim padding and any load-time
kernel-format handling stay vLLM's business. The layouts are element-wise
identical -- verified against Qwen3-30B-A3B-Base by loading the same checkpoint
through HF and through vLLM and comparing ``gate_up_proj`` with ``w13_weight``
bit for bit on layers 0/1/23/47.

``assert_weight_sync_coverage`` is the second half: the original bug survived 54
training steps only because ``load_weights()``'s return value was discarded, so
every sync now has to account for every parameter in the rollout model.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Sequence

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LUMENRL_LOGGING_LEVEL", "WARN"))

# transformers-5.x fused expert parameter names -> vLLM shard ids. gate_up_proj
# carries two logical shards (w1 = gate, w3 = up) concatenated along dim 1, in
# that order, matching HF's `linear(x, gate_up_proj[e]).chunk(2, dim=-1)`.
_FUSED_GATE_UP = "gate_up_proj"
_FUSED_DOWN = "down_proj"
_FUSED_LEAVES = (_FUSED_GATE_UP, _FUSED_DOWN)

# Parameters that a BF16 trainer legitimately never sends: quantization
# artifacts that vLLM derives locally in process_weights_after_loading.
_COVERAGE_IGNORE_SUFFIXES = (
    "_scale",
    "_scale_inv",
    "_scale_2",
    "_offset",
    "_zero_point",
    "_shape",
    "g_idx",
)


def _split_name(name: str) -> tuple[str, str]:
    prefix, _, leaf = name.rpartition(".")
    return prefix, leaf


class FusedMoEWeightRouter:
    """Load transformers-style fused expert tensors into vLLM FusedMoE params.

    Instantiate once per weight-sync round (discovery walks ``named_modules``)
    and call :meth:`route` on every bucket. When the rollout model has no
    FusedMoE layers -- the dense 8B path -- :attr:`active` is False and
    :meth:`route` degenerates to returning its input.
    """

    def __init__(self, model: torch.nn.Module):
        # experts module prefix -> (module, w13 param name, w2 param name)
        self._experts: dict[str, tuple[torch.nn.Module, str, str]] = {}
        self._params = dict(model.named_parameters())
        self._verify = os.environ.get("LUMENRL_WEIGHT_SYNC_VERIFY", "0") == "1"

        for mod_name, module in model.named_modules():
            if not callable(getattr(module, "weight_loader", None)):
                continue
            owned = {n for n, _ in module.named_parameters()}
            w13 = next((n for n in owned if n.endswith("w13_weight")), None)
            w2 = next((n for n in owned if n.endswith("w2_weight")), None)
            if w13 is None or w2 is None:
                continue
            self._experts[mod_name] = (module, f"{mod_name}.{w13}", f"{mod_name}.{w2}")

        if self._experts:
            logger.info(
                "fused-MoE weight router: %d expert modules (e.g. %s)",
                len(self._experts),
                next(iter(self._experts)),
            )

    @property
    def active(self) -> bool:
        return bool(self._experts)

    def route(
        self, weights: Sequence[tuple[str, torch.Tensor]]
    ) -> tuple[list[tuple[str, torch.Tensor]], set[str]]:
        """Split a bucket into (weights for ``model.load_weights``, names loaded here)."""
        if not self._experts:
            return list(weights), set()

        passthrough: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        for name, tensor in weights:
            prefix, leaf = _split_name(name)
            entry = self._experts.get(prefix)
            if entry is None or leaf not in _FUSED_LEAVES:
                passthrough.append((name, tensor))
                continue
            loaded.add(self._load_fused(entry, name, leaf, tensor))
        return passthrough, loaded

    def _load_fused(
        self,
        entry: tuple[torch.nn.Module, str, str],
        name: str,
        leaf: str,
        tensor: torch.Tensor,
    ) -> str:
        module, w13_name, w2_name = entry
        if tensor.ndim != 3:
            raise RuntimeError(
                f"fused MoE weight {name} must be 3D (experts, out, in), got "
                f"{tuple(tensor.shape)}. The training side is not emitting the "
                "transformers-5.x fused layout this router expects."
            )

        param_name = w13_name if leaf == _FUSED_GATE_UP else w2_name
        param = self._params[param_name]
        if leaf == _FUSED_DOWN:
            shards = [("w2", tensor)]
        elif self._is_act_and_mul(module):
            half = tensor.shape[1] // 2
            shards = [("w1", tensor[:, :half]), ("w3", tensor[:, half:])]
        else:
            shards = [("w1", tensor)]

        for shard_id, shard in shards:
            self._dispatch(module, param, param_name, shard_id, shard)
        if self._verify:
            self._verify_written(module, param, param_name, shards)
        return param_name

    def _verify_written(
        self,
        module: torch.nn.Module,
        param: torch.nn.Parameter,
        param_name: str,
        shards: list[tuple[str, torch.Tensor]],
    ) -> None:
        """Read back what vLLM stored and require it to equal what was sent.

        Only meaningful without TP/EP sharding, where the destination holds the
        whole tensor; that is the colocated TP=1 replica layout this path uses.
        """
        if self._ep_size(module) != 1 or self._tp_size(module) != 1:
            logger.warning("weight sync verify skipped for %s: sharded layer", param_name)
            return
        offset = 0
        for shard_id, shard in shards:
            dest = param.data.narrow(1, offset, shard.shape[1])
            offset += shard.shape[1]
            if dest.shape != shard.shape:
                logger.warning(
                    "weight sync verify skipped for %s shard %s: padded destination %s",
                    param_name, shard_id, tuple(dest.shape),
                )
                continue
            if not torch.equal(dest, shard):
                raise RuntimeError(
                    f"weight sync verify failed for {param_name} shard {shard_id}: "
                    "vLLM's buffer does not match the tensor the trainer sent"
                )

    def _dispatch(
        self,
        module: torch.nn.Module,
        param: torch.nn.Parameter,
        param_name: str,
        shard_id: str,
        shard: torch.Tensor,
    ) -> None:
        """Hand one logical shard to vLLM's loader, whole or expert by expert.

        A 3D ``loaded_weight`` puts ``FusedMoE.weight_loader`` on its ``full_load``
        branch, which writes all experts in one ``copy_``. That branch assumes the
        parameter holds every expert, so under expert parallelism -- where the
        parameter only holds this rank's slice -- fall back to per-expert calls
        and let the loader map global expert ids to local ones.
        """
        if self._ep_size(module) == 1:
            ok = module.weight_loader(
                param, shard, param_name, shard_id=shard_id, expert_id=0,
                return_success=True,
            )
            if ok is False:
                raise RuntimeError(
                    f"vLLM refused the fused {shard_id} shard of {param_name}"
                )
            return

        loaded_any = False
        for expert_id in range(shard.shape[0]):
            loaded_any |= bool(
                module.weight_loader(
                    param,
                    shard[expert_id],
                    param_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
            )
        if not loaded_any:
            raise RuntimeError(
                f"no expert of the {shard_id} shard of {param_name} was local to "
                "this rank; the expert-parallel map and the sent tensor disagree"
            )

    @staticmethod
    def _is_act_and_mul(module: torch.nn.Module) -> bool:
        moe_config = getattr(module, "moe_config", None)
        return bool(getattr(moe_config, "is_act_and_mul", True))

    @staticmethod
    def _ep_size(module: torch.nn.Module) -> int:
        return FusedMoEWeightRouter._parallel_size(module, "ep_size")

    @staticmethod
    def _tp_size(module: torch.nn.Module) -> int:
        return FusedMoEWeightRouter._parallel_size(module, "tp_size")

    @staticmethod
    def _parallel_size(module: torch.nn.Module, attr: str) -> int:
        moe_config = getattr(module, "moe_config", None)
        parallel = getattr(moe_config, "moe_parallel_config", None)
        return int(getattr(parallel, attr, 1) or 1)


def assert_weight_sync_coverage(
    model: torch.nn.Module, loaded: Iterable[str], *, context: str = "ipc"
) -> None:
    """Fail loudly when a weight sync left parameters at their previous values.

    vLLM's ``load_weights`` skips names it does not recognise without raising,
    so an unnoticed naming change silently freezes part of the rollout policy
    while training continues -- the failure mode is a slowly growing
    train/rollout divergence, not a crash. Comparing the loaded set against
    ``named_parameters`` turns that into an immediate error.

    ``LUMENRL_WEIGHT_SYNC_CHECK`` selects ``error`` (default), ``warn`` or ``off``.
    """
    mode = os.environ.get("LUMENRL_WEIGHT_SYNC_CHECK", "error").lower()
    if mode == "off":
        return

    loaded = set(loaded)
    expected = {
        name
        for name in dict(model.named_parameters())
        if not name.endswith(_COVERAGE_IGNORE_SUFFIXES)
    }
    missing = sorted(expected - loaded)
    if not missing:
        logger.info("weight sync coverage (%s): %d/%d params", context, len(expected), len(expected))
        return

    sample = ", ".join(missing[:8])
    message = (
        f"weight sync ({context}) left {len(missing)}/{len(expected)} rollout "
        f"parameters untouched: {sample}"
        f"{' ...' if len(missing) > 8 else ''}. The rollout engine is now serving "
        "a mix of current and stale weights. Set LUMENRL_WEIGHT_SYNC_CHECK=warn "
        "to downgrade this to a log line."
    )
    if mode == "warn":
        logger.warning(message)
        return
    raise RuntimeError(message)
