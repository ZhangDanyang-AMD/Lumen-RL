"""DeepSeek-V4.1 on the ``megatron_native`` engine.

V4.1 is not a ``GPTModel`` and does not need to be one. Its non-expert weights
are ~16 GB in BF16, so every data-parallel rank holds them whole (TP=1), and
only the MoE is Megatron's: expert-parallel alltoall and grouped GEMMs
(``models.deepseek_v41.megatron_moe``). The rest of a pipeline stage is the
release-checked PyTorch backbone (``models.deepseek_v41.model.DSV41Stage``),
loaded straight from the released FP8 / FP4 checkpoint, with the ~98 GB Engram
tables mapped from the checkpoint file in host memory. ``DSV41MegatronModel``
gives it what Megatron's pipeline schedule calls.

Trained: everything except the routed experts (FP4 in the release, and ATOM
cannot take them in a weight sync), the indexers (no gradient reaches what is
a discrete top-k; left trainable, weight decay alone would move them), the
router's expert bias and the Engram tables, which are not parameters.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

FROZEN = (".ffn.layer.experts.", ".attn.indexer.")


def is_dsv41(hf: dict) -> bool:
    """The released config.json (``model_type`` ``deepseek_v41``, the language
    model under ``text_config``) or that ``text_config`` itself."""
    types = (hf.get("model_type"), (hf.get("text_config") or {}).get("model_type"))
    if any(str(t or "").lower().startswith("deepseek_v41") for t in types):
        return True
    return any("deepseekv41" in str(a).lower() for a in hf.get("architectures") or [])


def language_config(hf: dict) -> dict:
    return hf.get("text_config", hf)


def build_dsv41_config(
    hf: dict,
    ec: dict,
    *,
    tp: int,
    pp: int,
    cp: int,
    ep: int,
    etp: int,
    sp: bool,
    max_tokens_per_gpu: int = 0,
    deterministic: bool = True,
) -> Any:
    """The ``TransformerConfig`` the MoE layers and Megatron's schedule read.

    ``dsv4_mode`` is what makes the DSv4-patched p2p exchange the 4-D shape of
    the stream crossing a stage boundary (``[b, s, hc, d + 2]``, see
    ``pack_stream``). ``deterministic`` (the engine's default) matters as much
    as for DSv4: without it the MoE's combine sums in a different order every
    run, and on slice B two forwards of the same batch differ by 0.28 in mean
    token log-prob, the training forward from the log-prob one by a PPO-KL of
    0.03; with it both are exactly 0.
    """
    from lumenrl.engine.training.bridges.dsv4 import _require_dsv4_megatron
    from lumenrl.engine.training.megatron_base_engine import moe_dispatcher_kwargs
    from lumenrl.models.deepseek_v41.config import DSV41Config
    from lumenrl.models.deepseek_v41.megatron_moe import moe_config

    if (tp, cp, etp) != (1, 1, 1):
        raise NotImplementedError(
            f"DeepSeek-V4.1 replicates its non-expert weights over data parallelism: "
            f"tp, cp and etp must be 1, got {tp}, {cp}, {etp}"
        )
    _require_dsv4_megatron()
    cfg = DSV41Config.from_hf(hf)
    if cfg.n_layers % pp:
        raise ValueError(f"{cfg.n_layers} layers do not split evenly over PP {pp}")
    overrides = {
        "dsv4_mode": True,
        "dsv4_hc_mult": cfg.hc_mult,
        "freeze_e_score_correction_bias": True,
        "deterministic_mode": deterministic,
        "gradient_accumulation_fusion": False,
        "perform_initialization": False,
        "moe_permute_fusion": bool(ec.get("moe_permute_fusion", False)),
        **moe_dispatcher_kwargs(ec, tp=tp, cp=cp, sp=sp, max_tokens_per_gpu=max_tokens_per_gpu),
    }
    return moe_config(cfg, ep=ep, pp=pp, **overrides)


def stage_layers(n_layers: int, pp: int, pp_rank: int) -> tuple[int, int]:
    per = n_layers // pp
    return pp_rank * per, (pp_rank + 1) * per


@functools.cache
def megatron_model_class() -> type:
    """``DSV41MegatronModel``, defined on first use so that importing this
    module (every engine does, through ``model_specs``) needs no Megatron."""
    from megatron.core.enums import ModelType
    from megatron.core.transformer.module import MegatronModule

    class DSV41MegatronModel(MegatronModule):
        """One pipeline stage of V4.1 in ``GPTModel``'s place.

        ``forward`` takes the packed microbatch ``_pp_forward_model`` builds,
        which for V4.1 is one sequence ``[1, s]`` (its attention reads the whole
        tensor as one sequence, so ``packed_seq_params`` means nothing to it).
        The last stage returns fp32 logits ``[1, s, vocab]``, any other the
        packed stream for the next stage, whose input the schedule hands over
        with ``set_input_tensor``.
        """

        def __init__(self, config, stage):
            super().__init__(config=config)
            self.stage = stage
            self.model_type = ModelType.encoder_or_decoder
            self.pre_process, self.post_process = stage.pre_process, stage.post_process
            self.share_embeddings_and_output_weights = False
            self.input_tensor = None

        def set_input_tensor(self, input_tensor):
            if isinstance(input_tensor, (list, tuple)):
                input_tensor = input_tensor[0]
            self.input_tensor = input_tensor

        def forward(self, input_ids, position_ids=None, attention_mask=None, packed_seq_params=None, **_):
            return self.stage(input_ids, None if self.pre_process else self.input_tensor)

    return DSV41MegatronModel


def quant_points(ec: dict):
    """``QuantPoints`` from ``megatron_cfg.dsv41_quant_points`` (field -> bool);
    unset fields keep the release's defaults."""
    from lumenrl.models.deepseek_v41.model import QuantPoints

    return QuantPoints(**dict(ec.get("dsv41_quant_points") or {}))


def engram_token_map(model_dir: str, cfg) -> torch.Tensor:
    from transformers import PreTrainedTokenizerFast

    from lumenrl.models.deepseek_v41.engram import build_compressed_token_map

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
    lookup, size = build_compressed_token_map(tokenizer)
    if size != cfg.engram_compressed_vocab_size:
        raise ValueError(
            f"the tokenizer compresses to {size} ids, the config says {cfg.engram_compressed_vocab_size}"
        )
    return torch.tensor(lookup)


def build_dsv41_model(engine, hf: dict, tfcfg) -> torch.nn.Module:
    """This rank's stage, on its GPU, loaded from the release at ``engine.model_name``."""
    from megatron.core import parallel_state as mpu

    from lumenrl.models.deepseek_v41.config import DSV41Config
    from lumenrl.models.deepseek_v41.convert import load_released
    from lumenrl.models.deepseek_v41.megatron_moe import megatron_ffn
    from lumenrl.models.deepseek_v41.model import DSV41Stage

    ec = engine.engine_config
    if engine._r3_enabled:
        raise NotImplementedError("DeepSeek-V4.1 has no routing replay yet")
    cfg = DSV41Config.from_hf(hf)
    pp, pp_rank = mpu.get_pipeline_model_parallel_world_size(), mpu.get_pipeline_model_parallel_rank()
    start, end = stage_layers(cfg.n_layers, pp, pp_rank)
    quant = quant_points(ec)
    recompute = ec.get("recompute_granularity") == "full"
    token_map = engram_token_map(engine.model_name, cfg) if cfg.engram_layout_layer_ids else None
    stage = DSV41Stage(
        cfg, start, end, token_map=token_map, quant=quant, recompute=recompute,
        ffn=megatron_ffn(tfcfg), engram_on_host=True,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    stage.to(device)
    # read and dequantized on the GPU: a rank's FP4 experts are ~70 GB of BF16 at EP=8
    report = load_released(stage, engine.model_name, device=device)
    for name, p in stage.named_parameters():
        if any(part in name for part in FROZEN):
            p.requires_grad_(False)
    trained = sum(p.numel() for p in stage.parameters() if p.requires_grad)
    held = sum(p.numel() for p in stage.parameters())
    logger.info(
        "MegatronNativeEngine[%d]: DeepSeek-V4.1 layers [%d, %d) of %d, %d tensors loaded, "
        "%d engram tables mapped from %s | %.2fB params here, %.2fB trained | ep=%d pp=%d "
        "recompute=%s quant=%s",
        engine._rank(), start, end, cfg.n_layers, report["loaded"], report["mapped"],
        engine.model_name, held / 1e9, trained / 1e9, engine._ep, pp, recompute, quant,
    )
    return megatron_model_class()(tfcfg, stage)


def checkpoint_names(name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """A ``DSV41MegatronModel`` parameter as the released checkpoint names it
    (what ATOM's weight updater takes), split where Megatron fuses."""
    name = name.removeprefix("module.").removeprefix("stage.")
    if ".ffn.layer." not in name:
        return [(name, tensor)]
    block, rest = name.split(".ffn.layer.", 1)
    if rest == "router.weight":
        return [(f"{block}.ffn.gate.weight", tensor)]
    if rest == "shared_experts.linear_fc1.weight":
        gate, up = tensor.chunk(2)
        return [(f"{block}.ffn.shared_experts.w1.weight", gate), (f"{block}.ffn.shared_experts.w3.weight", up)]
    if rest == "shared_experts.linear_fc2.weight":
        return [(f"{block}.ffn.shared_experts.w2.weight", tensor)]
    raise KeyError(f"{name}: no checkpoint name for this MoE parameter")


def _trained(module) -> list[tuple[str, torch.Tensor]]:
    return [
        pair
        for name, p in module.named_parameters()
        if p.requires_grad
        for pair in checkpoint_names(name, p.detach())
    ]


def export_dsv41(engine, **_) -> Iterator[tuple[str, torch.Tensor]]:
    """The trained parameters of the whole model under their checkpoint names,
    one tensor at a time, on every rank: each stage's in turn, broadcast over
    the PP group (a stage's names are already the model's layer numbers).
    Collective: every rank must drain it."""
    from megatron.core import parallel_state as mpu

    local = _trained(engine.module)
    pp = mpu.get_pipeline_model_parallel_world_size()
    if pp == 1:
        yield from local
        return
    group, rank = mpu.get_pipeline_model_parallel_group(), mpu.get_pipeline_model_parallel_rank()
    meta: list = [None] * pp
    dist.all_gather_object(meta, [(k, tuple(t.shape), t.dtype) for k, t in local], group=group)
    mine = dict(local)
    for src in range(pp):
        root = dist.get_global_rank(group, src)
        for name, shape, dtype in meta[src]:
            t = mine[name].contiguous() if src == rank else torch.empty(shape, dtype=dtype, device="cuda")
            dist.broadcast(t, src=root, group=group)
            yield name, t
