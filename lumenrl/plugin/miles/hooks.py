"""miles CLI hook functions for LumenRL injection.

Usage — pass each function via its dotted import path::

    python train.py \\
      --custom-megatron-init-path              lumenrl.plugin.miles.hooks.custom_megatron_init \\
      --custom-model-provider-path             lumenrl.plugin.miles.hooks.custom_model_provider \\
      --rollout-function-path                  lumenrl.plugin.miles.hooks.generate_rollout \\
      --custom-megatron-post-save-hook-path    lumenrl.plugin.miles.hooks.custom_megatron_post_save_hook \\
      --custom-megatron-before-train-step-hook-path lumenrl.plugin.miles.hooks.custom_megatron_before_train_step_hook \\
      --custom-megatron-before-log-prob-hook-path   lumenrl.plugin.miles.hooks.custom_megatron_before_log_prob_hook \\
      --custom-update-weight-post-write-path   lumenrl.plugin.miles.hooks.custom_update_weight_post_write

All hooks are self-contained — no cross-plugin imports from vime or slime.
``generate_rollout`` returns miles-native ``RolloutFnTrainOutput`` /
``RolloutFnEvalOutput`` from ``miles.rollout.base_types``.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook 1: --custom-megatron-init-path
# Signature: def custom_init(args: Namespace) -> None
# Called at the end of miles/backends/megatron_utils/initialize.py:init()
# ---------------------------------------------------------------------------

def custom_megatron_init(args: Namespace) -> None:
    """Set up Lumen FP8 training after Megatron distributed init.

    Reads ``args.lumenrl_fp8_config`` (a dict or QuantizationConfig).
    Stores the FP8TrainingManager on ``args._lumenrl_fp8_manager`` so that
    ``custom_model_provider`` can call ``manager.enable(model)`` later.
    """
    fp8_config = getattr(args, "lumenrl_fp8_config", None)
    if fp8_config is None:
        logger.info("lumenrl: args.lumenrl_fp8_config not set; FP8 disabled.")
        return

    from lumenrl.core.config import QuantizationConfig
    from lumenrl.quantization.fp8_training import FP8TrainingManager

    if isinstance(fp8_config, dict):
        quant_cfg = QuantizationConfig(**fp8_config)
    else:
        quant_cfg = fp8_config

    args._lumenrl_fp8_manager = FP8TrainingManager(quant_cfg)
    logger.info("lumenrl: FP8TrainingManager created and stored on args.")


# ---------------------------------------------------------------------------
# Hook 2: --custom-model-provider-path
# Signature: def custom_model_provider(
#     pre_process: bool, post_process: bool, vp_stage: int | None = None
# ) -> GPTModel
# Called from miles/backends/megatron_utils/model_provider.py
# ---------------------------------------------------------------------------

def custom_model_provider(
    pre_process: bool,
    post_process: bool,
    vp_stage: int | None = None,
):
    """Build a Megatron GPTModel and optionally enable Lumen FP8.

    Returns a ``megatron.core.models.gpt.GPTModel`` instance. If
    ``args._lumenrl_fp8_manager`` was set by ``custom_megatron_init``,
    FP8 quantization is applied to the model before returning.
    """
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.global_vars import get_args as megatron_get_args

    args = megatron_get_args()
    transformer_config = core_transformer_config_from_args(args)
    layer_spec = get_gpt_layer_local_spec()

    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
    )

    fp8_manager = getattr(args, "_lumenrl_fp8_manager", None)
    if fp8_manager is not None:
        fp8_manager.enable(model)
        logger.info("lumenrl: FP8 enabled on GPTModel.")

    return model


# ---------------------------------------------------------------------------
# Hook 3: --rollout-function-path
# Signature: def generate_rollout(
#     args, rollout_id: int, data_source, evaluation: bool = False
# ) -> miles.rollout.base_types.RolloutFnTrainOutput | RolloutFnEvalOutput
# Called from miles/ray/rollout/rollout_manager.py via call_rollout_fn()
# ---------------------------------------------------------------------------

def generate_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
):
    """Drive ATOM rollout in place of miles' default sglang rollout.

    Requires ``args._lumenrl_atom_manager`` (an ``ATOMReplicaManager``)
    to be set by the trainer script before the first rollout call.
    """
    manager = getattr(args, "_lumenrl_atom_manager", None)
    if manager is None:
        raise RuntimeError(
            "args._lumenrl_atom_manager not set. Create ATOMReplicaManager "
            "and store it on args before the first rollout call."
        )

    if evaluation:
        logger.info("lumenrl: ATOM eval rollout (rollout_id=%d)", rollout_id)
        from miles.rollout.base_types import RolloutFnEvalOutput

        return RolloutFnEvalOutput(data={})

    import ray
    from miles.rollout.base_types import RolloutFnTrainOutput
    from miles.utils.processing_utils import load_tokenizer
    from miles.utils.types import Sample

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    max_tokens = getattr(args, "rollout_max_response_len", 2048)
    batch_size = getattr(args, "rollout_batch_size", 1)
    num_servers = len(manager.servers)

    manager.wake_all(tags=["weights", "kv_cache"])

    sample_groups: list[list[Sample]] = data_source.get_samples(batch_size)
    results: list[list[Sample]] = []

    for group in sample_groups:
        group_out: list[Sample] = []
        for sample in group:
            prompt_text = sample.prompt if isinstance(sample.prompt, str) else ""
            prompt_ids = sample.tokens if sample.tokens else tokenizer.encode(prompt_text)

            sampling_params = {
                "max_tokens": max_tokens,
                "temperature": getattr(args, "temperature", 1.0),
                "top_p": getattr(args, "top_p", 1.0),
            }

            server_idx = len(group_out) % num_servers
            server = manager.servers[server_idx]
            out = ray.get(
                server.generate.remote(
                    prompt=prompt_ids,
                    sampling_params=sampling_params,
                )
            )

            token_ids = out.get("token_ids", [])
            logprobs = out.get("logprobs")
            text = out.get("text", "")

            sample.tokens = list(prompt_ids) + list(token_ids)
            sample.response = text if text else tokenizer.decode(token_ids, skip_special_tokens=True)
            sample.response_length = len(token_ids)
            sample.rollout_log_probs = [float(x) for x in logprobs] if logprobs else None
            sample.status = Sample.Status.COMPLETED

            group_out.append(sample)
        results.append(group_out)

    manager.sleep_all()
    return RolloutFnTrainOutput(samples=results)


# ---------------------------------------------------------------------------
# Hook 4: --custom-megatron-post-save-hook-path
# Signature: (args, rollout_id: int, checkpoint_dir: str,
#             hf_checkpoint_dir: str) -> None
# ---------------------------------------------------------------------------

def custom_megatron_post_save_hook(
    args: Any,
    rollout_id: int,
    checkpoint_dir: str,
    hf_checkpoint_dir: str,
) -> None:
    """No-op by default; extend for custom checkpoint export."""
    logger.debug(
        "lumenrl: post_save_hook rollout_id=%d ckpt=%s",
        rollout_id,
        checkpoint_dir,
    )


# ---------------------------------------------------------------------------
# Hook 5: --custom-megatron-before-train-step-hook-path
# Signature: (args, rollout_id, step_id, model, optimizer,
#             opt_param_scheduler) -> None
# ---------------------------------------------------------------------------

def custom_megatron_before_train_step_hook(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Reset FP8 state before each training micro-step."""
    fp8_manager = getattr(args, "_lumenrl_fp8_manager", None)
    if fp8_manager is None:
        return

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        fp8_manager.reset_fp8_state(chunk)


# ---------------------------------------------------------------------------
# Hook 6: --custom-megatron-before-log-prob-hook-path
# Signature: (args, model, store_prefix: str) -> None
# ---------------------------------------------------------------------------

def custom_megatron_before_log_prob_hook(
    args: Any,
    model: Any,
    store_prefix: str,
) -> None:
    """Reset FP8 state before the forward-only log-prob pass."""
    fp8_manager = getattr(args, "_lumenrl_fp8_manager", None)
    if fp8_manager is None:
        return

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        fp8_manager.reset_fp8_state(chunk)


# ---------------------------------------------------------------------------
# Hook 7: --custom-update-weight-post-write-path
# Signature: (args, version_dir: str, rollout_engines: list) -> None
# Called by UpdateWeightFromDiskDelta after writing HF weights to disk.
# ---------------------------------------------------------------------------

def custom_update_weight_post_write(
    args: Any,
    version_dir: str,
    rollout_engines: list,
) -> None:
    """Reload ATOM server weights from the newly written HF checkpoint."""
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    manager = getattr(args, "_lumenrl_atom_manager", None)
    if manager is None:
        logger.debug("lumenrl: custom_update_weight_post_write: no ATOM manager, skipping.")
        return

    manager.reload_weights_from_path(version_dir)
    logger.info("lumenrl: ATOM weights reloaded from %s", version_dir)
