"""FlowGRPO entry that applies Lumen without editing the SimWAM tree.

Same Hydra config as ``examples/SimWAM/scripts/train_grpo.py``. ``LUMEN_ENABLE``
is read inside the ``configure_grpo`` hook (see ``plugin.install_grpo_hook``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_SIMWAM = Path(os.environ.get("SIMWAM_ROOT", _REPO / "examples" / "SimWAM"))
_NAVSIM = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", _SIMWAM / "navsim"))
for _path in (str(_REPO), str(_SIMWAM / "src"), str(_NAVSIM)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import hydra
from omegaconf import DictConfig

from examples.Embodied_SimWAM_NAVSIM.plugin import (
    install_action_velocity_graphs,
    install_aiter_rmsnorm,
    install_batched_logprobs,
    install_compiled_rmsnorm,
    install_drop_true_masks,
    install_foreach_grad_checks,
    install_fused_modulate,
    install_fused_lora_add,
    install_grpo_hook,
    install_lean_lumen_attention,
    install_lora_init_seed,
    install_manual_data_parallel,
    install_numa_affinity,
    install_overlap_reward_logprobs,
    install_parallel_pdm_reward,
    install_prefill_graph,
    install_ref_logprob_cache,
    install_train_graphs,
    install_triton_rope,
)
from simwam.runtime_grpo import run_grpo_training
from simwam.utils.config_resolvers import register_default_resolvers

install_grpo_hook()
# before the trainer is built; decides whether DeepSpeed is used
install_manual_data_parallel()
install_lora_init_seed()
# before the reference cache, which wraps action_chain_logprobs
install_batched_logprobs()
install_ref_logprob_cache()
# before the CUDA graphs, which capture its attention path
install_lean_lumen_attention()
# after the lean attention (its SDPA wrapper must be outermost), before the graphs that capture it
install_drop_true_masks()
# outside the mask check of the prefill, which must not run under capture
install_prefill_graph()
install_action_velocity_graphs()
# after the no_grad graphs, which it falls through to
install_train_graphs()
install_parallel_pdm_reward()
install_compiled_rmsnorm()
install_aiter_rmsnorm()
install_triton_rope()
install_fused_modulate()
install_fused_lora_add()
install_overlap_reward_logprobs()
install_foreach_grad_checks()
install_numa_affinity()
register_default_resolvers()


@hydra.main(
    config_path=str(_SIMWAM / "configs"),
    config_name="train_grpo",
    version_base="1.3",
)
def main(cfg: DictConfig):
    run_grpo_training(cfg)


if __name__ == "__main__":
    main()
