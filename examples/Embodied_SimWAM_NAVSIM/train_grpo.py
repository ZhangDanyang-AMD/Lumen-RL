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

from examples.Embodied_SimWAM_NAVSIM.plugin import install_grpo_hook
from simwam.runtime_grpo import run_grpo_training
from simwam.utils.config_resolvers import register_default_resolvers

install_grpo_hook()
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
