from __future__ import annotations

from lumenrl.architecture.assembly.runtime_assembler import RuntimeAssembler
from lumenrl.core.config import LumenRLConfig


def test_async_flag_selects_async_trainer() -> None:
    cfg = LumenRLConfig()
    cfg.async_training.enabled = True
    assembler = RuntimeAssembler(cfg)
    graph = assembler.build_graph()
    assert graph["trainer_cls"].__name__ == "AsyncRLTrainer"
