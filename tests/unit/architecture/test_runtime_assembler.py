from __future__ import annotations

from lumenrl.architecture.assembly.runtime_assembler import RuntimeAssembler
from lumenrl.core.config import LumenRLConfig


def test_assembler_resolves_default_actor_worker() -> None:
    cfg = LumenRLConfig()
    assembler = RuntimeAssembler(cfg)
    graph = assembler.build_graph()
    assert "actor_worker_cls" in graph
    assert graph["actor_worker_cls"].__name__ == "LumenActorWorker"
