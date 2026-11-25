"""FSDP2 sharding returns the full-model staging memory to the device.

_apply_fsdp2_sharding moves the whole model to the GPU before fully_shard. The
full tensors are freed once the parameters become shards, but the caching
allocator keeps them: on Qwen3-30B-A3B each actor held 123 GB through rollout,
and with colocated vLLM (63 GB) the first training update OOMed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lumenrl.engine.training import fsdp_backend


class _M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


def test_cache_is_emptied_after_the_root_is_sharded(monkeypatch) -> None:
    events: list[str] = []
    import torch.distributed as dist
    import torch.distributed._composable.fsdp as fsdp_mod
    import lumenrl.engine.training.fsdp_chunk_cat_fallback as cc

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(cc, "install_chunk_cat_fallback", lambda: None)
    monkeypatch.setattr(nn.Module, "to", lambda self, *a, **k: self)
    monkeypatch.setattr(fsdp_mod, "fully_shard",
                        lambda m, **k: events.append("root" if isinstance(m, _M) else "layer"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))

    fsdp_backend._apply_fsdp2_sharding(_M(), {})

    assert "empty_cache" in events
    assert events.index("empty_cache") > events.index("root")
