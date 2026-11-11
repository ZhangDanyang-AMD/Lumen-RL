"""Fused actor/ref workers must export prefixed methods Ray can register."""

from lumenrl.controller.colocation import create_fused_worker_cls


class _Actor:
    def __init__(self, rank: int, world_size: int, **kwargs) -> None:
        self.rank = rank
        self.world_size = world_size

    def get_node_ip(self) -> str:
        return f"actor-{self.rank}"

    def setup_distributed(self, rank: int) -> int:
        return rank + self.world_size


class _Ref:
    def __init__(self, rank: int, world_size: int, **kwargs) -> None:
        self.rank = rank

    def init_model(self) -> str:
        return f"ref-{self.rank}"


def test_prefixed_methods_are_real_class_attributes():
    cls = create_fused_worker_cls({"actor": _Actor, "ref": _Ref}, name="ActorRefFusedWorker")
    assert "actor_get_node_ip" in cls.__dict__
    assert "actor_setup_distributed" in cls.__dict__
    assert "ref_init_model" in cls.__dict__
    assert "get_node_ip" not in cls.__dict__

    worker = cls(rank=0, world_size=8)
    assert worker.actor_get_node_ip() == "actor-0"
    assert worker.actor_setup_distributed(3) == 11
    assert worker.ref_init_model() == "ref-0"


def test_ray_handle_exports_prefixed_method_only():
    import ray

    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True, logging_level="ERROR")
    try:
        cls = create_fused_worker_cls({"actor": _Actor, "ref": _Ref}, name="ActorRefFusedWorker")
        handle = ray.remote(num_cpus=1)(cls).remote(rank=1, world_size=2)
        assert ray.get(handle.actor_get_node_ip.remote()) == "actor-1"
        missing = getattr(handle, "_ray_method_signatures", {})
        assert "get_node_ip" not in missing
        try:
            handle.get_node_ip
        except AttributeError as exc:
            assert "get_node_ip" in str(exc)
        else:
            raise AssertionError("unprefixed get_node_ip must not be on the actor handle")
    finally:
        ray.shutdown()
