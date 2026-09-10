"""The engine kwargs ATOMRayServer supplies rather than inherit from ATOM.

Both pins exist because an ATOM default changed under a pin bump and the release
measurements were taken against the old one. They are only meaningful together:
each applies exactly when torch.compile is on, and neither may fire for an eager
rollout.
"""

from __future__ import annotations

from lumenrl.engine.inference.atom_ray_server import ATOMRayServer


def _server() -> ATOMRayServer:
    return ATOMRayServer(model_name="/models/Qwen3-8B-Base", engine_kwargs={}, replica_rank=0)


def test_sleep_stays_resident_for_a_no_eager_rollout() -> None:
    # Releasing makes ATOM re-derive the KV block count on every wake, charging
    # the colocated trainer against the budget. Qwen3-30B-A3B then asks for a
    # negative pool and asserts in resume_memory.
    for kwargs in (
        {"enforce_eager": False},
        {"compilation_config": {"level": 3}},
    ):
        _server()._pin_sleep_keeps_memory_resident(kwargs)
        assert kwargs["sleep_keeps_memory_resident"] is True


def test_sleep_pin_is_absent_for_an_eager_rollout() -> None:
    # No graphs to keep valid, and ATOM ignores the field under enforce_eager
    # anyway -- so do not claim to have set something that does nothing.
    kwargs: dict = {"enforce_eager": True}
    _server()._pin_sleep_keeps_memory_resident(kwargs)
    assert "sleep_keeps_memory_resident" not in kwargs

    kwargs = {}
    _server()._pin_sleep_keeps_memory_resident(kwargs)
    assert "sleep_keeps_memory_resident" not in kwargs


def test_an_explicit_setting_wins() -> None:
    kwargs = {"enforce_eager": False, "sleep_keeps_memory_resident": False}
    _server()._pin_sleep_keeps_memory_resident(kwargs)
    assert kwargs["sleep_keeps_memory_resident"] is False


def test_the_shared_gate_reads_both_ways_of_asking_for_torch_compile() -> None:
    # run_dapo.sh passes enforce_eager=false *and* compilation_config.level=3 for
    # ATOM FP8; either alone still means graphs will be captured.
    for kwargs, expected in (
        ({"enforce_eager": False}, True),
        ({"compilation_config": {"level": 3}}, True),
        ({"enforce_eager": False, "compilation_config": {"level": 3}}, True),
        ({"compilation_config": {"level": 0}}, False),
        ({"compilation_config": {"level": None}}, False),
        ({"enforce_eager": True}, False),
        ({}, False),
    ):
        assert ATOMRayServer._is_no_eager(kwargs) is expected, kwargs
