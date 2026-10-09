"""Backend registry for sandbox implementations."""

from __future__ import annotations

from typing import Any

from .base import SandboxBackend

_BACKENDS: dict[str, type[SandboxBackend]] = {}


def register_backend(name: str, cls: type[SandboxBackend]) -> None:
    _BACKENDS[name] = cls


def get_backend(name: str = "geak", **kwargs: Any) -> SandboxBackend:
    if name not in _BACKENDS:
        if name == "geak":
            from .backends import geak as _  # noqa: F811
        if name not in _BACKENDS:
            available = ", ".join(sorted(_BACKENDS)) or "(none)"
            raise KeyError(
                f"sandbox backend {name!r} not registered; available: {available}"
            )
    return _BACKENDS[name](**kwargs)


def list_backends() -> list[str]:
    return sorted(_BACKENDS)
