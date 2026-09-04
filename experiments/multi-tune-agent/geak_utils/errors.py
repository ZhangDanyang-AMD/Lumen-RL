"""Exceptions raised by GEAK utility adapters."""


class SandboxError(RuntimeError):
    """Raised when a kernel task cannot be prepared or evaluated safely."""
