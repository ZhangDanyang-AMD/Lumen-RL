"""Async client of the sandbox service (`sandbox_server`)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

# (method, url, JSON body or None, timeout s) -> (HTTP status, JSON body)
RequestFn = Callable[[str, str, "dict[str, Any] | None", float], Awaitable["tuple[int, dict[str, Any]]"]]


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecResult:
    output: str
    exit_code: int
    timed_out: bool
    elapsed_s: float = 0.0


class SandboxClient:
    def __init__(self, url: str, *, request: RequestFn | None = None, slack_s: float = 60.0) -> None:
        self._url = url.rstrip("/")
        self._request = request or self._aiohttp_request
        self._slack_s = float(slack_s)
        self._http: Any = None

    async def open(self, image: str, *, env: dict[str, str] | None = None, workdir: str = "/",
                   timeout_s: float = 1800.0) -> SandboxSession:
        """A sandbox from `image`; the first open of an image on a node pulls it, hence the timeout."""
        body = await self._call("POST", "/sessions", {"image": image, "env": env or {}, "workdir": workdir}, timeout_s)
        return SandboxSession(self, body["id"])

    async def _call(self, method: str, path: str, body: dict[str, Any] | None, timeout_s: float) -> dict[str, Any]:
        status, reply = await self._request(method, self._url + path, body, timeout_s)
        if status != 200:
            raise SandboxError(f"{method} {path}: {status} {reply.get('error', reply)}")
        return reply

    async def _aiohttp_request(self, method: str, url: str, body: dict[str, Any] | None,
                               timeout_s: float) -> tuple[int, dict[str, Any]]:
        import aiohttp

        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        async with self._http.request(method, url, json=body, timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            return r.status, await r.json(content_type=None)

    async def close(self) -> None:
        if self._http is not None and not self._http.closed:
            await self._http.close()


class SandboxSession:
    def __init__(self, client: SandboxClient, sid: str) -> None:
        self._client, self.id = client, sid
        self._closed = False

    async def run(self, command: str, timeout_s: float = 60.0, *, workdir: str | None = None,
                  env: dict[str, str] | None = None, output_cap_bytes: int | None = None) -> ExecResult:
        body = {"command": command, "timeout_s": timeout_s, "workdir": workdir, "env": env or {},
                "output_cap_bytes": output_cap_bytes}
        reply = await self._client._call("POST", f"/sessions/{self.id}/exec", body, timeout_s + self._client._slack_s)
        return ExecResult(reply["output"], int(reply["exit_code"]), bool(reply["timed_out"]),
                          float(reply.get("elapsed_s", 0.0)))

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._client._call("DELETE", f"/sessions/{self.id}", None, 120.0)
