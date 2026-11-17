"""A sandbox service: task containers opened, run in and closed on request over HTTP.

It is the only process that holds the docker socket. The training containers reach it on an
internal network and can do three things: open a session from an allowed image, run a command
in it, close it. Every task container runs with no network, CPU / memory / process limits and
`no-new-privileges`; sessions not closed within `ttl_s` are reaped, and containers a previous
run of the service left behind are removed at start.

    python -m lumenrl.agent.swe.sandbox_server --host 172.30.0.11 --port 8090 \
        --image-pattern 'swebench/sweb\\.eval\\.x86_64\\..+:latest'

API (JSON):
    POST   /sessions                {"image", "env"?, "workdir"?}   -> {"id"}
    POST   /sessions/<id>/exec      {"command", "timeout_s"?, "workdir"?, "env"?, "output_cap_bytes"?}
                                    -> {"output", "exit_code", "timed_out", "elapsed_s", "total_bytes"}
    DELETE /sessions/<id>                                           -> {}
    GET    /health                                                  -> {"sessions"}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from lumenrl.agent.swe.docker_api import DEFAULT_SOCKET, DockerAPI

logger = logging.getLogger(__name__)

LABEL = "org.lumenrl.swe.session"


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Limits:
    image_pattern: str
    cpus: float = 4.0
    memory_bytes: int = 16 << 30
    pids_limit: int = 4096
    max_sessions: int = 256
    ttl_s: float = 4 * 3600.0
    max_exec_timeout_s: float = 3600.0
    default_exec_timeout_s: float = 60.0
    output_cap_bytes: int = 1 << 20
    max_output_cap_bytes: int = 64 << 20


@dataclass
class _Session:
    container: str
    image: str
    opened: float
    lock: threading.Lock = field(default_factory=threading.Lock)


class SandboxService:
    def __init__(self, docker: DockerAPI, limits: Limits):
        self._docker, self._limits = docker, limits
        self._pattern = re.compile(limits.image_pattern)
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def remove_leftovers(self) -> int:
        stale = self._docker.list_labelled(LABEL)
        for container in stale:
            self._docker.remove(container)
        return len(stale)

    def open(self, image: str, env: dict[str, str] | None = None, workdir: str = "/") -> str:
        if not isinstance(image, str) or not self._pattern.fullmatch(image):
            raise RequestError(403, f"image {image!r} is not allowed here")
        with self._lock:
            if len(self._sessions) >= self._limits.max_sessions:
                raise RequestError(429, f"{len(self._sessions)} sessions open, the limit")
            sid = uuid.uuid4().hex
            self._sessions[sid] = None  # holds the slot while the container starts
        try:
            if not self._docker.image_exists(image):
                self._docker.pull(image)
            container = self._docker.create(
                image, name=f"lumenrl-swe-{sid[:12]}", labels={LABEL: sid}, workdir=workdir, env=env or {},
                cpus=self._limits.cpus, memory_bytes=self._limits.memory_bytes,
                pids_limit=self._limits.pids_limit,
            )
        except BaseException:
            with self._lock:
                self._sessions.pop(sid, None)
            raise
        with self._lock:
            self._sessions[sid] = _Session(container, image, time.monotonic())
        return sid

    def _get(self, sid: str) -> _Session:
        with self._lock:
            session = self._sessions.get(sid)
        if session is None:
            raise RequestError(404, f"no session {sid}")
        return session

    def exec(self, sid: str, command: str, timeout_s: float | None = None, workdir: str | None = None,
             env: dict[str, str] | None = None, output_cap_bytes: int | None = None) -> dict[str, Any]:
        """Output past `output_cap_bytes` (default `Limits.output_cap_bytes`) keeps its head and
        tail; a caller reading a whole log asks for more, up to `max_output_cap_bytes`."""
        if not isinstance(command, str):
            raise RequestError(400, "command must be a string")
        timeout_s = float(self._limits.default_exec_timeout_s if timeout_s is None else timeout_s)
        if not 0 < timeout_s <= self._limits.max_exec_timeout_s:
            raise RequestError(400, f"timeout_s must be in (0, {self._limits.max_exec_timeout_s:g}]")
        cap = int(self._limits.output_cap_bytes if output_cap_bytes is None else output_cap_bytes)
        if not 0 < cap <= self._limits.max_output_cap_bytes:
            raise RequestError(400, f"output_cap_bytes must be in (0, {self._limits.max_output_cap_bytes}]")
        session = self._get(sid)
        with session.lock:  # one command at a time per sandbox, in arrival order
            out = self._docker.exec(session.container, command, timeout_s=timeout_s, workdir=workdir, env=env,
                                    output_cap=cap)
        return asdict(out)

    def close(self, sid: str) -> None:
        with self._lock:
            session = self._sessions.pop(sid, None)
        if session is None:
            raise RequestError(404, f"no session {sid}")
        self._docker.remove(session.container)

    def reap(self, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s is not None and now - s.opened > self._limits.ttl_s]
        for sid in expired:
            logger.warning("reaping session %s, open for more than %gs", sid, self._limits.ttl_s)
            try:
                self.close(sid)
            except RequestError:
                pass
        return expired

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


def make_handler(service: SandboxService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def _reply(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as e:
                raise RequestError(400, f"bad JSON: {e}") from e
            if not isinstance(body, dict):
                raise RequestError(400, "the body must be a JSON object")
            return body

        def _route(self, method: str) -> None:
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            try:
                if method == "GET" and parts == ["health"]:
                    return self._reply(200, {"sessions": service.count()})
                if method == "POST" and parts == ["sessions"]:
                    b = self._body()
                    sid = service.open(b.get("image"), b.get("env"), b.get("workdir") or "/")
                    return self._reply(200, {"id": sid})
                if method == "POST" and len(parts) == 3 and parts[0] == "sessions" and parts[2] == "exec":
                    b = self._body()
                    out = service.exec(parts[1], b.get("command"), b.get("timeout_s"), b.get("workdir"), b.get("env"),
                                       b.get("output_cap_bytes"))
                    return self._reply(200, out)
                if method == "DELETE" and len(parts) == 2 and parts[0] == "sessions":
                    service.close(parts[1])
                    return self._reply(200, {})
                raise RequestError(404, f"no route {method} {self.path}")
            except RequestError as e:
                self._reply(e.status, {"error": str(e)})
            except Exception as e:
                logger.exception("%s %s failed", method, self.path)
                self._reply(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_DELETE(self):
            self._route("DELETE")

    return Handler


def serve(service: SandboxService, host: str, port: int, reap_every_s: float = 60.0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.daemon_threads = True

    def reaper():
        while True:
            time.sleep(reap_every_s)
            try:
                service.reap()
            except Exception:
                logger.exception("reaping failed")

    threading.Thread(target=reaper, daemon=True, name="sandbox-reaper").start()
    return server


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--socket", default=DEFAULT_SOCKET)
    p.add_argument("--image-pattern", required=True, help="regex an image name must fully match")
    p.add_argument("--cpus", type=float, default=Limits.cpus)
    p.add_argument("--memory-gb", type=float, default=Limits.memory_bytes / (1 << 30))
    p.add_argument("--pids-limit", type=int, default=Limits.pids_limit)
    p.add_argument("--max-sessions", type=int, default=Limits.max_sessions)
    p.add_argument("--ttl-s", type=float, default=Limits.ttl_s)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    limits = Limits(a.image_pattern, a.cpus, int(a.memory_gb * (1 << 30)), a.pids_limit, a.max_sessions, a.ttl_s)
    service = SandboxService(DockerAPI(a.socket), limits)
    logger.info("removed %d leftover sandbox containers", service.remove_leftovers())
    server = serve(service, a.host, a.port)
    logger.info("sandbox service on %s:%d, images matching %r", a.host, a.port, a.image_pattern)
    server.serve_forever()


if __name__ == "__main__":
    main()
