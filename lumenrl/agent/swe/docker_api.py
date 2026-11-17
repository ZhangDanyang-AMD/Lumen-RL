"""The few Docker Engine API calls the sandbox service makes, over the daemon's unix socket.

Standard library only, so the service runs in any image with Python and the socket mounted
-- no docker CLI or SDK.
"""

from __future__ import annotations

import http.client
import json
import socket
import struct
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

DEFAULT_SOCKET = "/var/run/docker.sock"


class DockerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecOutput:
    output: str
    exit_code: int
    timed_out: bool
    elapsed_s: float
    total_bytes: int


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


class CappedBuffer:
    """Keeps the first and the last `cap // 2` bytes of a stream and counts the rest."""

    def __init__(self, cap: int):
        self._half = max(1, cap // 2)
        self._head = bytearray()
        self._tail = bytearray()
        self.total = 0

    def write(self, data: bytes) -> None:
        self.total += len(data)
        room = self._half - len(self._head)
        if room > 0:
            self._head += data[:room]
            data = data[room:]
        if data:
            self._tail += data
            if len(self._tail) > self._half:
                del self._tail[: len(self._tail) - self._half]

    def text(self) -> str:
        elided = self.total - len(self._head) - len(self._tail)
        body = bytes(self._head)
        if elided > 0:
            body += f"\n[... {elided} bytes elided ...]\n".encode()
        return (body + bytes(self._tail)).decode("utf-8", errors="replace")


def demux(stream, sink: CappedBuffer) -> None:
    """Copy a non-TTY exec stream (8-byte frame headers: stream id, 3 zero bytes, big-endian
    length) into `sink`, stdout and stderr interleaved as they arrive."""
    while True:
        header = _read_exactly(stream, 8)
        if not header:
            return
        (length,) = struct.unpack(">I", header[4:])
        while length:
            chunk = stream.read(min(length, 1 << 16))
            if not chunk:
                return
            sink.write(chunk)
            length -= len(chunk)


def _read_exactly(stream, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


class DockerAPI:
    def __init__(self, socket_path: str = DEFAULT_SOCKET, timeout_s: float = 120.0):
        self._socket = socket_path
        self._timeout = timeout_s

    def _request(self, method: str, path: str, body: Any = None, *, timeout: float | None = None,
                 expect=(200, 201, 204)) -> http.client.HTTPResponse:
        conn = _UnixConnection(self._socket, timeout or self._timeout)
        payload = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        if response.status not in expect:
            detail = response.read().decode("utf-8", errors="replace")
            conn.close()
            raise DockerError(f"{method} {path} -> {response.status}: {detail.strip()}")
        return response

    def _json(self, method: str, path: str, body: Any = None, **kw) -> Any:
        response = self._request(method, path, body, **kw)
        data = response.read()
        return json.loads(data) if data else None

    def image_exists(self, image: str) -> bool:
        try:
            self._json("GET", f"/images/{urllib.parse.quote(image, safe='')}/json")
            return True
        except DockerError as e:
            if "-> 404" in str(e):
                return False
            raise

    def pull(self, image: str, timeout_s: float = 1800.0) -> None:
        name, _, tag = image.rpartition(":") if ":" in image.rsplit("/", 1)[-1] else (image, "", "latest")
        query = urllib.parse.urlencode({"fromImage": name, "tag": tag})
        response = self._request("POST", f"/images/create?{query}", timeout=timeout_s)
        for line in response.read().decode("utf-8", errors="replace").splitlines():
            if line.strip() and "error" in (event := json.loads(line)):
                raise DockerError(f"pulling {image}: {event['error']}")

    def create(self, image: str, *, name: str, labels: dict[str, str], workdir: str, env: dict[str, str],
               cpus: float, memory_bytes: int, pids_limit: int) -> str:
        body = {
            "Image": image,
            "Entrypoint": ["sleep"],
            "Cmd": ["infinity"],
            "Labels": labels,
            "WorkingDir": workdir,
            "Env": [f"{k}={v}" for k, v in env.items()],
            "HostConfig": {
                "NetworkMode": "none",
                "NanoCpus": int(cpus * 1e9),
                "Memory": int(memory_bytes),
                "PidsLimit": int(pids_limit),
                "SecurityOpt": ["no-new-privileges"],
                "Init": True,
            },
        }
        created = self._json("POST", f"/containers/create?name={urllib.parse.quote(name)}", body)
        self._request("POST", f"/containers/{created['Id']}/start").read()
        return created["Id"]

    def exec(self, container: str, command: str, *, timeout_s: float, workdir: str | None = None,
             env: dict[str, str] | None = None, output_cap: int = 1 << 20) -> ExecOutput:
        """Run `command` with `bash -c` in `container`. The container's own `timeout` kills it
        after `timeout_s`, so a timed-out command does not keep running in the sandbox."""
        cmd = ["timeout", "-s", "KILL", f"{timeout_s:g}", "bash", "-c", command]
        body = {"AttachStdout": True, "AttachStderr": True, "Tty": False, "Cmd": cmd,
                "Env": [f"{k}={v}" for k, v in (env or {}).items()]}
        if workdir:
            body["WorkingDir"] = workdir
        exec_id = self._json("POST", f"/containers/{container}/exec", body)["Id"]
        sink = CappedBuffer(output_cap)
        t0 = time.monotonic()
        response = self._request("POST", f"/exec/{exec_id}/start", {"Detach": False, "Tty": False},
                                 timeout=timeout_s + 30)
        demux(response, sink)
        elapsed = time.monotonic() - t0
        code = self._json("GET", f"/exec/{exec_id}/json").get("ExitCode")
        exit_code = -1 if code is None else int(code)
        timed_out = exit_code == 137 and elapsed >= 0.98 * timeout_s
        return ExecOutput(sink.text(), exit_code, timed_out, elapsed, sink.total)

    def remove(self, container: str) -> None:
        self._request("DELETE", f"/containers/{container}?force=true&v=true", expect=(200, 204, 404)).read()

    def list_labelled(self, label: str) -> list[str]:
        filters = urllib.parse.quote(json.dumps({"label": [label]}))
        return [c["Id"] for c in self._json("GET", f"/containers/json?all=true&filters={filters}")]
