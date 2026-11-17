import asyncio
import threading
import time

import pytest

from lumenrl.agent.swe.docker_api import ExecOutput
from lumenrl.agent.swe.sandbox_client import SandboxClient, SandboxError
from lumenrl.agent.swe.sandbox_server import (
    LABEL,
    Limits,
    RequestError,
    SandboxService,
    serve,
)


class FakeDocker:
    def __init__(self, images=("swebench/a:latest",)):
        self.images = set(images)
        self.pulled, self.removed, self.execs = [], [], []
        self.containers = {}
        self.running = 0
        self.max_running = 0

    def image_exists(self, image):
        return image in self.images

    def pull(self, image):
        self.pulled.append(image)
        self.images.add(image)

    def create(self, image, *, name, labels, workdir, env, cpus, memory_bytes, pids_limit):
        cid = f"c{len(self.containers)}"
        self.containers[cid] = {"image": image, "labels": labels, "workdir": workdir, "cpus": cpus}
        return cid

    def exec(self, container, command, *, timeout_s, workdir, env, output_cap):
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        time.sleep(0.01)
        self.running -= 1
        self.execs.append((container, command, timeout_s, workdir, env, output_cap))
        return ExecOutput(f"ran {command}", 3, False, 0.01, 7)

    def remove(self, container):
        self.removed.append(container)
        self.containers.pop(container, None)

    def list_labelled(self, label):
        return [c for c, v in self.containers.items() if label in v["labels"]]


def _service(docker=None, **limits):
    return SandboxService(docker or FakeDocker(), Limits(r"swebench/.+:latest", **limits))


def test_only_allowed_images_open():
    service = _service()
    with pytest.raises(RequestError, match="not allowed") as e:
        service.open("evil/image:latest")
    assert e.value.status == 403


def test_a_missing_image_is_pulled_and_the_container_labelled():
    docker = FakeDocker(images=())
    service = _service(docker)

    sid = service.open("swebench/b:latest", workdir="/testbed")

    assert docker.pulled == ["swebench/b:latest"]
    (info,) = docker.containers.values()
    assert info["labels"] == {LABEL: sid} and info["workdir"] == "/testbed"


def test_the_session_limit_holds():
    service = _service(max_sessions=1)
    service.open("swebench/a:latest")
    with pytest.raises(RequestError, match="limit") as e:
        service.open("swebench/a:latest")
    assert e.value.status == 429


def test_commands_in_one_sandbox_run_one_at_a_time():
    docker = FakeDocker()
    service = _service(docker)
    sid = service.open("swebench/a:latest")
    threads = [threading.Thread(target=service.exec, args=(sid, f"cmd {i}")) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert docker.max_running == 1 and len(docker.execs) == 8


def test_timeouts_and_output_caps_are_bounded():
    service = _service(max_exec_timeout_s=100, max_output_cap_bytes=1000)
    sid = service.open("swebench/a:latest")
    with pytest.raises(RequestError, match="timeout_s"):
        service.exec(sid, "ls", timeout_s=101)
    with pytest.raises(RequestError, match="output_cap_bytes"):
        service.exec(sid, "ls", output_cap_bytes=1001)


def test_close_removes_the_container_and_forgets_the_session():
    docker = FakeDocker()
    service = _service(docker)
    sid = service.open("swebench/a:latest")
    service.close(sid)

    assert docker.removed == ["c0"] and service.count() == 0
    with pytest.raises(RequestError) as e:
        service.exec(sid, "ls")
    assert e.value.status == 404


def test_sessions_past_their_ttl_are_reaped():
    docker = FakeDocker()
    service = _service(docker, ttl_s=10)
    old = service.open("swebench/a:latest")

    assert service.reap(now=time.monotonic() + 5) == []
    assert service.reap(now=time.monotonic() + 11) == [old] and docker.removed == ["c0"]


def test_leftovers_of_an_earlier_run_are_removed():
    docker = FakeDocker()
    docker.containers = {"old": {"labels": {LABEL: "x"}}, "other": {"labels": {}}}

    assert _service(docker).remove_leftovers() == 1 and docker.removed == ["old"]


def test_the_client_drives_the_service_over_http():
    docker = FakeDocker()
    server = serve(_service(docker), "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    async def scenario():
        client = SandboxClient(url)
        try:
            session = await client.open("swebench/a:latest", workdir="/testbed")
            result = await session.run("echo hi", 30, workdir="/testbed", env={"A": "1"}, output_cap_bytes=4096)
            await session.close()
            with pytest.raises(SandboxError, match="403"):
                await client.open("nope:latest")
            return result
        finally:
            await client.close()

    try:
        result = asyncio.run(scenario())
    finally:
        server.shutdown()
    assert result.output == "ran echo hi" and result.exit_code == 3 and not result.timed_out
    assert docker.execs[0][1:] == ("echo hi", 30.0, "/testbed", {"A": "1"}, 4096)
    assert docker.removed == ["c0"]
