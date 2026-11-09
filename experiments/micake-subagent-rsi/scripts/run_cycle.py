"""Train role LoRA from the recorded generation, serve the checkpoints, roll the challenger.

Stops the current specialist ATOM server first: the LoRA trainer needs all
eight GPUs. After the merge, Planner and Critic are served on separate endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
GEAK_ROOT = Path("/home/danyzhan/Lumen/experiments/GEAK-agent-coder")
for entry in (REPO_ROOT, REPO_ROOT / "experiments" / "sandbox", EXPERIMENT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from micake_rsi.cycle import run_cycle  # noqa: E402
from micake_rsi.loop import run_generation  # noqa: E402

LOGGER = logging.getLogger(__name__)
TRAIN_PYTHONPATH = ":".join([
    "/home/danyzhan/Lumen/third_party/aiter",
    "/home/danyzhan/Lumen",
    str(GEAK_ROOT / "src"),
])

SERVE_PYTHONPATH = "/home/danyzhan/ATOM:/home/danyzhan/Lumen/third_party/aiter"
SERVE_AITER_JIT_DIR = "/home/danyzhan/.aiter-micake-serve"


def _docker(image: str, name: str, script: str, gpus: str | None = None) -> None:
    command = [
        "docker", "run", "--rm", "--name", name, "--network", "host",
        "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video",
        "--ipc", "host", "--shm-size", "64g", "-v", "/home/danyzhan:/home/danyzhan",
        "-e", f"PYTHONPATH={TRAIN_PYTHONPATH}", "-e", "PYTORCH_ROCM_ARCH=gfx950",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-e", "SONIC_MOE_GEMM_BACKEND=triton", "-e", "SONIC_MOE_GROUPED_GEMM_BACKEND=triton",
    ]
    if gpus is not None:
        command.extend(["-e", f"HIP_VISIBLE_DEVICES={gpus}"])
    command.extend(["--entrypoint", "bash", image, "-lc", script])
    LOGGER.info("running %s", name)
    subprocess.run(command, check=True)


def _chown_on_exit(path: Path) -> str:
    return f"trap 'chown -R {os.getuid()}:{os.getgid()} {path} 2>/dev/null || true' EXIT"


def _train(config_path: Path, image: str, nproc: int) -> None:
    _docker(image, f"micake-rsi-train-{config_path.parent.name}", f"""
set -euo pipefail
{_chown_on_exit(config_path.parent.resolve())}
cd {GEAK_ROOT}
python3 -m torch.distributed.run --nnodes=1 --nproc-per-node={nproc} --standalone \
  -m geak_agent_coder.sft.train --config {config_path.resolve()}
""")


def _merge(image: str, base: Path, adapter: Path, output: Path) -> None:
    _docker(image, f"micake-rsi-merge-{output.name}", f"""
set -euo pipefail
{_chown_on_exit(output.resolve().parent.parent)}
cd {GEAK_ROOT}
python3 scripts/merge_to_hf.py --base-model {base} --adapter-dir {adapter} --output-dir {output}
""")


def _served_model(port: int) -> str | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
            return json.loads(response.read())["data"][0]["id"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def _serve(image: str, update: dict) -> None:
    name = f"atom-micake-{update['role']}"
    model = update["checkpoint"]
    if _served_model(update["port"]) == model:
        LOGGER.info("%s already serves %s on port %s", update["role"], model, update["port"])
        return
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    command = [
        "docker", "run", "-d", "--name", name, "--device", "/dev/kfd", "--device", "/dev/dri",
        "--group-add", "video", "--ipc", "host", "--network", "host", "--shm-size", "32g",
        "-v", "/home/danyzhan:/home/danyzhan", "-e", f"PYTHONPATH={SERVE_PYTHONPATH}", "-e", f"AITER_JIT_DIR={SERVE_AITER_JIT_DIR}",
        "-e", f"HIP_VISIBLE_DEVICES={update['gpus']}", "--entrypoint", "python", image,
        "-m", "atom.entrypoints.openai.api_server", "--model", model, "--trust-remote-code",
        "-tp", str(update["tp"]), "--max-model-len", "32768", "--server-port", str(update["port"]),
        "--gpu-memory-utilization", "0.85", "--enforce-eager", "--level", "0",
    ]
    LOGGER.info("serving %s on port %s", update["role"], update["port"])
    subprocess.run(command, check=True)
    deadline = time.monotonic() + 900
    url = f"http://127.0.0.1:{update['port']}/v1/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    LOGGER.info("%s is accepting requests", update["role"])
                    return
        except OSError:
            time.sleep(10)
    raise TimeoutError(f"{update['role']} did not become ready on port {update['port']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["rsi"]["root"] = str(Path(cfg["rsi"]["root"]).resolve())
    train = cfg["train"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(train["base_model"], trust_remote_code=True)
    subprocess.run(["docker", "rm", "-f", "atom-qwen3-specialist"], check=False, capture_output=True)
    manifest = run_cycle(
        cfg,
        tokenizer,
        train_role=lambda path: _train(path, train["image"], int(train["nproc_per_node"])),
        merge_role=lambda base, adapter, output: _merge(train["image"], base, adapter, output),
        serve=lambda update: _serve(train["serve"]["image"], update),
        rollout=lambda challenger: asyncio.run(run_generation(challenger)),
    )
    LOGGER.info("promotion: %s", manifest["promotion"])


if __name__ == "__main__":
    main()
