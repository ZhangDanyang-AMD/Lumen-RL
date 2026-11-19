#!/usr/bin/env python3
"""Compare the merged MMKCA planner with DSV4.1 on held-out Qwen3-30B-A3B GEMMs.

One episode is one regime and one dtype. Both arms see the same prompt, tool schema and submit budget.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import yaml

from run_ab_eval import DEFAULT_CONFIG, REPO_ROOT
from run_cycle import _serve

CHECKPOINT = REPO_ROOT / "output" / "mmkca-planner" / "g1" / "checkpoint"
DATA = REPO_ROOT / "output" / "mmkca-planner" / "heldout-g1"
MMKCA = Path("/home/danyzhan/MICake-MMKCA")
KEY_FILE = Path("/home/danyzhan/DSV4-API.key")
CELLS = (("decode", "bf16"), ("decode", "fp8"), ("prefill", "bf16"), ("prefill", "fp8"))
EVAL_GPUS = ("0", "1", "2", "3")


def _gid(path: str) -> str:
    return str(os.stat(path).st_gid)


def wait_for_checkpoint(path: Path, log: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (path / "merge_manifest.json").exists():
            return
        if log.exists() and "train exit" in log.read_text(encoding="utf-8", errors="replace"):
            raise SystemExit(f"training finished without {path / 'merge_manifest.json'}")
        time.sleep(30)
    raise SystemExit(f"checkpoint was not merged within {timeout_s:.0f}s")


def launch(gpu: str, arm: str, portfolio: str, dtype: str, base_url: str, model: str, repeats: int, budget: int) -> subprocess.Popen:
    out = DATA / f"{arm}-{portfolio}-{dtype}.jsonl"
    command = [
        "docker", "run", "--rm", "--network", "host", "-e", f"HIP_VISIBLE_DEVICES={gpu}",
        "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
        "--device", "/dev/kfd", "--device", "/dev/dri",
        "--group-add", _gid("/dev/kfd"), "--group-add", _gid("/dev/dri/renderD128"),
        "--security-opt", "seccomp=unconfined",
        "-v", f"{MMKCA}:/workspace/MMKCA", "-v", "/opt/rocm-7.3.0:/opt/rocm-7.3.0:ro",
        "-v", f"{KEY_FILE}:{KEY_FILE}:ro", "-v", f"{DATA}:{DATA}",
        "-e", "MMKCA_ROCM_LLVM=/opt/rocm-7.3.0/llvm/bin", "-w", "/workspace/MMKCA",
        "mmkca-gpu:rocm7.2.4", "python3", "-u", "scripts/eval_planner.py",
        "--portfolio", portfolio, "--dtype", dtype, "--budget", str(budget), "--repeats", str(repeats),
        "--base-url", base_url, "--model", model, "--out", str(out),
    ]
    if arm == "dsv4":
        command.extend(["--api-key-file", str(KEY_FILE)])
    log = open(DATA / f"{arm}-{portfolio}-{dtype}.log", "w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)


def run_arm(arm: str, base_url: str, model: str, repeats: int, budget: int) -> None:
    procs = [launch(gpu, arm, portfolio, dtype, base_url, model, repeats, budget)
             for gpu, (portfolio, dtype) in zip(EVAL_GPUS, CELLS)]
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        raise SystemExit(f"{arm} eval failed: {codes}")


def _bests(path: Path) -> list[float]:
    if not path.exists():
        return []
    return [json.loads(line)["best_speedup"] for line in path.read_text().splitlines() if line.strip()]


def summarize() -> None:
    print(f"{'cell':<22}{'trained':>10}{'dsv4':>10}")
    trained_all, dsv4_all = [], []
    for portfolio, dtype in CELLS:
        trained = _bests(DATA / f"trained-{portfolio}-{dtype}.jsonl")
        dsv4 = _bests(DATA / f"dsv4-{portfolio}-{dtype}.jsonl")
        trained_all.extend(trained)
        dsv4_all.extend(dsv4)
        def fmt(values: list[float]) -> str:
            return f"{statistics.mean(values):.4f}" if values else "—"
        print(f"{portfolio + ' ' + dtype:<22}{fmt(trained):>10}{fmt(dsv4):>10}")
    if trained_all and dsv4_all:
        print(f"{'mean of episodes':<22}{statistics.mean(trained_all):>10.4f}{statistics.mean(dsv4_all):>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--wait-s", type=float, default=4 * 3600)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()
    log = REPO_ROOT / "output" / "mmkca-planner" / "train-g1.log"
    if not args.skip_wait:
        wait_for_checkpoint(CHECKPOINT, log, args.wait_s)
    DATA.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    _serve(cfg["train"]["serve"]["image"], {"role": "mmkca-g1", "checkpoint": str(CHECKPOINT), "port": 8108, "gpus": "4", "tp": 1})
    run_arm("trained", "http://127.0.0.1:8108/v1", str(CHECKPOINT), args.repeats, args.budget)
    run_arm("dsv4", "https://127.0.0.1:8443/v1", "deepseek-ai/DeepSeek-V4.1-Flash", args.repeats, args.budget)
    summarize()


if __name__ == "__main__":
    main()
