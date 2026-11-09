"""Planner-only RSI loop: every generation continues from the champion Planner.

Manager and Critic are DSV4.1 in every arm. One generation:

1. build:  improvement targets (``new_best`` turns + MICake-measured exploit
           corrections) from all end-to-end experience so far;
2. train:  LoRA SFT on top of the merged champion checkpoint, merged into the
           challenger checkpoint;
3. eval:   interleaved end-to-end episodes champion / challenger / DSV4.1
           Planner; the episodes join the experience pool;
4. gate:   ``planner_gate.gate_decision`` promotes or holds; the lineage file
           records every decision and the comparison against DSV4.1.

State lives in ``<root>/lineage.json``; rerunning resumes the unfinished stage.

Example:
    export DSV4_API_KEY=...   # read from the key file, never printed
    python experiments/micake-subagent-rsi/scripts/rsi_planner.py --generations 3
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_ab_eval import DEFAULT_CONFIG, REPO_ROOT  # noqa: E402
from run_cycle import _merge, _serve, _train  # noqa: E402
from train_rft import DATA_PARALLEL, specialist_containers  # noqa: E402
from micake_rsi.cycle import MERGE_MANIFEST, training_config  # noqa: E402
from micake_rsi.planner_gate import PlannerGate, gate_decision, versus  # noqa: E402
from micake_rsi.sft_data import write_role_sft  # noqa: E402

logger = logging.getLogger("rsi_planner")

MICAKE_GPUS = (0, 1, 6, 7)
CHAMPION_SERVE = (8108, "4")
CHALLENGER_SERVE = (8111, "3")
SEED_EXPERIENCE = ("output/micake-subagent-rsi/e2e_v1", "output/micake-subagent-rsi/e2e_v2")
SEED_CACHES = ("output/micake-subagent-rsi/corrections_v3/*.cache.json", "output/micake-subagent-rsi/corrections_gen7/*.cache.json")


class Lineage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "lineage.json"
        root.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.path)


def run_parallel(commands: list[list[str]], logs: list[Path]) -> None:
    procs = []
    for command, log in zip(commands, logs):
        log.parent.mkdir(parents=True, exist_ok=True)
        procs.append(subprocess.Popen(command, cwd=REPO_ROOT, stdout=log.open("w"), stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"parallel job failed: exit codes {codes}; see {logs}")


def run_e2e(out: Path, arms: dict[str, tuple[int, str]], samples_per_shard: int) -> list[dict]:
    """Interleaved end-to-end episodes for ``arms`` plus the all-DSV4.1 arm, sharded over the MICake GPUs."""
    names = ",".join([*arms, "dsv4"])
    planner_args = [x for name, (port, ckpt) in arms.items() for x in ("--planner-arm", f"{name}={port}:{ckpt}")]
    commands, logs = [], []
    for i, gpu in enumerate(MICAKE_GPUS):
        commands.append([sys.executable, str(SCRIPTS_DIR / "run_ab_eval.py"), "--study", "e2e", "--arms", names,
                         *planner_args, "--fallback", "dsv4", "--samples", str(samples_per_shard),
                         "--sample-start", str(i * samples_per_shard), "--gpu", str(gpu),
                         "--out", str(out), "--out-name", f"e2e_s{i}"])
        logs.append(out / f"log_s{i}.txt")
    run_parallel(commands, logs)
    return load_records(out)


def load_records(directory: Path) -> list[dict]:
    return [json.loads(line) for path in sorted(glob.glob(str(directory / "e2e*.jsonl")))
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build(gen_dir: Path, experience: list[str]) -> Path:
    data = gen_dir / "data"
    caches = [p for pattern in SEED_CACHES for p in glob.glob(str(REPO_ROOT / pattern))]
    caches += glob.glob(str(gen_dir.parent / "*" / "data" / "*.cache.json"))
    commands, logs = [], []
    for i, gpu in enumerate(MICAKE_GPUS):
        command = [sys.executable, str(SCRIPTS_DIR / "build_improvements.py"), "--gpu", str(gpu), "--shard", str(i),
                   "--shards", str(len(MICAKE_GPUS)), "--out", str(data / f"s{i}.jsonl")]
        for directory in experience:
            command += ["--exp-dir", str(REPO_ROOT / directory)]
        for cache in caches:
            command += ["--cache-in", cache]
        commands.append(command)
        logs.append(data / f"log_s{i}.txt")
    run_parallel(commands, logs)
    merged = data / "planner.jsonl"
    merged.write_text("".join(Path(data / f"s{i}.jsonl").read_text() for i in range(len(MICAKE_GPUS))))
    return merged


def train(cfg: dict, gen_dir: Path, samples_path: Path, champion: dict, epochs: float) -> dict:
    train_cfg = {**cfg["train"], "base_model": champion["checkpoint"]}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(champion["checkpoint"], trust_remote_code=True)
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data_dir = gen_dir / "train" / "planner"
    manifest = write_role_sft(samples, tokenizer, data_dir, min_advantage=0.0,
                              max_sequence_length=int(train_cfg["max_sequence_length"]),
                              tokenizer_sha256=train_cfg["tokenizer_sha256"], tokenizer_path=champion["checkpoint"])
    rows = manifest["artifacts"]["train"]["rows"]
    steps = max(1, math.ceil(epochs * rows / DATA_PARALLEL))
    config = training_config({**train_cfg, "max_steps": steps}, role="planner", base_model=champion["checkpoint"],
                             data_dir=data_dir, manifest=manifest)
    config_path = data_dir / "sft.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint = gen_dir / "checkpoint"
    logger.info("training %s from %s: %d samples, %d rows, %d steps", gen_dir.name, champion["name"], len(samples), rows, steps)
    for name in specialist_containers():
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    if not (checkpoint / MERGE_MANIFEST).exists():
        _train(config_path, train_cfg["image"], DATA_PARALLEL)
        _merge(train_cfg["image"], Path(champion["checkpoint"]), Path(config["output"]["adapter_dir"]), checkpoint)
    return {"samples": len(samples), "rows": rows, "steps": steps, "epochs": epochs, "parent": champion["name"],
            "checkpoint": str(checkpoint), "adapter": config["output"]["adapter_dir"]}


def serve(cfg: dict, name: str, checkpoint: str, where: tuple[int, str]) -> None:
    _serve(cfg["train"]["serve"]["image"], {"role": f"planner-{name}", "checkpoint": checkpoint, "port": where[0],
                                            "gpus": where[1], "tp": 1})


def report(records: list[dict], arms: list[str], tau: float) -> dict:
    return {arm: versus(records, arm, "dsv4", tau) for arm in arms}


def log_report(title: str, comparison: dict) -> None:
    for arm, c in comparison.items():
        if not c:
            continue
        a, r = c["arm"], c["reference"]
        logger.info("%s %s vs dsv4: best %.4f vs %.4f (p=%.3f) max %.4f vs %.4f | reached %.2f vs %.2f | "
                    "median t_tau %.0fs vs %.0fs | median t_best %.0fs vs %.0fs", title, arm, a["mean_best"], r["mean_best"],
                    c["best"].get("p", float("nan")), a["max_best"], r["max_best"], a["reached_rate"], r["reached_rate"],
                    a["median_time_to_tau"], r["median_time_to_tau"], a["median_time_to_best"], r["median_time_to_best"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "output" / "micake-subagent-rsi" / "rsi_planner")
    parser.add_argument("--champion", default="gen7", help="initial champion name (bootstrap only)")
    parser.add_argument("--champion-checkpoint", default=str(REPO_ROOT / "output/micake-subagent-rsi/gen7/checkpoints/planner"))
    parser.add_argument("--generations", type=int, default=1, help="generations to run in this invocation")
    parser.add_argument("--samples-per-shard", type=int, default=3, help="e2e episodes per arm per MICake GPU")
    parser.add_argument("--epochs", type=float, default=2.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not os.environ.get("DSV4_API_KEY"):
        raise SystemExit("DSV4_API_KEY is not set")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = PlannerGate()
    lineage = Lineage(args.root)
    state = lineage.state
    if not state:
        state.update({"champion": {"name": args.champion, "checkpoint": args.champion_checkpoint},
                      "experience": list(SEED_EXPERIENCE), "generations": []})
        lineage.save()

    if "bootstrap" not in state:
        champion = state["champion"]
        serve(cfg, champion["name"], champion["checkpoint"], CHAMPION_SERVE)
        out = args.root / "bootstrap_eval"
        records = run_e2e(out, {champion["name"]: (CHAMPION_SERVE[0], champion["checkpoint"])}, args.samples_per_shard)
        state["bootstrap"] = {"eval_dir": str(out.relative_to(REPO_ROOT)), "vs_dsv4": report(records, [champion["name"]], gate.tau)}
        state["experience"].append(str(out.relative_to(REPO_ROOT)))
        lineage.save()
        log_report("bootstrap", state["bootstrap"]["vs_dsv4"])

    for _ in range(args.generations):
        gens = state["generations"]
        if not gens or gens[-1].get("decision"):
            gens.append({"name": f"p{len(gens) + 1}", "parent": state["champion"]["name"], "started": time.time()})
            lineage.save()
        gen = gens[-1]
        champion = state["champion"]
        gen_dir = args.root / gen["name"]
        if "data" not in gen:
            gen["data"] = str(build(gen_dir, state["experience"]))
            lineage.save()
        if "train" not in gen:
            gen["train"] = train(cfg, gen_dir, Path(gen["data"]), champion, args.epochs)
            lineage.save()
        serve(cfg, champion["name"], champion["checkpoint"], CHAMPION_SERVE)
        serve(cfg, gen["name"], gen["train"]["checkpoint"], CHALLENGER_SERVE)
        out = gen_dir / "eval"
        records = run_e2e(out, {champion["name"]: (CHAMPION_SERVE[0], champion["checkpoint"]),
                                gen["name"]: (CHALLENGER_SERVE[0], gen["train"]["checkpoint"])}, args.samples_per_shard)
        rel = str(out.relative_to(REPO_ROOT))
        if rel not in state["experience"]:
            state["experience"].append(rel)
        gen["eval_dir"] = rel
        gen["vs_dsv4"] = report(records, [champion["name"], gen["name"]], gate.tau)
        gen["gate"] = gate_decision(records, champion["name"], gen["name"], gate)
        gen["decision"] = gen["gate"]["decision"]
        gen["finished"] = time.time()
        log_report(gen["name"], gen["vs_dsv4"])
        logger.info("%s gate: %s %s", gen["name"], gen["decision"], gen["gate"]["reasons"])
        if gen["decision"] == "PROMOTE":
            previous = champion
            state["champion"] = {"name": gen["name"], "checkpoint": gen["train"]["checkpoint"]}
            if Path(previous["checkpoint"]).is_relative_to(args.root):
                shutil.rmtree(previous["checkpoint"], ignore_errors=True)
        for held in gens[:-1]:
            if held.get("decision") == "HOLD":
                shutil.rmtree(held["train"]["checkpoint"], ignore_errors=True)
        lineage.save()


if __name__ == "__main__":
    main()
