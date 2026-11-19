"""Planner-only RSI loop.

Manager and Critic are DSV4.1 in every arm. The champion is the only model
that writes training data and the only LoRA base; it changes only when the gate
promotes. One generation:

0. collect (``--test-portfolio`` set): champion episodes on the C144 training
           shapes join the experience pool (the bootstrap also collects DSV4.1);
1. build:  the champion's own ``new_best`` turns. Fresh episodes are kept whole;
           older episodes are replayed only up to the same count. Scripted
           one-axis corrections are not training targets;
2. train:  LoRA SFT on the champion checkpoint, merged into the challenger;
3. eval:   interleaved end-to-end episodes champion / challenger / DSV4.1
           Planner. With ``--test-portfolio heldout`` they are scored on shapes
           disjoint from the training shapes and never join the experience pool;
           without it they are scored on C144 and join the pool;
4. gate:   ``planner_gate.gate_decision`` promotes or holds; the lineage file
           records every decision and the comparison against DSV4.1.
           Held merged checkpoints are removed after the next merge; adapters stay.

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
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_ab_eval import DEFAULT_CONFIG, REPO_ROOT  # noqa: E402
from run_cycle import _merge, _serve, _train  # noqa: E402
from train_rft import DATA_PARALLEL, specialist_containers  # noqa: E402
from micake_rsi.cycle import MERGE_MANIFEST, training_config  # noqa: E402
from micake_rsi.planner_gate import PlannerGate, beats_reference, gate_decision, versus  # noqa: E402
from micake_rsi.sft_data import write_role_sft  # noqa: E402

logger = logging.getLogger("rsi_planner")

MICAKE_GPUS = (0, 1, 6, 7)
GIB = 1024 ** 3
# A merged checkpoint is about 57 GiB. Start a merge only with a full copy plus slack,
# and stop if free space falls through the floor while that write is in progress.
E2E_ATTEMPTS = 3
MERGE_HEADROOM = 100 * GIB
DISK_FLOOR = 30 * GIB
MEM_FLOOR = 16 * GIB
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


def run_e2e(out: Path, arms: dict[str, tuple[int, str]], samples_per_shard: int, dtypes: str = "",
            portfolio: str = "c144", with_dsv4: bool = True) -> list[dict]:
    """Interleaved end-to-end episodes for ``arms`` (plus the all-DSV4.1 arm), sharded over the MICake GPUs."""
    names = ",".join([*arms, *(["dsv4"] if with_dsv4 else [])])
    planner_args = [x for name, (port, ckpt) in arms.items() for x in ("--planner-arm", f"{name}={port}:{ckpt}")]
    expected = len(MICAKE_GPUS) * samples_per_shard * len(names.split(","))
    commands, logs = [], []
    for i, gpu in enumerate(MICAKE_GPUS):
        commands.append([sys.executable, str(SCRIPTS_DIR / "run_ab_eval.py"), "--study", "e2e", "--arms", names,
                         *planner_args, "--fallback", "dsv4", "--samples", str(samples_per_shard),
                         "--sample-start", str(i * samples_per_shard), "--gpu", str(gpu), "--dtypes", dtypes,
                         "--portfolio", portfolio, "--out", str(out), "--out-name", f"e2e_s{i}"])
        logs.append(out / f"log_s{i}.txt")
    # run_ab_eval skips episodes that fail before their first turn (not recorded) and resumes finished keys,
    # so rerunning fills the gaps. Never gate on a partial set: the arms would be compared on different tasks.
    # Shards resume only from their own file, so a changed shard layout can record the same trial twice.
    # Count distinct trial keys, and dedupe before gating.
    for attempt in range(1, E2E_ATTEMPTS + 1):
        run_parallel(commands, logs)
        dropped = drop_infra_failures(out)
        records = dedupe_records(load_records(out))
        if dropped == 0 and len(records) >= expected:
            return records
        logger.warning("%s: %d/%d distinct episodes recorded after attempt %d (dropped %d serving-stall episodes); "
                       "rerunning to fill gaps", out, len(records), expected, attempt, dropped)
    raise RuntimeError(f"{out}: {len(records)}/{expected} distinct episodes recorded; refusing to gate on partial data")


def dedupe_records(records: list[dict]) -> list[dict]:
    """One record per trial key (``e2e:<arm>:<sample>``); the first recorded copy wins."""
    seen: dict[str, dict] = {}
    for record in records:
        seen.setdefault(record["key"], record)
    return list(seen.values())


def is_infra_failure(record: dict) -> bool:
    """A planner delegation that ended ERROR with zero turns: the serving endpoint stalled, not a policy outcome."""
    return any(c.get("terminal_state") == "ERROR" and not c.get("turns") for c in record.get("children", []))


def drop_infra_failures(out: Path) -> int:
    """Remove episodes hit by serving stalls so that the resume logic reruns them (never score them as failures)."""
    dropped = 0
    for path in sorted(glob.glob(str(out / "e2e*.jsonl"))):
        kept = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if is_infra_failure(json.loads(line)):
                dropped += 1
                continue
            kept.append(line)
        Path(path).write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    return dropped


def load_records(directory: Path) -> list[dict]:
    return [json.loads(line) for path in sorted(glob.glob(str(directory / "e2e*.jsonl")))
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build(gen_dir: Path, experience: list[str], fresh: list[str]) -> Path:
    data = gen_dir / "data"
    caches = [p for pattern in SEED_CACHES for p in glob.glob(str(REPO_ROOT / pattern))]
    caches += glob.glob(str(gen_dir.parent / "*" / "data" / "*.cache.json"))
    commands, logs = [], []
    for i, gpu in enumerate(MICAKE_GPUS):
        command = [sys.executable, str(SCRIPTS_DIR / "build_improvements.py"), "--gpu", str(gpu), "--shard", str(i),
                   "--shards", str(len(MICAKE_GPUS)), "--max-corrections", "0", "--out", str(data / f"s{i}.jsonl")]
        for directory in experience:
            command += ["--exp-dir", str(REPO_ROOT / directory)]
        for cache in caches:
            command += ["--cache-in", cache]
        commands.append(command)
        logs.append(data / f"log_s{i}.txt")
    run_parallel(commands, logs)
    merged = data / "planner.jsonl"
    merged.write_text("".join(Path(data / f"s{i}.jsonl").read_text() for i in range(len(MICAKE_GPUS))))
    kept = rebalance_training(merged, set(fresh))
    logger.info("training rows: %d fresh, %d replayed, %d written", kept["fresh"], kept["replayed"], kept["written"])
    return merged


def champion_experience(state: dict, producer: str) -> list[str]:
    """Collect dirs written while ``producer`` was the serving planner.

    Held-out tests are not in ``collect_dir``. Generations whose parent is an older policy stay on disk
    and out of the next update, so the LoRA base and the data producer stay the same model.
    """
    owned: list[str] = []
    for gen in state.get("generations") or []:
        if gen.get("parent") != producer:
            continue
        for directory in gen.get("collect_dir") or []:
            if directory not in owned:
                owned.append(directory)
    return owned


def rebalance_training(path: Path, fresh: set[str]) -> dict[str, int]:
    """Keep every fresh episode and at most that many older ones, so new traces are half the update."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    new = [row for row in rows if row.get("experience_dir") in fresh]
    old = [row for row in rows if row.get("experience_dir") not in fresh]
    replayed = old if not new or len(old) <= len(new) else random.Random(0).sample(old, len(new))
    kept = new + replayed
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    return {"fresh": len(new), "replayed": len(replayed), "written": len(kept)}


def resource_snapshot() -> dict[str, int]:
    available = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    return {"disk_free": shutil.disk_usage(REPO_ROOT).free, "mem_available": available}


def log_resources(where: str) -> dict[str, int]:
    snap = resource_snapshot()
    logger.info("resources %s: disk free %.1f GiB, mem available %.1f GiB",
                where, snap["disk_free"] / GIB, snap["mem_available"] / GIB)
    return snap


def _checkpoint_path(gen: dict) -> Path | None:
    raw = (gen.get("train") or {}).get("checkpoint")
    return Path(raw) if raw else None


def retire_held_checkpoints(state: dict, keep: set[Path]) -> None:
    """Remove merged weights of HOLD generations. Adapters under ``train/`` stay."""
    kept = {path.resolve() for path in keep if path.exists()}
    for gen in state.get("generations") or []:
        if gen.get("decision") != "HOLD":
            continue
        path = _checkpoint_path(gen)
        if path is None or not path.is_dir() or path.resolve() in kept:
            continue
        logger.info("removing held merged checkpoint %s", path)
        shutil.rmtree(path)


TMP_MAX_AGE_S = 24 * 3600


def clean_tmp(now: float | None = None) -> None:
    """Remove MICake worker scratch dirs from earlier runs. Anything older than a day cannot belong to a live eval."""
    cutoff = (now or time.time()) - TMP_MAX_AGE_S
    removed = 0
    for pattern in ("micake-triton-gpu*", "micake-worker-*", "micake-smoke"):
        for path in glob.glob(f"/tmp/{pattern}"):
            if os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
    if removed:
        logger.info("removed %d stale MICake temp dirs from /tmp", removed)


def ensure_headroom(state: dict, keep: set[Path], need: int = MERGE_HEADROOM) -> None:
    """Drop held checkpoints until a merge fits, and refuse to start one that would fill the disk."""
    snap = log_resources("headroom")
    if snap["disk_free"] < need:
        retire_held_checkpoints(state, keep)
        snap = log_resources("after cleanup")
    if snap["disk_free"] < need:
        raise SystemExit(f"disk free {snap['disk_free'] / GIB:.1f} GiB is below the {need / GIB:.0f} GiB merge headroom")
    if snap["mem_available"] < MEM_FLOOR:
        raise SystemExit(f"mem available {snap['mem_available'] / GIB:.1f} GiB is below {MEM_FLOOR / GIB:.0f} GiB")


def _stop_for_resources(reason: str) -> None:
    logger.error("stopping RSI: %s", reason)
    names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=False).stdout.split()
    for name in names:
        if name.startswith(("micake-rsi-train-", "micake-rsi-merge-")):
            subprocess.run(["docker", "stop", "-t", "20", name], check=False)
    for line in subprocess.run(["ps", "-o", "pid=", "--ppid", str(os.getpid())], capture_output=True, text=True, check=False).stdout.split():
        os.kill(int(line), signal.SIGTERM)
    os.kill(os.getpid(), signal.SIGTERM)


def _watch_resources(stop: threading.Event) -> None:
    while not stop.wait(60):
        snap = log_resources("watch")
        if snap["disk_free"] < DISK_FLOOR:
            _stop_for_resources(f"disk free {snap['disk_free'] / GIB:.1f} GiB < {DISK_FLOOR / GIB:.0f} GiB")
            return
        if snap["mem_available"] < MEM_FLOOR:
            _stop_for_resources(f"mem available {snap['mem_available'] / GIB:.1f} GiB < {MEM_FLOOR / GIB:.0f} GiB")
            return


def train(cfg: dict, gen_dir: Path, samples_path: Path, base: dict, epochs: float, state: dict) -> dict:
    train_cfg = {**cfg["train"], "base_model": base["checkpoint"]}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base["checkpoint"], trust_remote_code=True)
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data_dir = gen_dir / "train" / "planner"
    manifest = write_role_sft(samples, tokenizer, data_dir, min_advantage=0.0,
                              max_sequence_length=int(train_cfg["max_sequence_length"]),
                              tokenizer_sha256=train_cfg["tokenizer_sha256"], tokenizer_path=base["checkpoint"])
    rows = manifest["artifacts"]["train"]["rows"]
    steps = max(1, math.ceil(epochs * rows / DATA_PARALLEL))
    config = training_config({**train_cfg, "max_steps": steps}, role="planner", base_model=base["checkpoint"],
                             data_dir=data_dir, manifest=manifest)
    config_path = data_dir / "sft.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint = gen_dir / "checkpoint"
    logger.info("training %s from %s: %d samples, %d rows, %d steps", gen_dir.name, base["name"], len(samples), rows, steps)
    for name in specialist_containers():
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    if not (checkpoint / MERGE_MANIFEST).exists():
        ensure_headroom(state, {Path(base["checkpoint"])})
        _train(config_path, train_cfg["image"], DATA_PARALLEL)
        _merge(train_cfg["image"], Path(base["checkpoint"]), Path(config["output"]["adapter_dir"]), checkpoint)
    return {"samples": len(samples), "rows": rows, "steps": steps, "epochs": epochs, "parent": base["name"],
            "checkpoint": str(checkpoint), "adapter": config["output"]["adapter_dir"]}


def serve(cfg: dict, name: str, checkpoint: str, where: tuple[int, str]) -> None:
    _serve(cfg["train"]["serve"]["image"], {"role": f"planner-{name}", "checkpoint": checkpoint, "port": where[0],
                                            "gpus": where[1], "tp": 1})


def report(records: list[dict], arms: list[str], tau: float) -> dict:
    return {arm: versus(records, arm, "dsv4", tau) for arm in arms}


def _beats_dsv4(records: list[dict], name: str, gate: PlannerGate) -> bool:
    """Held-out stop: faster kernel than DSV4.1 overall and per dtype, and a median time to 1.04 that is not slower."""
    return beats_reference(records, name, gate)


def report_by_dtype(records: list[dict], arms: list[str], tau: float) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record.get("dtype") or "fp16", []).append(record)
    return {dtype: report(group, arms, tau) for dtype, group in sorted(groups.items())} if len(groups) > 1 else {}


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
    parser.add_argument("--until-dsv4", action="store_true",
                        help="keep training until a generation beats DSV4.1 on held-out kernel score and is not slower to reach 1.04")
    parser.add_argument("--samples-per-shard", type=int, default=3, help="held-out episodes per arm per MICake GPU")
    parser.add_argument("--collect-samples-per-shard", type=int, default=8,
                        help="training episodes per portfolio per MICake GPU")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--dtypes", default="", help="A/B element types rotated over e2e samples, e.g. bf16,fp8 (default FP16)")
    parser.add_argument("--gen-prefix", default="p", help="generation name prefix")
    parser.add_argument("--test-portfolio", choices=("", "heldout"), default="",
                        help="heldout: train on C144 shapes, gate on the disjoint held-out shapes (bootstrap only)")
    parser.add_argument("--seed-experience", action="append", help="initial experience dirs (bootstrap only; default e2e_v1, e2e_v2)")
    args = parser.parse_args()
    args.root = args.root.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not os.environ.get("DSV4_API_KEY"):
        raise SystemExit("DSV4_API_KEY is not set")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = PlannerGate()
    lineage = Lineage(args.root)
    state = lineage.state
    if not state:
        state.update({"champion": {"name": args.champion, "checkpoint": args.champion_checkpoint},
                      "experience": list(args.seed_experience or SEED_EXPERIENCE), "generations": [],
                      "dtypes": args.dtypes, "test_portfolio": args.test_portfolio})
        lineage.save()
    dtypes = state.get("dtypes", "")
    test_portfolio = state.get("test_portfolio", "")
    split = bool(test_portfolio)

    if "bootstrap" not in state:
        champion = state["champion"]
        serve(cfg, champion["name"], champion["checkpoint"], CHAMPION_SERVE)
        out = args.root / ("bootstrap_collect" if split else "bootstrap_eval")
        records = run_e2e(out, {champion["name"]: (CHAMPION_SERVE[0], champion["checkpoint"])}, args.samples_per_shard, dtypes)
        state["bootstrap"] = {"eval_dir": str(out.relative_to(REPO_ROOT)), "portfolio": "c144",
                              "vs_dsv4": report(records, [champion["name"]], gate.tau),
                              "vs_dsv4_by_dtype": report_by_dtype(records, [champion["name"]], gate.tau)}
        state["experience"].append(str(out.relative_to(REPO_ROOT)))
        lineage.save()
        log_report("bootstrap(train shapes)", state["bootstrap"]["vs_dsv4"])
        for dtype, comparison in state["bootstrap"]["vs_dsv4_by_dtype"].items():
            log_report(f"bootstrap[{dtype}]", comparison)

    stop_watch = threading.Event()
    threading.Thread(target=_watch_resources, args=(stop_watch,), name="rsi-resources", daemon=True).start()
    log_resources("start")
    pending = None if args.until_dsv4 else args.generations
    while pending is None or pending > 0:
        if pending is not None:
            pending -= 1
        gens = state["generations"]
        if not gens or gens[-1].get("decision"):
            gens.append({"name": f"{args.gen_prefix}{len(gens) + 1}", "parent": state["champion"]["name"], "started": time.time()})
            lineage.save()
        gen = gens[-1]
        champion = state["champion"]
        # Data and LoRA base are the same model: the champion. A HOLD generation's weights never feed the next one.
        base = champion
        if gen.get("parent") != base["name"]:
            gen["parent"] = base["name"]
            lineage.save()
        logger.info("%s LoRA base and data source: champion %s", gen["name"], champion["name"])
        gen_dir = args.root / gen["name"]
        retire_held_checkpoints(state, {Path(champion["checkpoint"])})
        clean_tmp()
        log_resources(gen["name"])
        if split and len(gens) > 1 and "collect_dir" not in gen:
            serve(cfg, champion["name"], champion["checkpoint"], CHAMPION_SERVE)
            collected = []
            for portfolio in ("c144", "large"):
                out = gen_dir / f"collect_{portfolio}"
                run_e2e(out, {champion["name"]: (CHAMPION_SERVE[0], champion["checkpoint"])}, args.collect_samples_per_shard,
                        dtypes, portfolio, with_dsv4=False)
                collected.append(str(out.relative_to(REPO_ROOT)))
                if collected[-1] not in state["experience"]:
                    state["experience"].append(collected[-1])
            gen["collect_dir"] = collected
            lineage.save()
        if "data" not in gen:
            experience = champion_experience(state, champion["name"])
            logger.info("training experience: %d dirs produced by %s (pool has %d)",
                        len(experience), champion["name"], len(state["experience"]))
            gen["data"] = str(build(gen_dir, experience, gen.get("collect_dir") or []))
            lineage.save()
        if "train" not in gen:
            gen["train"] = train(cfg, gen_dir, Path(gen["data"]), base, args.epochs, state)
            lineage.save()
        serve(cfg, champion["name"], champion["checkpoint"], CHAMPION_SERVE)
        serve(cfg, gen["name"], gen["train"]["checkpoint"], CHALLENGER_SERVE)
        out = gen_dir / ("test" if split else "eval")
        records = run_e2e(out, {champion["name"]: (CHAMPION_SERVE[0], champion["checkpoint"]),
                                gen["name"]: (CHALLENGER_SERVE[0], gen["train"]["checkpoint"])}, args.samples_per_shard, dtypes,
                          test_portfolio or "c144")
        rel = str(out.relative_to(REPO_ROOT))
        if not split and rel not in state["experience"]:
            state["experience"].append(rel)
        assert not split or rel not in state["experience"], "held-out test episodes must never become training experience"
        gen["eval_dir"] = rel
        gen["eval_portfolio"] = test_portfolio or "c144"
        gen["vs_dsv4"] = report(records, [champion["name"], gen["name"]], gate.tau)
        gen["vs_dsv4_by_dtype"] = report_by_dtype(records, [champion["name"], gen["name"]], gate.tau)
        gen["gate"] = gate_decision(records, champion["name"], gen["name"], gate)
        gen["decision"] = gen["gate"]["decision"]
        gen["finished"] = time.time()
        log_report(gen["name"], gen["vs_dsv4"])
        for dtype, comparison in gen["vs_dsv4_by_dtype"].items():
            log_report(f"{gen['name']}[{dtype}]", comparison)
        gen["vs_dsv4_gate"] = gate_decision(records, "dsv4", gen["name"], gate)
        logger.info("%s gate: %s %s", gen["name"], gen["decision"], gen["gate"]["reasons"])
        logger.info("%s vs dsv4 gate: %s %s", gen["name"], gen["vs_dsv4_gate"]["decision"], gen["vs_dsv4_gate"]["reasons"])
        promoted = gen["decision"] == "PROMOTE"
        if promoted:
            previous = champion
            state["champion"] = {"name": gen["name"], "checkpoint": gen["train"]["checkpoint"]}
            champion = state["champion"]
            previous_path = Path(previous["checkpoint"])
            if previous_path.is_relative_to(args.root) and previous_path.resolve() != Path(gen["train"]["checkpoint"]).resolve():
                logger.info("removing previous champion checkpoint %s", previous_path)
                shutil.rmtree(previous_path, ignore_errors=True)
        keep = {Path(champion["checkpoint"]), Path(gen["train"]["checkpoint"])}
        retire_held_checkpoints(state, keep)
        log_resources(f"{gen['name']} done")
        lineage.save()
        if split and _beats_dsv4(records, gen["name"], gate):
            state["met_dsv4"] = gen["name"]
            lineage.save()
            logger.info("%s beats DSV4.1 on held-out kernel score and reaches %.2f no later", gen["name"], gate.tau)
            break


if __name__ == "__main__":
    main()
