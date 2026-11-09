"""Train a specialist generation by step-level rejection-sampling SFT on a rollout pool.

Builds ``<root>/<generation>/rl/<role>.jsonl`` from ``collect_rft.py`` rollouts,
tokenizes the kept turns, trains one role LoRA per role on all eight GPUs
(stopping the specialist servers first), merges and serves each checkpoint.

Example:
    python experiments/micake-subagent-rsi/scripts/train_rft.py --generation gen2 --epochs 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_cycle import _merge, _serve, _train  # noqa: E402
from run_ab_eval import DEFAULT_CONFIG, REPO_ROOT  # noqa: E402
from micake_rsi.cycle import MERGE_MANIFEST, training_config  # noqa: E402
from micake_rsi.rft import RftFilter, rft_sample  # noqa: E402
from micake_rsi.sft_data import write_role_sft  # noqa: E402

logger = logging.getLogger("train_rft")

DATA_PARALLEL = 8
SPECIALIST_PREFIXES = ("atom-micake-", "atom-qwen3-specialist")


def specialist_containers() -> list[str]:
    names = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], check=True, capture_output=True, text=True).stdout.split()
    return [n for n in names if n.startswith(SPECIALIST_PREFIXES)]


def load_pools(pools: list[Path], role: str) -> list[dict]:
    records = []
    for pool in pools:
        path = pool / f"{role}.jsonl"
        if path.exists():
            records += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pool", type=Path, action="append", help="rollout pool directory (repeatable)")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--roles", default="planner,critic")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--min-speedup", type=float, default=0.95)
    parser.add_argument("--serve", default="planner:8103:3,critic:8104:4", help="role:port:hip_gpu after training")
    parser.add_argument("--extra-samples", type=Path, action="append", default=[],
                        help="ready-made planner samples (e.g. build_corrections.py output), repeatable")
    parser.add_argument("--extra-repeat", type=int, default=1, help="times each extra sample is repeated")
    parser.add_argument("--annotate-lds", action="store_true", help="prefix kept Planner submissions with their LDS arithmetic")
    parser.add_argument("--dry-run", action="store_true", help="build data and configs only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train = cfg["train"]
    root = (REPO_ROOT / cfg["rsi"]["root"]).resolve() / args.generation
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(train["base_model"], trust_remote_code=True)
    pools = args.pool or [REPO_ROOT / "output" / "micake-subagent-rsi" / "rft_pool"]
    rft = RftFilter(min_speedup=args.min_speedup)
    jobs = []
    for role in args.roles.split(","):
        samples = [rft_sample(r, role, rft, annotate_lds=args.annotate_lds) for r in load_pools(pools, role)]
        if role == "planner":
            for path in args.extra_samples:
                extra = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                samples += extra * args.extra_repeat
        rl_path = root / "rl" / f"{role}.jsonl"
        rl_path.parent.mkdir(parents=True, exist_ok=True)
        rl_path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
        data_dir = root / "train" / role
        manifest = write_role_sft(samples, tokenizer, data_dir, min_advantage=0.0,
                                  max_sequence_length=int(train["max_sequence_length"]),
                                  tokenizer_sha256=train["tokenizer_sha256"], tokenizer_path=train["base_model"])
        rows = manifest["artifacts"]["train"]["rows"]
        steps = max(1, math.ceil(args.epochs * rows / DATA_PARALLEL))
        config = training_config({**train, "max_steps": steps}, role=role, base_model=train["base_model"],
                                 data_dir=data_dir, manifest=manifest)
        config_path = data_dir / "sft.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        kept = sum(1 for s in samples for st in s["steps"] if st["advantage"] > 0)
        logger.info("%s: %d episodes, %d kept turns, %d rows, %d steps", role, len(samples), kept, rows, steps)
        jobs.append({"role": role, "config_path": config_path, "adapter_dir": Path(config["output"]["adapter_dir"]),
                     "checkpoint": root / "checkpoints" / role, "rows": rows, "steps": steps})
    (root / "rft_manifest.json").write_text(json.dumps(
        {"pools": [str(p) for p in pools], "extra_samples": [str(p) for p in args.extra_samples],
         "extra_repeat": args.extra_repeat, "annotate_lds": args.annotate_lds, "epochs": args.epochs, "min_speedup": args.min_speedup,
         "jobs": [{k: str(v) if isinstance(v, Path) else v for k, v in j.items()} for j in jobs]}, indent=2), encoding="utf-8")
    if args.dry_run:
        return

    for name in specialist_containers():
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    for job in jobs:
        if (job["checkpoint"] / MERGE_MANIFEST).exists():
            logger.info("%s already merged at %s", job["role"], job["checkpoint"])
            continue
        _train(job["config_path"], train["image"], DATA_PARALLEL)
        _merge(train["image"], Path(train["base_model"]), job["adapter_dir"], job["checkpoint"])
    serve_image = train["serve"]["image"]
    for entry in filter(None, args.serve.split(",")):
        role, port, gpus = entry.split(":")
        job = next(j for j in jobs if j["role"] == role)
        _serve(serve_image, {"role": f"{role}-{args.generation}", "checkpoint": str(job["checkpoint"]), "port": int(port),
                             "gpus": gpus, "tp": 1})


if __name__ == "__main__":
    main()
