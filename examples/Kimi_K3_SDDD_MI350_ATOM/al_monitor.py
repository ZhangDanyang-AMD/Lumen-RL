#!/usr/bin/env python3
"""Track acceptance length on both sides of the training loop.

Acceptance length is what the draft model is ultimately judged on: the expected
number of tokens the target accepts per speculation round. The trainer reports
it for eval (``eval/simulated_acc_len``) but not for train, where only the
per-position acceptance rates are logged. Both are the same quantity, so this
recomputes the train side with the trainer's own formula and puts the two
series next to each other.

    AL = sum over positions i of (acc_0 * acc_1 * ... * acc_i)

A position only contributes if every position before it was accepted, hence the
running product rather than a sum of rates. AL = 1.0 means nothing beyond the
first token survives on average, so speculation buys nothing; the useful range
starts above that.

Watching the two together is the point. Train AL rising while eval AL stalls or
falls is the draft memorising the cached teacher batches rather than learning
the target's distribution -- and in batch-alternating mode each round trains 50
optimizer steps against one frozen cache, which is exactly the setup where that
shows up.

    python3 examples/Kimi_K3_SDDD_MI350_ATOM/al_monitor.py [logfile] [--tail N]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

DEFAULT_LOG = (
    "/root/lumenrl/output/Kimi_K3_SDDD/LumenRL/kimi-k3-dspark-atom-mi350.log"
)

TRAIN_STEP = re.compile(r"callbacks: step=(\d+)\s")
EVAL_STEP = re.compile(r"callbacks: eval step=(\d+)\s")
TRAIN_ACC = re.compile(r"(?<!eval/)\bstep_(\d+)_acc=([\d.eE+-]+)")
EVAL_ACC = re.compile(r"eval/step_(\d+)_acc=([\d.eE+-]+)")
EVAL_AL = re.compile(r"eval/simulated_acc_len=([\d.eE+-]+)")
LOSS = re.compile(r"\bloss=([\d.eE+-]+)")
EVAL_LOSS = re.compile(r"eval/loss=([\d.eE+-]+)")


def acceptance_length(acc_by_pos: dict[int, float]) -> float:
    """The trainer's formula: positions weighted by surviving every prior one."""
    total = 0.0
    running = 1.0
    for pos in sorted(acc_by_pos):
        running *= acc_by_pos[pos]
        total += running
    return total


def parse(path: pathlib.Path) -> tuple[list[tuple], list[tuple]]:
    train: list[tuple] = []
    evals: list[tuple] = []

    with path.open(errors="replace") as handle:
        for line in handle:
            is_eval = EVAL_STEP.search(line)
            if is_eval:
                step = int(is_eval.group(1))
                reported = EVAL_AL.search(line)
                accs = {int(p): float(v) for p, v in EVAL_ACC.findall(line)}
                al = float(reported.group(1)) if reported else acceptance_length(accs)
                loss = EVAL_LOSS.search(line)
                evals.append((step, al, float(loss.group(1)) if loss else None, accs))
                continue

            is_train = TRAIN_STEP.search(line)
            if is_train:
                step = int(is_train.group(1))
                accs = {int(p): float(v) for p, v in TRAIN_ACC.findall(line)}
                if not accs:
                    continue
                loss = LOSS.search(line)
                train.append(
                    (step, acceptance_length(accs), float(loss.group(1)) if loss else None, accs)
                )

    return train, evals


def fmt(value: float | None, width: int = 8, places: int = 4) -> str:
    return f"{value:>{width}.{places}f}" if value is not None else " " * width


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", nargs="?", default=DEFAULT_LOG)
    ap.add_argument("--tail", type=int, default=20, help="rows to show (0 = all)")
    args = ap.parse_args()

    path = pathlib.Path(args.logfile)
    if not path.exists():
        print(f"log not found: {path}", file=sys.stderr)
        return 1

    train, evals = parse(path)
    if not train and not evals:
        print("No acceptance metrics logged yet (still in Phase A?).")
        return 0

    eval_at = {step: al for step, al, _, _ in evals}
    rows = train[-args.tail :] if args.tail else train

    npos = len(train[-1][3]) if train else len(evals[-1][3])
    print(f"positions per round: {npos}   (AL ranges 0..{npos})")
    print()
    print(f"{'step':>6} {'train AL':>9} {'eval AL':>9} {'train loss':>11} {'gap':>8}")
    print("-" * 48)
    for step, al, loss, _ in rows:
        ev = eval_at.get(step)
        gap = (al - ev) if ev is not None else None
        print(f"{step:>6} {fmt(al, 9)} {fmt(ev, 9)} {fmt(loss, 11)} {fmt(gap, 8)}")

    if train:
        print()
        step, al, _, accs = train[-1]
        print(f"latest train step {step} per-position acceptance:")
        print("  " + "  ".join(f"p{p}={accs[p]:.4f}" for p in sorted(accs)))

    if len(evals) >= 2:
        first, last = evals[0], evals[-1]
        print()
        print(
            f"eval AL: {first[1]:.4f} (step {first[0]}) -> "
            f"{last[1]:.4f} (step {last[0]})   over {len(evals)} evals"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
