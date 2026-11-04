#!/usr/bin/env python3
"""Collect the bench_<label>-<set>.json files of one label into a table.

One row per question set, then the unweighted mean tok/fwd over complete sets.
A set still running only has <out>.partial.json; it is listed with
--include-partial but never enters the mean.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

PARTIAL_SUFFIX = ".json.partial.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="run_benchmark.sh results dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--include-partial", action="store_true")
    a = ap.parse_args()

    prefix = f"bench_{a.label}-"
    found = {}
    for path in sorted(glob.glob(os.path.join(a.results, f"{prefix}*.json"))):
        base = os.path.basename(path)
        partial = base.endswith(PARTIAL_SUFFIX)
        if partial and not a.include_partial:
            continue
        name = base[len(prefix):-len(PARTIAL_SUFFIX if partial else ".json")]
        if name in found and partial:
            continue
        with open(path) as f:
            found[name] = (json.load(f), partial)

    if not found:
        print(f"no {prefix}*.json under {a.results}")
        return 1

    print(f"{'set':28} {'n':>6} {'tok/fwd':>8} {'accept':>7} {'steps':>10}")
    complete = []
    for name in sorted(found):
        d, partial = found[name]
        tf = d.get("tok_per_fwd", 0.0)
        flag = " *partial" if partial else ""
        print(f"{name:28} {d.get('n', 0):6d} {tf:8.4f} "
              f"{100 * d.get('acceptance_rate', 0.0):6.2f}% "
              f"{d.get('forward_steps', 0):10d}{flag}")
        if not partial:
            complete.append(tf)

    print()
    print(f"complete sets: {len(complete)}/{len(found)}")
    if complete:
        print(f"mean tok/fwd : {sum(complete) / len(complete):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
