#!/usr/bin/env python3
"""Resume a model-directory copy with bounded per-node file parallelism."""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=True)
    # Hugging Face download metadata can be owner-only and is not part of the
    # loadable checkpoint. Copy model artifacts only; traversing `.cache` makes
    # an otherwise complete staging run fail on unreadable tree metadata.
    files = [
        path
        for path in source.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(source).parts
    ]
    pending = []
    for src in files:
        dst = destination / src.relative_to(source)
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            continue
        pending.append((src, dst))

    def copy_one(pair: tuple[Path, Path]) -> int:
        src, dst = pair
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"{dst.name}.partial-{os.getpid()}")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return src.stat().st_size

    copied = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(copy_one, pair) for pair in pending]
        for future in as_completed(futures):
            copied += future.result()
            print(f"copied={copied} remaining_files={len(futures) - 1}", flush=True)
            futures.pop()

    (destination / ".copy_complete").touch()
    print(
        f"complete files={len(files)} newly_copied_bytes={copied}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
