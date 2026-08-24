#!/usr/bin/env python3
"""Fetch and stage the ATOM-regenerated K3 responses, then check the schema.

The training path consumes ids re-rendered from this file through K3's chat
template, not the stored text, so the two things worth checking before spending a
teacher load on it are that every row carries the fields the K3 parser reads, and
that the final assistant turn keeps its reasoning in `reasoning_content` — the
field `encoding_k3` looks for. A row whose reasoning silently lands somewhere else
still trains, just against a prompt the teacher never saw.

The repo is gated (`gated: manual`), so this needs a token belonging to an account
whose access request was accepted:

    HF_TOKEN=hf_... python3 fetch_dataset.py

Naming the local copy `train.jsonl` is deliberate: run_docker.sh stages
`DATASET_SRC/train.jsonl` onto tmpfs and the configs read it from there, and the
tokenize cache key includes the file's size and mtime, so dropping the full
release over the partial one invalidates the cache by itself.

This writes to DATA_ROOT rather than straight to /dev/shm on purpose — tmpfs is
wiped between jobs, and run_docker.sh re-stages from the on-disk copy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REPO = "slippedJim/ATOM_regen_seeklight_kimi_mtp"
REMOTE = "data/train-partial.jsonl"


def download(url: str, token: str, dst: str) -> None:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print(f"\r  {done / 2**30:6.2f} / {total / 2**30:.2f} GiB "
                      f"({pct:5.1f}%)", end="", flush=True)
    print()


def check_schema(path: str, sample: int) -> int:
    """Return row count; complain about anything the K3 parser would mishandle."""
    n = 0
    no_convs = 0
    no_reasoning = 0
    bad_last_role = 0
    roles: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            n += 1
            if n > sample:
                continue
            row = json.loads(line)
            convs = row.get("conversations") or row.get("messages")
            if not convs or not isinstance(convs, list):
                no_convs += 1
                continue
            for m in convs:
                roles[m.get("role", "?")] = roles.get(m.get("role", "?"), 0) + 1
            last = convs[-1]
            if last.get("role") != "assistant":
                bad_last_role += 1
            if not last.get("reasoning_content"):
                no_reasoning += 1

    checked = min(n, sample)
    print(f"rows: {n}")
    print(f"schema, first {checked} rows:")
    print(f"  roles seen           : {roles}")
    print(f"  missing conversations: {no_convs}")
    print(f"  last turn not asst   : {bad_last_role}")
    print(f"  last turn no reasoning_content: {no_reasoning}")

    ok = True
    if no_convs:
        print("FAIL: rows without a 'conversations' list are skipped silently by "
              "the loader, which would shrink the dataset without saying so")
        ok = False
    if bad_last_role:
        print("FAIL: last_turn_loss_only supervises the final assistant turn; "
              "rows that do not end on one have nothing to supervise")
        ok = False
    if no_reasoning:
        # Not fatal on its own -- reasoning may be inline in <think> tags, which
        # the parser also handles -- but for this dataset it is the documented
        # layout, so an absence means the format is not what is expected here.
        print("WARN: expected reasoning_content on every final assistant turn "
              "for this dataset; check whether reasoning is inline instead")
    return n if ok else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dst-dir",
        default=os.path.join(
            os.environ.get("DATA_ROOT", os.path.expanduser("~")),
            "datasets", "atom-regen-kimi-mtp",
        ),
    )
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    dst = os.path.join(args.dst_dir, "train.jsonl")
    os.makedirs(args.dst_dir, exist_ok=True)

    if not args.skip_download:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print("HF_TOKEN is not set, and this repo is gated (gated: manual). "
                  "A token from an account with accepted access is required.")
            return 2
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{args.remote}"
        print(f"downloading {REPO}/{args.remote}")
        try:
            download(url, token, dst)
        except urllib.error.HTTPError as e:
            print(f"FAIL: HTTP {e.code} — "
                  f"{'token lacks access to this gated repo' if e.code in (401, 403) else e.reason}")
            return 1

    n = check_schema(dst, args.sample)
    if n < 0:
        return 1
    print(f"\nstaged: {dst}")
    print("next: preprocess_dataset.py --max-length 16384 (the largest window "
          "under consideration) to get survival, step count and cache_batches "
          "from the measured length distribution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
