"""Decide which files of DeepSeek-V4-Flash-Base a given layer count needs.

Reads the checkpoint's model.safetensors.index.json, config.json and the Hub's file
listing for the pinned revision (`/api/models/<repo>/tree/<rev>`), and writes one
line per file:  <path> TAB <lfs|git> TAB <size> TAB <hash>

`lfs` files are checked against the LFS sha256 the Hub reports; small files that
live in git are checked against their git blob sha1. For a slice of N layers a
shard is needed only if it holds `layers.<i>` with i < N or a top-level tensor;
`mtp.*` is never needed (make_native_subset.py drops it). When N equals the model's
own layer count, every shard is needed.
"""

import argparse
import json
import sys

SMALL_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json",
               "tokenizer_config.json", "generation_config.json")


def wanted(name: str, n_layers: int) -> bool:
    if name.startswith("mtp."):
        return False
    if name.startswith("layers."):
        return int(name.split(".")[1]) < n_layers
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    with open(cli.index) as f:
        weight_map = json.load(f)["weight_map"]
    with open(cli.config) as f:
        total = int(json.load(f)["num_hidden_layers"])
    with open(cli.tree) as f:
        tree = {e["path"]: e for e in json.load(f) if e.get("type") == "file"}

    if cli.layers < 1 or cli.layers > total:
        print(f"layers must be in 1..{total}, got {cli.layers}", file=sys.stderr)
        return 2
    if cli.layers == total:
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted({f for n, f in weight_map.items() if wanted(n, cli.layers)})

    rows = []
    for path in [p for p in SMALL_FILES if p in tree] + shards:
        e = tree.get(path)
        if e is None:
            print(f"{path} is not in the Hub listing", file=sys.stderr)
            return 1
        if "lfs" in e:
            rows.append((path, "lfs", e["lfs"]["size"], e["lfs"]["oid"]))
        else:
            rows.append((path, "git", e["size"], e["oid"]))

    with open(cli.out, "w") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    size = sum(r[2] for r in rows)
    print(f"layers={cli.layers}/{total} files={len(rows)} shards={len(shards)} "
          f"bytes={size} ({size / 1e9:.2f} GB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
