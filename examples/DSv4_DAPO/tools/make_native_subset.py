"""Carve an N-layer subset out of the native DeepSeek-V4-Flash-Base checkpoint.

Copies `layers.0..N-1` plus the top-level tensors verbatim (still FP8, still the
checkpoint's own names) and writes a config with `num_hidden_layers=N`. `mtp.*` is
dropped: a truncated model has no use for it.

Only the source shards that hold a wanted tensor are opened, which is what lets the
downloader fetch just those shards (`hf_plan.py` applies the same `wanted()` rule).

    python3 make_native_subset.py --src <DeepSeek-V4-Flash-Base> --dst <out> --layers 4
"""

import argparse
import json
import os
import shutil
import sys
import time

from safetensors import safe_open
from safetensors.torch import save_file

DROP_PREFIXES = ("mtp.",)


def wanted(name: str, n_layers: int) -> bool:
    if name.startswith(DROP_PREFIXES):
        return False
    if name.startswith("layers."):
        return int(name.split(".")[1]) < n_layers
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="native FP8 checkpoint directory")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--shard-gb", type=float, default=4.0)
    cli = ap.parse_args()
    shard_bytes = int(cli.shard_gb * 1024**3)

    with open(os.path.join(cli.src, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    keep = sorted(n for n in weight_map if wanted(n, cli.layers))
    print(f"src {len(weight_map)} tensors -> keeping {len(keep)} for {cli.layers} layer(s)")

    by_shard: dict[str, list[str]] = {}
    for n in keep:
        by_shard.setdefault(weight_map[n], []).append(n)
    missing = [s for s in sorted(by_shard) if not os.path.exists(os.path.join(cli.src, s))]
    if missing:
        print(f"missing source shards: {missing}", file=sys.stderr)
        return 1
    print(f"reading {len(by_shard)} source shard(s): {sorted(by_shard)}")

    os.makedirs(cli.dst, exist_ok=True)
    buf = {}
    buf_bytes = 0
    out_index: dict[str, str] = {}
    shard_id = 0
    total_bytes = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, buf_bytes, shard_id
        if not buf:
            return
        shard_id += 1
        fname = f"model-{shard_id:05d}.safetensors"
        save_file(buf, os.path.join(cli.dst, fname), metadata={"format": "pt"})
        for n in buf:
            out_index[n] = fname
        print(f"  wrote {fname}  {len(buf)} tensors  {buf_bytes / 1024**3:.2f} GiB  "
              f"[{time.time() - t0:.0f}s]", flush=True)
        buf = {}
        buf_bytes = 0

    # Group by source shard so each shard is opened once and read front-to-back.
    for shard in sorted(by_shard):
        with safe_open(os.path.join(cli.src, shard), framework="pt", device="cpu") as f:
            for n in by_shard[shard]:
                t = f.get_tensor(n)
                buf[n] = t
                nbytes = t.numel() * t.element_size()
                buf_bytes += nbytes
                total_bytes += nbytes
                if buf_bytes >= shard_bytes:
                    flush()
    flush()

    total_shards = shard_id
    rename = {}
    for i in range(1, total_shards + 1):
        old = f"model-{i:05d}.safetensors"
        new = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        os.rename(os.path.join(cli.dst, old), os.path.join(cli.dst, new))
        rename[old] = new
    out_index = {n: rename[s] for n, s in out_index.items()}

    with open(os.path.join(cli.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": out_index}, f, indent=1)

    with open(os.path.join(cli.src, "config.json")) as f:
        cfg = json.load(f)
    cfg["num_hidden_layers"] = cli.layers
    # `compress_ratios` / `num_hash_layers` are per-layer schedules that the config
    # class truncates to num_hidden_layers, so the full-length lists stay correct for
    # any prefix of layers.
    cfg.pop("num_nextn_predict_layers", None)
    with open(os.path.join(cli.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "special_tokens_map.json"):
        src = os.path.join(cli.src, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(cli.dst, extra))

    print(f"done: {cli.dst}  {total_bytes / 1024**3:.2f} GiB in {total_shards} shard(s), "
          f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
