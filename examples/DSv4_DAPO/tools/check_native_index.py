"""Sanity-check a bf16-native checkpoint directory's tensor names.

The native-rename stage is a pure rename, and the failure it guards against is
silent: fp8_cast_bf16 emits DeepSeek-HF names (``model.layers.0.self_attn.wq_a``)
which vLLM's DSv4 loader rejects with ``KeyError: 'layers.0.input_layernorm.weight'``,
and the MTP block stays FP8 in a directory whose config.json no longer declares
quantization_config. So this counts both rather than trusting the directory size.

    python3 check_native_index.py <dir> [--expect-tensors N --expect-native M]
"""

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--expect-tensors", type=int, default=None)
    ap.add_argument("--expect-native", type=int, default=None)
    cli = ap.parse_args()

    with open(f"{cli.dir}/model.safetensors.index.json") as f:
        keys = list(json.load(f)["weight_map"])
    mtp = sum(1 for k in keys if k.startswith("mtp"))
    hf = sum(1 for k in keys if k.startswith("model.layers."))
    native = sum(1 for k in keys if k.startswith("layers."))
    ok = mtp == 0 and hf == 0 and native > 0
    if cli.expect_tensors is not None and len(keys) != cli.expect_tensors:
        ok = False
    if cli.expect_native is not None and native != cli.expect_native:
        ok = False
    print(f"tensors={len(keys)} native={native} hf_style={hf} mtp={mtp} "
          f"-> {'OK' if ok else 'BAD'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
