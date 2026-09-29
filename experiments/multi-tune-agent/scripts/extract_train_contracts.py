"""Extract operator+shape contracts from training data to exclude from benchmarks."""

import json
import re
import hashlib
from collections import defaultdict


def extract():
    train_hashes = set()
    train_op_shapes = set()
    train_contract_hashes = set()

    for split in ["train", "dev"]:
        path = f"/home/danyzhan/Lumen/experiments/GEAK-agent-coder/data/build/qwen3_30b_a3b_phase1/{split}.tokenized.jsonl"
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("sample_domain") != "kernel":
                    continue
                sid = row.get("sample_id", "")
                train_hashes.add(sid)

                msgs = row.get("messages", [])
                for m in msgs:
                    if m.get("role") != "user":
                        continue
                    c = m.get("content", "")
                    if not isinstance(c, str):
                        continue
                    try:
                        payload = json.loads(c)
                        inp = payload.get("input", {})
                        direction = inp.get("task_direction", "")

                        match = re.search(r'Frozen contract JSON:\s*(\{.*?\})', direction)
                        if match:
                            try:
                                contract = json.loads(match.group(1))
                                op = contract.get("operator", "")
                                shape = json.dumps(contract.get("shape", {}), sort_keys=True)
                                train_op_shapes.add(f"{op}|{shape}")
                                ch = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
                                train_contract_hashes.add(ch)
                            except json.JSONDecodeError:
                                pass

                        # Also look in source_files for metadata.json
                        source = inp.get("source_files", {})
                        if isinstance(source, dict):
                            meta_str = source.get("metadata.json", "")
                            if meta_str:
                                try:
                                    meta = json.loads(meta_str)
                                    ch = meta.get("contract_hash", "")
                                    if ch:
                                        train_contract_hashes.add(ch)
                                except:
                                    pass
                    except json.JSONDecodeError:
                        pass

    print(f"Train+dev kernel samples: {len(train_hashes)}")
    print(f"Operator|shape combos: {len(train_op_shapes)}")
    print(f"Contract hashes: {len(train_contract_hashes)}")

    for combo in sorted(train_op_shapes)[:30]:
        print(f"  {combo}")
    if len(train_op_shapes) > 30:
        print(f"  ... and {len(train_op_shapes) - 30} more")

    result = {
        "sample_hashes": sorted(train_hashes),
        "op_shapes": sorted(train_op_shapes),
        "contract_hashes": sorted(train_contract_hashes),
    }
    with open("/tmp/train_exclusions.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to /tmp/train_exclusions.json")


if __name__ == "__main__":
    extract()
