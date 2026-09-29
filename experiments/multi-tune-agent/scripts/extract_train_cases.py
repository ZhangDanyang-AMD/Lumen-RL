"""Extract operator+shape+dtype contracts from training data for benchmark exclusion."""

import json
import hashlib
import re
from collections import Counter


def extract():
    case_ids = set()
    contract_hashes = set()
    op_shape_keys = set()

    for split in ["train", "dev"]:
        path = f"/home/danyzhan/Lumen/experiments/GEAK-agent-coder/data/build/qwen3_30b_a3b_phase1/{split}.tokenized.jsonl"
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("sample_domain") != "kernel":
                    continue
                msgs = row.get("messages", [])
                for m in msgs:
                    if m.get("role") != "user":
                        continue
                    c = m.get("content", "")
                    if not isinstance(c, str):
                        continue
                    try:
                        payload = json.loads(c)
                        contract_info = payload.get("input", {}).get("contract", {})
                        cid = contract_info.get("case_id", "")
                        if cid:
                            case_ids.add(cid)

                        # Extract from request text in contract
                        inner = contract_info.get("contract", {})
                        req_text = inner.get("request", "")
                        if req_text:
                            match = re.search(r'Frozen contract JSON:\s*(\{.*?\})\s*\.', req_text)
                            if match:
                                try:
                                    frozen = json.loads(match.group(1))
                                    ch = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
                                    contract_hashes.add(ch)
                                    op = frozen.get("operator", "")
                                    shape = json.dumps(frozen.get("shape", {}), sort_keys=True)
                                    op_shape_keys.add(f"{op}|{shape}")
                                except json.JSONDecodeError:
                                    pass
                    except json.JSONDecodeError:
                        pass

    print(f"Training case_ids: {len(case_ids)}")
    print(f"Training contract_hashes: {len(contract_hashes)}")
    print(f"Training op|shape keys: {len(op_shape_keys)}")
    print(f"\nSample op|shape keys:")
    for k in sorted(op_shape_keys)[:20]:
        print(f"  {k}")
    if len(op_shape_keys) > 20:
        print(f"  ... and {len(op_shape_keys) - 20} more")

    result = {
        "case_ids": sorted(case_ids),
        "contract_hashes": sorted(contract_hashes),
        "op_shape_keys": sorted(op_shape_keys),
    }
    with open("/tmp/train_exclusions.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to /tmp/train_exclusions.json")


if __name__ == "__main__":
    extract()
