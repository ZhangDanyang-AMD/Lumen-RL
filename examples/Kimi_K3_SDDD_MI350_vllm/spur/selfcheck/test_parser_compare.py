"""Compare the committed KimiK3Parser (HF-wrapper encode) against the local one
(native tiktoken encode) on real dataset records.

The question that matters: does routing K3's XTML text through
``tokenizer(text, add_special_tokens=False)`` still resolve ``<|open|>`` and
friends to their real ids, or does it split them into literal characters /
register them as out-of-vocab added tokens?
"""

import importlib.util
import json
import os
import sys

import torch
from transformers import AutoTokenizer

MODEL = "/mnt/m2m_nobackup/jimguo12/models/Kimi-K3"
DATA = "/dev/shm/kimi-mtp-dataset-phase1/train.jsonl"
REF_PARSER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "their_parser_ref.py")
N = 200
VOCAB = 163840

STRUCTURAL = {
    "<|end_of_msg|>": 163586,
    "<|open|>": 163587,
    "<|close|>": 163588,
    "<|sep|>": 163589,
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print("=" * 70)
    print("1. Do the structural tokens survive the HF wrapper?")
    print("=" * 70)
    for text, want in STRUCTURAL.items():
        hf = tok(text, add_special_tokens=False).input_ids
        native = tok.encode(text, allow_special_tokens=True)
        print(f"  {text:<16} want={want:<7} hf={hf}  native={native}")

    from lumenrl.data.kimi_k3_parser import KimiK3Parser as Local

    their = load_module("their_k3", REF_PARSER)
    Their = their.KimiK3Parser

    p_local = Local(tok, MODEL)
    p_their = Their(tok)

    rows = []
    with open(DATA) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= N:
                break

    stats = {
        "local": {"oov": 0, "sup": 0, "tot": 0, "empty": 0, "fail": 0},
        "their": {"oov": 0, "sup": 0, "tot": 0, "empty": 0, "fail": 0},
    }
    len_mismatch = 0
    mask_mismatch = 0
    examples = []

    from lumenrl.data.dataset import _resolve_last_turn_loss_only
    from lumenrl.data.kimi_k25_parser import normalize_conversation

    for row in rows:
        conv = row.get("conversations") or row.get("messages")
        if not conv:
            continue
        conv = normalize_conversation(conv)
        last_turn_only = _resolve_last_turn_loss_only(conv)
        out = {}
        for tag, parser in (("local", p_local), ("their", p_their)):
            try:
                ids, mask = parser.parse(
                    conv, max_length=2048, last_turn_only=last_turn_only
                )
            except Exception as exc:  # noqa: BLE001
                stats[tag]["fail"] += 1
                out[tag] = None
                if len(examples) < 3:
                    examples.append(f"{tag} raised {type(exc).__name__}: {exc}")
                continue
            ids = torch.as_tensor(ids).flatten()
            mask = torch.as_tensor(mask).flatten()
            s = stats[tag]
            s["tot"] += len(ids)
            s["sup"] += int(mask.sum())
            if int(mask.sum()) == 0:
                s["empty"] += 1
            if len(ids) and int(ids.max()) >= VOCAB:
                s["oov"] += 1
            out[tag] = (ids, mask)

        a, b = out.get("local"), out.get("their")
        if a and b:
            if len(a[0]) != len(b[0]):
                len_mismatch += 1
            else:
                if not torch.equal(a[1], b[1]):
                    mask_mismatch += 1

    print()
    print("=" * 70)
    print(f"2. Parsing {len(rows)} real records")
    print("=" * 70)
    for tag in ("local", "their"):
        s = stats[tag]
        frac = s["sup"] / s["tot"] if s["tot"] else 0.0
        print(
            f"  {tag:<6} tokens={s['tot']:<9} supervised={frac:.3f}  "
            f"oov_records={s['oov']}  zero_mask={s['empty']}  failures={s['fail']}"
        )
    print(f"  token-length disagreement: {len_mismatch}/{len(rows)}")
    print(f"  loss-mask disagreement (same length): {mask_mismatch}/{len(rows)}")

    official_mismatch = 0
    for row in rows:
        conv = row.get("conversations") or row.get("messages")
        if not conv:
            continue
        conv = normalize_conversation(conv)
        if p_local.format(conv) != p_local.official_format(conv):
            official_mismatch += 1
    print(f"  local format() vs apply_chat_template: {official_mismatch}/{len(rows)} differ")
    for line in examples:
        print(f"  ! {line}")


if __name__ == "__main__":
    main()
