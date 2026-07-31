"""Standalone check of KimiK3Parser against real dataset records."""

import json
import sys

sys.path.insert(0, "/root/lumenrl")

from transformers import AutoTokenizer

from lumenrl.data.kimi_k25_parser import normalize_conversation, pack_loss_mask
from lumenrl.data.kimi_k3_parser import KimiK3Parser

MODEL = "/mnt/m2m_nobackup/jimguo12/models/Kimi-K3"
DATA = "/dev/shm/kimi-mtp-dataset-phase1/train.jsonl"
VOCAB = 163840
MAX_LEN = 2048

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
parser = KimiK3Parser(tok, MODEL)


def show(idx, conv):
    messages = normalize_conversation(conv)
    ids, mask = parser.parse(messages, MAX_LEN)
    hi = int(ids.max()) if len(ids) else -1
    n_turns = sum(1 for m in messages if m["role"] == "assistant")
    print(f"--- record {idx}: {len(messages)} msgs ({n_turns} assistant), "
          f"{len(ids)} tokens, max_id={hi}, loss={int(mask.sum())}")
    assert hi < VOCAB, f"OUT OF VOCAB: {hi}"
    assert len(ids) == len(mask)
    print("    packed:", pack_loss_mask(mask)[:10])
    sup = [int(t) for t, m in zip(ids, mask) if m]
    unsup = [int(t) for t, m in zip(ids, mask) if not m]
    print("    supervised head :", repr(tok.decode(sup[:60])))
    print("    supervised tail :", repr(tok.decode(sup[-40:])))
    print("    unsupervised head:", repr(tok.decode(unsup[:60])))
    return ids, mask


records = []
with open(DATA) as f:
    for line in f:
        d = json.loads(line)
        conv = d.get("conversations")
        if conv:
            records.append(conv)
        if len(records) >= 400:
            break

two_turn = next(r for r in records if len(r) == 2)
multi_turn = next(r for r in records if len(r) >= 6)

print("=" * 70)
print("STRUCTURE (2-turn record, first 400 chars of rendering)")
print("=" * 70)
print(repr(parser.format(normalize_conversation(two_turn))[:400]))
print()

print("=" * 70)
print("SINGLE-TURN")
print("=" * 70)
show("2-turn", two_turn)
print()

print("=" * 70)
print(f"MULTI-TURN ({len(multi_turn)} messages)")
print("=" * 70)
ids, mask = show("multi", multi_turn)
print()

print("=" * 70)
print("MARKER SANITY")
print("=" * 70)
for name in ("OPEN_TOKEN", "CLOSE_TOKEN", "SEP_TOKEN", "END_OF_MSG_TOKEN"):
    text = getattr(parser.encoding, name)
    print(f"  {name:18s} {text:18s} -> {tok.encode(text, allow_special_tokens=True)}")
print()

print("=" * 70)
print(f"BULK SCAN over {len(records)} records")
print("=" * 70)
worst = -1
zero_loss = 0
total_tokens = 0
total_loss = 0
for i, conv in enumerate(records):
    ids, mask = parser.parse(normalize_conversation(conv), MAX_LEN)
    if len(ids):
        worst = max(worst, int(ids.max()))
    total_tokens += len(ids)
    total_loss += int(mask.sum())
    if int(mask.sum()) == 0:
        zero_loss += 1
print(f"  max token id across all records : {worst}  (vocab {VOCAB})")
print(f"  records with zero loss mask      : {zero_loss}/{len(records)}")
print(f"  supervised token fraction        : {total_loss / max(1, total_tokens):.3f}")
assert worst < VOCAB, "FAIL: out-of-vocab token produced"
print("\nPASS: all token ids within vocabulary")
