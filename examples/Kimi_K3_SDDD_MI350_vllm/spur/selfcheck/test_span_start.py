"""Where does generation actually begin, and how is multi-turn thinking kept?

The loss span should start exactly where the model starts generating, i.e. at
the first token the generation prompt does NOT already provide. Printing the
generation prompt tail settles whether `<|open|>think<|sep|>` is given by the
server or must be predicted by the draft.
"""

from transformers import AutoTokenizer

MODEL = "/mnt/m2m_nobackup/jimguo12/models/Kimi-K3"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    from lumenrl.data.kimi_k3_parser import KimiK3Parser

    p = KimiK3Parser(tok, MODEL)

    conv = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>t1</think>a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "<think>t2</think>a2"},
    ]

    prepared = p._to_k3_messages(conv)
    print("=" * 70)
    print("1. Generation prompt tail (what the server hands the draft)")
    print("=" * 70)
    gen = p._segments(p._to_k3_messages(conv[:3]), add_generation_prompt=True)
    print("  last 8 segments:", [s.text for s in gen[-8:]])

    print()
    print("=" * 70)
    print("2. Supervised span for each assistant turn (local parser)")
    print("=" * 70)
    ids, mask = p.parse(conv, max_length=4096)
    on = False
    for i, (tid, m) in enumerate(zip(ids.tolist(), mask.tolist())):
        if m and not on:
            print(f"  span opens at token {i}: {tok.decode(ids[i:i+8].tolist())!r}")
            on = True
        elif not m and on:
            print(f"  span closes before token {i}: {tok.decode(ids[max(0,i-6):i+4].tolist())!r}")
            on = False

    print()
    print("=" * 70)
    print("3. Multi-turn thinking: is earlier reasoning kept in history?")
    print("=" * 70)
    print("  prepared messages (local pre-processing):")
    for m in prepared:
        print(f"    role={m['role']:<10} reasoning={m.get('reasoning_content')!r} content={m['content']!r}")
    both = [dict(m) for m in prepared]
    both[1]["reasoning_content"] = "t1"
    txt_both = "".join(s.text for s in p._segments(both, False))
    print(f"  rendering WITH turn-1 reasoning kept contains 't1': {'t1' in txt_both}")
    txt_ours = "".join(s.text for s in p._segments(prepared, False))
    print(f"  our rendering contains 't1': {'t1' in txt_ours}")


if __name__ == "__main__":
    main()
