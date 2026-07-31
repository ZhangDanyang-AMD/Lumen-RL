"""Disentangle the two sources of divergence between the parsers:

  (a) rendering  -- apply_chat_template(thinking=True) vs build_chat_segments()
  (b) encoding   -- tokenizer(text, add_special_tokens=False) vs the native
                    tiktoken path with allow_special_tokens

Prints the first record where each differs, so the cause is visible rather
than inferred.
"""

import difflib
import importlib.util
import json
import os
import sys

from transformers import AutoTokenizer

MODEL = "/mnt/m2m_nobackup/jimguo12/models/Kimi-K3"
DATA = "/dev/shm/kimi-mtp-dataset-phase1/train.jsonl"
REF_PARSER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "their_parser_ref.py")
N = 200


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    from lumenrl.data.dataset import _resolve_last_turn_loss_only
    from lumenrl.data.kimi_k25_parser import normalize_conversation
    from lumenrl.data.kimi_k3_parser import KimiK3Parser as Local

    their = load_module("their_k3", REF_PARSER)
    p_local = Local(tok, MODEL)
    p_their = their.KimiK3Parser(tok)

    rows = []
    with open(DATA) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= N:
                break

    text_diff = 0
    enc_diff = 0
    shown_text = shown_enc = False
    turn_hist = {}

    for row in rows:
        conv = row.get("conversations") or row.get("messages")
        if not conv:
            continue
        conv = normalize_conversation(conv)
        n_turns = sum(1 for m in conv if m.get("role") == "assistant")

        t_local = p_local.format(conv)
        t_their = p_their.format(conv)

        same_text = t_local == t_their
        if not same_text:
            text_diff += 1
            turn_hist[n_turns] = turn_hist.get(n_turns, 0) + 1

        # Encoding comparison on the SAME text, isolating (b) from (a).
        ids_hf = tok(t_local, add_special_tokens=False).input_ids
        ids_native = []
        for seg in p_local._segments(p_local._to_k3_messages(conv), False):
            ids_native.extend(p_local._encode_segment(seg))
        same_enc = list(ids_hf) == list(ids_native)
        if not same_enc:
            enc_diff += 1

        if not same_text and not shown_text:
            shown_text = True
            print("=" * 70)
            print(f"FIRST RENDERING DIFFERENCE (assistant turns = {n_turns})")
            print("=" * 70)
            d = difflib.unified_diff(
                t_local.splitlines(), t_their.splitlines(),
                "local(build_chat_segments)", "their(apply_chat_template)",
                lineterm="", n=1,
            )
            for i, line in enumerate(d):
                if i > 24:
                    print("  ...")
                    break
                print("  " + line[:200])

        if not same_enc and not shown_enc:
            shown_enc = True
            print()
            print("=" * 70)
            print("FIRST ENCODING DIFFERENCE (identical input text)")
            print("=" * 70)
            print(f"  hf_len={len(ids_hf)}  native_len={len(ids_native)}")
            for j, (x, y) in enumerate(zip(ids_hf, ids_native)):
                if x != y:
                    lo = max(0, j - 4)
                    print(f"  first divergence at token {j}")
                    print(f"    hf     {list(ids_hf[lo:j+6])}")
                    print(f"    native {list(ids_native[lo:j+6])}")
                    print(f"    hf     decoded: {tok.decode(list(ids_hf[lo:j+6]))!r}")
                    print(f"    native decoded: {tok.decode(list(ids_native[lo:j+6]))!r}")
                    break

    print()
    print("=" * 70)
    print(f"rendering differs:  {text_diff}/{len(rows)}   by assistant-turn count: {turn_hist}")
    print(f"encoding differs:   {enc_diff}/{len(rows)}   (same text in, both paths)")
    print("=" * 70)


if __name__ == "__main__":
    main()
