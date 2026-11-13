"""Build the DAPO parquets: DAPO-Math-17k (train) and AIME-2024 (validation).

Recipe of examples/docs/03-data.md on the main branch: keep prompts whose length
under the Qwen3-8B-Base chat template (add_generation_prompt=True) is <= 1024.

The validation file is loaded at startup even though the DSv4 configs never
evaluate (val_steps: 0): run_dapo.sh falls back to it when VAL_FILE is empty, and
the trainer fails if it is missing.

--filter selects how the prompt length is measured:

  as-validated  (default) the expression the validated run's files were built with,
                ``len(apply_chat_template(..., tokenize=True))``. Under transformers
                5.x that call returns a BatchEncoding, whose len() is its number of
                keys (2), so no prompt is removed: the outputs are the datasets'
                rows unchanged and in order -- exactly the files the reference
                numbers were measured on. Needed for those numbers, because the
                trainer shuffles the rows with a fixed seed.
  tokens        counts tokens, as the docs intend; removes 800 training rows (8
                distinct prompts), so step-by-step numbers differ from the reference.

The DSv4 run truncates prompts at 1024 tokens either way.

All three Hub repos are pinned to a revision, so the output does not drift with
them. Honours HF_ENDPOINT / HF_TOKEN / HF_HOME through huggingface_hub.

    python3 make_dapo_data.py --out-dir <DATA_ROOT>/data_cached/qwen3-8b-maxprompt1024 \
        --raw-dir <DATA_ROOT>/raw [--filter tokens]
"""

import argparse
import glob
import hashlib
import os

TOKENIZER = ("Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4")
DATASETS = (
    # repo, revision, local dir, output file
    ("BytedTsinghua-SIA/DAPO-Math-17k", "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
     "DAPO-Math-17k", "dapo-math-17k.filtered.parquet"),
    ("BytedTsinghua-SIA/AIME-2024", "aa49075e24ad594b79fdf0bdcefa735c2181be67",
     "AIME-2024", "aime-2024.filtered.parquet"),
)
PROMPT_KEY = "prompt"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw-dir", required=True, help="where the raw datasets and tokenizer go")
    ap.add_argument("--max-prompt", type=int, default=1024)
    ap.add_argument("--filter", choices=("as-validated", "tokens"), default="as-validated")
    ap.add_argument("--num-proc", type=int, default=0, help="0 = min(64, cpus/4)")
    cli = ap.parse_args()

    import datasets
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tok_repo, tok_rev = TOKENIZER
    tok_dir = snapshot_download(tok_repo, revision=tok_rev,
                                local_dir=os.path.join(cli.raw_dir, "Qwen3-8B-Base-tokenizer"),
                                allow_patterns=["tokenizer*", "vocab.json", "merges.txt",
                                                "config.json"])
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    def doc2len(doc) -> int:
        enc = tokenizer.apply_chat_template(doc[PROMPT_KEY], add_generation_prompt=True,
                                            tokenize=True)
        if cli.filter == "as-validated":
            return len(enc)
        # transformers 4.x returns a flat list of ids; 5.x returns a BatchEncoding.
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return len(ids)

    nproc = cli.num_proc or max(1, min(64, (os.cpu_count() or 8) // 4))
    os.makedirs(cli.out_dir, exist_ok=True)
    for repo, rev, local, out_name in DATASETS:
        ds_dir = snapshot_download(repo, repo_type="dataset", revision=rev,
                                   local_dir=os.path.join(cli.raw_dir, local),
                                   allow_patterns=["*.parquet"])
        src = sorted(glob.glob(os.path.join(ds_dir, "**", "*.parquet"), recursive=True))[0]
        out = os.path.join(cli.out_dir, out_name)
        ds = datasets.Dataset.from_parquet(src)
        before = len(ds)
        ds = ds.filter(lambda d: doc2len(d) <= cli.max_prompt, num_proc=nproc,
                       desc=f"Filtering prompts > {cli.max_prompt} tokens")
        ds.to_parquet(out)
        print(f"[{repo}@{rev[:9]}] -> {out}: {before} -> {len(ds)} rows "
              f"(removed {before - len(ds)}, --filter {cli.filter})")
        print(f"sha256 {sha256(out)}  {out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
