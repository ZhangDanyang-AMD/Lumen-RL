"""Dataset preprocessing pipeline with multiprocessing and caching.

Preprocesses ShareGPT-format datasets using KimiK25Parser to produce
input_ids + packed_loss_mask, cached as .pt files for fast reloading.

Data processing pipeline ported from TorchSpec (torchspec/data/dataset.py).
Reference: https://github.com/LightSeek-Foundation/TorchSpec
License: MIT
"""

# 6: thinking=True now preserves reasoning in history assistant turns (K3's
#    preserved thinking history mode). Any thinking=True cache built before this
#    holds empty think blocks and must not be reused.
PARSER_VERSION = 6

import hashlib
import logging
import multiprocessing as mp
import os

import torch
from tqdm import tqdm

from lumenrl.data.kimi_k25_parser import (
    KimiK25Parser,
    has_thinking_content,
    has_unbalanced_thinking_tags,
    normalize_conversation,
    pack_loss_mask,
    serialize_packed_loss_mask,
)

logger = logging.getLogger(__name__)

_worker_state = {}


def _init_worker(
    tokenizer_path,
    max_length,
    last_turn_loss_only,
    min_loss_tokens,
    chat_template="kimi-k25",
    drop_overlong=False,
    max_prompt_tokens=0,
    thinking=True,
):
    """Per-worker initializer — loads tokenizer once."""
    from transformers import AutoTokenizer

    _worker_state["tokenizer"] = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )

    if chat_template == "kimi-k3":
        from lumenrl.data.kimi_k3_parser import KimiK3Parser
        _worker_state["parser"] = KimiK3Parser(_worker_state["tokenizer"])
        _worker_state["supports_thinking"] = True
    elif chat_template == "kimi-k25":
        _worker_state["parser"] = KimiK25Parser(_worker_state["tokenizer"])
    else:
        from lumenrl.data.hf_generation_parser import HFGenerationParser
        override = None if chat_template == "hf-generation" else chat_template
        _worker_state["parser"] = HFGenerationParser(
            _worker_state["tokenizer"], chat_template_override=override
        )

    _worker_state["max_length"] = max_length
    _worker_state["last_turn_loss_only"] = last_turn_loss_only
    _worker_state["min_loss_tokens"] = min_loss_tokens
    _worker_state["drop_overlong"] = drop_overlong
    _worker_state["max_prompt_tokens"] = max_prompt_tokens
    _worker_state["thinking"] = thinking


def _resolve_last_turn_loss_only(messages):
    """Resolve last_turn_loss_only, supporting "auto" mode.

    When "auto", returns True if the conversation contains real thinking
    content (non-empty <think> blocks), so that loss is only computed on
    the last assistant turn where the model actually reasons.

    Ported from torchspec/data/dataset.py::_resolve_last_turn_loss_only.
    """
    ltlo = _worker_state.get("last_turn_loss_only", False)
    if ltlo == "auto":
        return has_thinking_content(messages)
    return bool(ltlo)


# Far past any real window, so `truncation=True` becomes a no-op and the true
# length is observable.
_PROBE_MAX_LENGTH = 1 << 20

DROP_NO_LOSS = "no_loss"
DROP_OVERLONG = "overlong"
DROP_NO_PROMPT = "no_prompt"
DROP_PROMPT_TOO_LONG = "prompt_too_long"
_DROP_REASONS = (DROP_NO_LOSS, DROP_OVERLONG, DROP_NO_PROMPT, DROP_PROMPT_TOO_LONG)


def _tokenize_single(messages):
    """Worker function — tokenize one sample, return dict or a drop reason."""
    parser = _worker_state["parser"]
    max_length = _worker_state["max_length"]
    min_loss_tokens = _worker_state.get("min_loss_tokens", 1)
    drop_overlong = _worker_state.get("drop_overlong", False)
    max_prompt_tokens = _worker_state.get("max_prompt_tokens", 0)
    thinking = _worker_state.get("thinking", True)
    supports_thinking = _worker_state.get("supports_thinking", False)

    if not thinking and not supports_thinking:
        raise RuntimeError(
            f"{type(parser).__name__} has no thinking switch; "
            "dataset.thinking=false needs chat_template=kimi-k3."
        )
    thinking_kwargs = {"thinking": thinking} if supports_thinking else {}

    messages = normalize_conversation(messages)
    last_turn_only = _resolve_last_turn_loss_only(messages)

    # Both the training sampler and the eval cache slice to max_length - 1, so a
    # sample of exactly max_length still loses its last token. That is the bar.
    usable = max(1, max_length - 1)
    # With max_prompt_tokens set, the reference answer is regenerated and never
    # trained on, so its length must not decide whether the sample survives —
    # only the prompt has to fit. Applying both would throw away short-prompt
    # samples purely for having a long stored answer.
    check_total = drop_overlong and max_prompt_tokens <= 0
    # Parse without truncation whenever a length filter is active. Truncating
    # first makes a long conversation lose its assistant span and then fail the
    # empty-loss-mask check, which reads as a malformed sample when it is really
    # just a long one — 57k samples were rejected that way.
    filtering = drop_overlong or max_prompt_tokens > 0
    parse_length = _PROBE_MAX_LENGTH if filtering else max_length
    input_ids, loss_mask = parser.parse(
        messages, parse_length, last_turn_only=last_turn_only, **thinking_kwargs
    )

    if check_total and len(input_ids) > usable:
        return DROP_OVERLONG

    if min_loss_tokens >= 0 and loss_mask.sum() < max(1, min_loss_tokens):
        return DROP_NO_LOSS

    prompt_ids = None
    if max_prompt_tokens > 0:
        build_prompt = getattr(parser, "parse_generation_prompt", None)
        if build_prompt is None:
            raise RuntimeError(
                f"{type(parser).__name__} cannot build generation prompts; "
                "dataset.max_prompt_tokens needs chat_template=kimi-k3."
            )
        prompt_ids = build_prompt(messages, **thinking_kwargs)
        if len(prompt_ids) == 0:
            return DROP_NO_PROMPT
        if len(prompt_ids) > max_prompt_tokens:
            return DROP_PROMPT_TOO_LONG

    formatted_text = parser.format(messages, **thinking_kwargs)

    # The checks above needed true lengths, but nothing downstream reads past the
    # window: prefill slices to `usable`, and generate mode ignores these ids
    # entirely in favour of prompt_ids. Trim so the cache does not carry 15k-token
    # sequences it will never use.
    if filtering:
        input_ids = input_ids[:usable]
        loss_mask = loss_mask[:usable]

    packed = pack_loss_mask(loss_mask)
    result = {
        "input_ids": input_ids.tolist(),
        "packed_loss_mask": serialize_packed_loss_mask(packed),
        "formatted_prompt": formatted_text,
    }
    if prompt_ids is not None:
        result["prompt_ids"] = prompt_ids.tolist()
    return result


def _drop_out_of_vocab_samples(data: list, tokenizer_path: str) -> list:
    """Drop samples holding token ids the model has no embedding row for.

    K3's tokenizer registers ChatML markers (<|im_end|>, <|im_user|>, ...) as
    added tokens at ids 163840-163846, past the last row of an embedding table
    that stops at 163839. A conversation quoting one of those markers as
    literal text therefore tokenizes into an id the gather cannot resolve, and
    on ROCm that surfaces as a queue abort with no Python frame -- so it has to
    be caught here rather than diagnosed later.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # The added markers sit above len(), which tracks the embedding table.
    limit = len(tokenizer)

    kept = []
    offenders = {}
    for item in data:
        bad = set()
        for field in ("input_ids", "prompt_ids"):
            ids = item.get(field)
            if ids is None:
                continue
            bad.update(i for i in ids if i < 0 or i >= limit)
        if bad:
            for i in bad:
                offenders[i] = offenders.get(i, 0) + 1
            continue
        kept.append(item)

    if offenders:
        named = {}
        for i in sorted(offenders):
            try:
                named[i] = tokenizer.convert_ids_to_tokens(i)
            except Exception:  # noqa: BLE001 - id is outside the tokenizer too
                named[i] = "<unknown>"
        logger.warning(
            "Dropped %d/%d samples containing token ids >= %d, which the model "
            "has no embedding row for: %s",
            len(data) - len(kept), len(data), limit, named,
        )

    return kept


def load_and_preprocess_dataset(
    dataset_path: str,
    tokenizer_path: str,
    max_length: int,
    chat_template: str = "kimi-k25",
    seed: int = 42,
    last_turn_loss_only: str = "false",
    min_loss_tokens: int = 0,
    num_workers: int = 16,
    cache_dir: str = "/dev/shm/lumenrl_cache",
    dataset_split: str = "train",
    drop_overlong: bool = False,
    max_prompt_tokens: int = 0,
    thinking: bool = True,
) -> list:
    """Load, tokenize, and cache a conversation dataset.

    Data processing pipeline ported from TorchSpec
    (torchspec/data/dataset.py::load_conversation_dataset).

    Args:
        last_turn_loss_only: "true", "false", or "auto". When "auto", only
            computes loss on the last assistant turn for samples that contain
            real <think> content.
        min_loss_tokens: Skip samples with fewer supervised tokens than this.

    Returns list of dicts with keys: input_ids (List[int]), packed_loss_mask (str).
    """
    if isinstance(last_turn_loss_only, bool):
        last_turn_loss_only = str(last_turn_loss_only).lower()

    file_stat = ""
    if os.path.isfile(dataset_path):
        st = os.stat(dataset_path)
        file_stat = f"-{st.st_size}-{st.st_mtime}"
    cache_params = (
        f"{os.path.basename(dataset_path)}-{dataset_path}{file_stat}"
        f"-{tokenizer_path}-{max_length}-{chat_template}"
        f"-ltlo={last_turn_loss_only}-mlt={min_loss_tokens}"
        f"-split={dataset_split}-pv={PARSER_VERSION}"
        f"-drop={drop_overlong}-mpt={max_prompt_tokens}-think={thinking}"
    )
    cache_key = hashlib.md5(cache_params.encode()).hexdigest()
    cache_subdir = os.path.join(cache_dir, "tokenized_dataset")
    cache_path = os.path.join(cache_subdir, f"{cache_key}.pt")

    if os.path.exists(cache_path):
        logger.info("Loading preprocessed dataset from cache: %s", cache_path)
        data = torch.load(cache_path, weights_only=False)
        logger.info("Loaded %d cached samples", len(data))
        return _drop_out_of_vocab_samples(data, tokenizer_path)

    logger.info("Preprocessing dataset (cache will be saved to %s)", cache_path)

    from datasets import load_dataset as _load_dataset

    if os.path.isfile(dataset_path):
        if dataset_path.endswith((".jsonl", ".json")):
            ds = _load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_path.endswith(".parquet"):
            ds = _load_dataset("parquet", data_files=dataset_path, split="train")
        else:
            ds = _load_dataset(dataset_path, split=dataset_split)
    elif os.path.isdir(dataset_path):
        ds = _load_dataset(dataset_path, split=dataset_split)
    else:
        ds = _load_dataset(dataset_path, split=dataset_split)

    ds = ds.shuffle(seed=seed)

    raw_conversations = []
    for sample in tqdm(ds, desc="Loading samples"):
        convs = sample.get("conversations") or sample.get("messages")
        if convs and isinstance(convs, list):
            raw_conversations.append(convs)

    logger.info("Loaded %d samples, tokenizing with %d workers...", len(raw_conversations), num_workers)

    worker_args = (
        tokenizer_path, max_length, last_turn_loss_only, min_loss_tokens,
        chat_template, drop_overlong, max_prompt_tokens, thinking,
    )
    if num_workers <= 1:
        _init_worker(*worker_args)
        results = [_tokenize_single(c) for c in tqdm(raw_conversations, desc="Tokenizing")]
    else:
        with mp.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=worker_args,
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(_tokenize_single, raw_conversations, chunksize=64),
                    total=len(raw_conversations),
                    desc="Tokenizing",
                )
            )

    data = []
    dropped = {reason: 0 for reason in _DROP_REASONS}
    unbalanced_think = 0
    for r in results:
        if r is None:
            dropped[DROP_NO_LOSS] += 1
            continue
        if isinstance(r, str):
            dropped[r] = dropped.get(r, 0) + 1
            continue
        if has_unbalanced_thinking_tags(r.get("formatted_prompt", "")):
            unbalanced_think += 1
        entry = {
            "input_ids": r["input_ids"],
            "packed_loss_mask": r["packed_loss_mask"],
        }
        if "prompt_ids" in r:
            entry["prompt_ids"] = r["prompt_ids"]
        data.append(entry)

    total_dropped = sum(dropped.values())
    if total_dropped:
        logger.warning(
            "Dropped %d/%d samples: no_loss=%d overlong=%d (> %d tokens) "
            "no_prompt=%d prompt_too_long=%d (> %d tokens); kept %d",
            total_dropped, len(results),
            dropped[DROP_NO_LOSS], dropped[DROP_OVERLONG], max(1, max_length - 1),
            dropped[DROP_NO_PROMPT], dropped[DROP_PROMPT_TOO_LONG], max_prompt_tokens,
            len(data),
        )

    if unbalanced_think:
        logger.warning(
            "%d/%d samples have unbalanced <think>/<​/think> tags "
            "after chat-template formatting. This usually means the data was generated by a "
            "thinking model that emits the opening <think> in the generation prompt, so the saved "
            "assistant content lacks it and re-tokenization produces malformed turns. Restore the "
            "opening <think> in the data (or verify the chat template) before training.",
            unbalanced_think, len(data),
        )

    data = _drop_out_of_vocab_samples(data, tokenizer_path)

    os.makedirs(cache_subdir, exist_ok=True)
    torch.save(data, cache_path)
    logger.info("Saved %d preprocessed samples to %s", len(data), cache_path)

    return data
