"""DAPO-style math reward: verify boxed answers against ground truth.

Adapted from verl/utils/reward_score/math_dapo.py (Apache-2.0, Bytedance & EleutherAI).
Optionally uses ``math_verify`` for robust symbolic comparison when installed.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import torch

from lumenrl.core.protocol import DataProto

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{")

SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""),
    (" ", ""), ("mbox", "text"), (",\\text{and}", ","),
    ("\\text{and}", ","), ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes",
    "digits", "cents", "degrees", "cm", "gm", "pounds", "meters",
    "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2",
    "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX ``\\boxed{...}`` expression from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx: right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left):-1]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def _try_math_verify(solution_str: str, ground_truth: str) -> Optional[bool]:
    """Attempt to verify using the ``math_verify`` package if available."""
    try:
        from math_verify import parse, verify

        answer = parse(solution_str, parsing_timeout=5)
        gt = parse(ground_truth, parsing_timeout=5)
        return verify(answer, gt, timeout_seconds=5)
    except ImportError:
        return None
    except Exception:
        return None


def compute_score(solution_str: str, ground_truth: str) -> dict:
    """Compute reward for a single (solution, ground_truth) pair.

    Returns dict with keys: score (float), acc (bool), pred (str|None).
    """
    solution_str = solution_str[-300:]

    mv_result = _try_math_verify(solution_str, ground_truth)
    if mv_result is not None:
        return {"score": 1.0 if mv_result else -1.0, "acc": mv_result, "pred": None}

    boxed = last_boxed_only_string(solution_str)
    if boxed is not None:
        pred = normalize_final_answer(remove_boxed(boxed))
    else:
        match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", solution_str)
        pred = normalize_final_answer(match[-1]) if match else "[INVALID]"

    gt_boxed = last_boxed_only_string(ground_truth)
    if gt_boxed is not None:
        gt = normalize_final_answer(remove_boxed(gt_boxed))
    else:
        gt = normalize_final_answer(ground_truth)

    correct = pred == gt
    return {"score": 1.0 if correct else -1.0, "acc": correct, "pred": pred}


def compute_math_reward(
    responses: list[str],
    ground_truths: list[str],
    overlong_buffer: int = 0,
    max_response_len: int = 0,
    overlong_penalty: float = -1.0,
) -> tuple[torch.Tensor, list[dict]]:
    """Batch math reward computation.

    Returns (rewards [B], details [list of dicts]).
    """
    rewards = []
    details = []
    for resp, gt in zip(responses, ground_truths):
        if max_response_len > 0 and len(resp) > max_response_len + overlong_buffer:
            result = {"score": overlong_penalty, "acc": False, "pred": "[OVERLONG]"}
        else:
            result = compute_score(resp, gt)
        rewards.append(result["score"])
        details.append(result)
    return torch.tensor(rewards, dtype=torch.float32), details


def dapo_math_reward(batch: DataProto) -> torch.Tensor:
    """Reward function compatible with ``RewardWorker``'s function-based interface.

    Expects ``batch.meta`` to contain:
      - ``responses``: list[str] — decoded model outputs
      - ``ground_truths``: list[str] — reference answers

    Optional meta keys:
      - ``overlong_buffer``: int (default 0)
      - ``max_response_len``: int (default 0 = no limit)
      - ``overlong_penalty``: float (default -1.0)
    """
    responses = batch.meta.get("responses", [])
    ground_truths = batch.meta.get("ground_truths", [])
    if not responses or not ground_truths:
        b = batch.batch_size
        logger.warning("dapo_math_reward: no responses/ground_truths in meta; returning zeros.")
        return torch.zeros(b, dtype=torch.float32)

    rewards, details = compute_math_reward(
        responses,
        ground_truths,
        overlong_buffer=int(batch.meta.get("overlong_buffer", 0)),
        max_response_len=int(batch.meta.get("max_response_len", 0)),
        overlong_penalty=float(batch.meta.get("overlong_penalty", -1.0)),
    )
    batch.meta["reward_details"] = details
    return rewards
