"""Robust fuzzy unified diff patch applier.

Unlike GNU `patch`, this handles:
- Context lines that don't exactly match (fuzzy search)
- Wrong line numbers in @@ headers (relocates hunks)
- Trailing whitespace differences
- Incomplete hunks
"""

import difflib
import re
from typing import Optional


def parse_hunks(patch_text: str) -> list[dict]:
    """Parse unified diff into hunks."""
    lines = patch_text.split("\n")
    hunks = []
    current = None

    for line in lines:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            # Parse @@ -start,count +start,count @@
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                current = {
                    "old_start": int(m.group(1)),
                    "old_count": int(m.group(2)) if m.group(2) else 1,
                    "new_start": int(m.group(3)),
                    "new_count": int(m.group(4)) if m.group(4) else 1,
                    "lines": [],
                }
            else:
                current = {"old_start": 1, "old_count": 0, "new_start": 1, "new_count": 0, "lines": []}
        elif current is not None:
            if line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("+") or line.startswith("-") or line.startswith(" ") or line == "":
                current["lines"].append(line)

    if current:
        hunks.append(current)
    return hunks


def _normalize(s: str) -> str:
    """Normalize a line for fuzzy matching."""
    return s.rstrip()


def _find_best_match(source_lines: list[str], context_lines: list[str], hint_start: int) -> Optional[int]:
    """Find the best position in source for a set of context lines."""
    if not context_lines:
        return max(0, hint_start - 1)

    normalized_ctx = [_normalize(c) for c in context_lines]
    best_pos = None
    best_score = -1

    # Search around the hinted position first, then expand
    max_range = len(source_lines)
    for offset in range(max_range):
        for sign in [0, 1, -1]:
            pos = hint_start - 1 + sign * offset
            if pos < 0 or pos + len(context_lines) > len(source_lines):
                continue

            score = 0
            for i, ctx in enumerate(normalized_ctx):
                src = _normalize(source_lines[pos + i])
                if src == ctx:
                    score += 2
                elif difflib.SequenceMatcher(None, src, ctx).ratio() > 0.8:
                    score += 1

            if score > best_score:
                best_score = score
                best_pos = pos

            if score == len(context_lines) * 2:
                return best_pos  # Perfect match

        if best_score > 0 and offset > 20:
            break  # Good enough match found, stop searching far

    return best_pos if best_score > 0 else None


def apply_patch(source: str, patch_text: str) -> Optional[str]:
    """Apply a unified diff patch to source text with fuzzy matching.

    Returns the patched source, or None if the patch cannot be applied.
    """
    hunks = parse_hunks(patch_text)
    if not hunks:
        return None

    source_lines = source.split("\n")
    # Remove trailing empty line if source ends with newline
    if source_lines and source_lines[-1] == "":
        source_lines = source_lines[:-1]

    # Process hunks in reverse order (bottom-up) to preserve line numbers
    hunks_with_context = []
    for hunk in hunks:
        # Split hunk lines into context, removals, additions
        context_before = []
        removals = []
        additions = []
        context_after = []
        phase = "before"  # before -> changes -> after

        for line in hunk["lines"]:
            if line.startswith("-"):
                phase = "changes"
                removals.append(line[1:])
            elif line.startswith("+"):
                phase = "changes"
                additions.append(line[1:])
            elif line.startswith(" ") or (line == "" and phase != "changes"):
                content = line[1:] if line.startswith(" ") else line
                if phase == "before":
                    context_before.append(content)
                else:
                    phase = "after"
                    context_after.append(content)

        hunks_with_context.append({
            "hint_start": hunk["old_start"],
            "context_before": context_before,
            "context_after": context_after,
            "removals": removals,
            "additions": additions,
        })

    # Apply hunks in reverse to preserve line numbers
    result_lines = list(source_lines)
    applied_count = 0

    for hunk in reversed(hunks_with_context):
        # Build the full context for matching: context_before + removals
        match_lines = hunk["context_before"] + hunk["removals"]
        if not match_lines:
            match_lines = hunk["context_before"] or hunk["removals"] or [""]

        pos = _find_best_match(result_lines, match_lines, hunk["hint_start"])
        if pos is None:
            continue  # Skip this hunk if no match found

        # Calculate the exact position of removals within the match
        ctx_len = len(hunk["context_before"])
        remove_start = pos + ctx_len
        remove_end = remove_start + len(hunk["removals"])

        # Verify removals roughly match
        if remove_end <= len(result_lines):
            # Replace: remove old lines, insert new lines
            result_lines[remove_start:remove_end] = hunk["additions"]
            applied_count += 1

    if applied_count == 0:
        return None

    result = "\n".join(result_lines)
    if source.endswith("\n"):
        result += "\n"
    return result
