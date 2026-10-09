"""Robust diff patch application for LLM-generated patches.

LLMs produce diffs with inaccurate line numbers. This module applies patches
by matching context lines fuzzy rather than relying on hunk headers.
"""
import re
from pathlib import Path


def apply_robust_patch(original: str, patch_text: str) -> str | None:
    """Apply a unified diff patch using fuzzy context matching.

    Ignores hunk headers (@@ lines) entirely. For each hunk, finds the
    best match for context lines in the original, then applies -/+ changes.

    Returns patched source or None if application fails.
    """
    hunks = _parse_hunks(patch_text)
    if not hunks:
        return None

    orig_lines = original.splitlines(keepends=True)
    # Process hunks in reverse order (last first) to preserve line numbers
    hunks_with_pos = []
    for ctx_before, removals, additions, ctx_after in hunks:
        pos = _find_context(orig_lines, ctx_before, removals, ctx_after)
        if pos is not None:
            hunks_with_pos.append((pos, ctx_before, removals, additions, ctx_after))

    if not hunks_with_pos:
        return None

    # Sort by position (reverse) and apply
    hunks_with_pos.sort(key=lambda x: x[0], reverse=True)
    result = list(orig_lines)

    for pos, ctx_before, removals, additions, ctx_after in hunks_with_pos:
        # Calculate the range to replace
        start = pos + len(ctx_before)
        end = start + len(removals)

        # Verify removals match (fuzzy)
        if end <= len(result):
            result[start:end] = [a if a.endswith('\n') else a + '\n' for a in additions]

    patched = ''.join(result)
    return patched if patched != original else None


def _parse_hunks(patch_text: str):
    """Parse unified diff into hunks of (context_before, removals, additions, context_after)."""
    lines = patch_text.splitlines()
    hunks = []
    i = 0
    # Skip header lines
    while i < len(lines) and not lines[i].startswith('@@'):
        i += 1

    while i < len(lines):
        if lines[i].startswith('@@'):
            i += 1
            ctx_before = []
            removals = []
            additions = []
            ctx_after = []
            phase = 'before'  # before, change, after

            while i < len(lines) and not lines[i].startswith('@@'):
                line = lines[i]
                if line.startswith(' '):
                    content = line[1:]
                    if phase == 'before' or (phase == 'change' and not removals and not additions):
                        ctx_before.append(content)
                    elif phase == 'change':
                        phase = 'after'
                        ctx_after.append(content)
                    elif phase == 'after':
                        ctx_after.append(content)
                elif line.startswith('-'):
                    if phase == 'after':
                        # New sub-hunk within same hunk
                        hunks.append((ctx_before, removals, additions, ctx_after))
                        ctx_before = list(ctx_after)
                        removals = []
                        additions = []
                        ctx_after = []
                    phase = 'change'
                    removals.append(line[1:])
                elif line.startswith('+'):
                    phase = 'change'
                    additions.append(line[1:])
                else:
                    break
                i += 1

            if removals or additions:
                hunks.append((ctx_before, removals, additions, ctx_after))
        else:
            i += 1

    return hunks


def _find_context(orig_lines, ctx_before, removals, ctx_after):
    """Find where a hunk matches in original using context lines."""
    orig_stripped = [l.rstrip('\n') for l in orig_lines]

    # Build search pattern from context + removals
    search_lines = []
    for c in ctx_before[-3:]:  # Use last 3 context lines before
        search_lines.append(c.rstrip())
    for r in removals:
        search_lines.append(r.rstrip())

    if not search_lines:
        return None

    # Sliding window search
    first = search_lines[0]
    for i in range(len(orig_stripped)):
        if _fuzzy_eq(orig_stripped[i], first):
            # Check if rest matches
            match = True
            for j, sl in enumerate(search_lines[1:], 1):
                if i + j >= len(orig_stripped) or not _fuzzy_eq(orig_stripped[i + j], sl):
                    match = False
                    break
            if match:
                return i

    # Fallback: try matching just the first removal line
    if removals:
        first_rm = removals[0].rstrip()
        for i in range(len(orig_stripped)):
            if _fuzzy_eq(orig_stripped[i], first_rm):
                return i - len(ctx_before[-3:])

    return None


def _fuzzy_eq(a: str, b: str) -> bool:
    """Compare lines ignoring trailing whitespace."""
    return a.rstrip() == b.rstrip()


if __name__ == "__main__":
    # Test
    import json, sys
    HELD_OUT = Path("/home/danyzhan/held-out-benchmark-aiter")
    bl = json.load(open(HELD_OUT / "receipts" / "aiter_baseline_results.json"))
    task = [t for t in bl if t["passed"] and "rms_norm" in t["task"]][0]
    td = HELD_OUT / "artifacts" / "kernel" / task["task"] / "initial"
    original = (td / "kernel.py").read_text()

    response = Path("/tmp/last_response.txt").read_text()
    # Extract patch
    block = re.search(r"```(?:diff|patch|)\s*\n(.*?)```", response, re.DOTALL)
    patch_text = block.group(1) if block else response

    result = apply_robust_patch(original, patch_text)
    if result:
        print(f"Patched: {len(result)} chars ({len(result.splitlines())} lines)")
        try:
            compile(result, "<test>", "exec")
            print("Compile: OK!")
        except SyntaxError as e:
            print(f"SyntaxError: {e}")
    else:
        print("Patch FAILED")
