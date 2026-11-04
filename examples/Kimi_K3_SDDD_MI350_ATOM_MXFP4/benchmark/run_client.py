#!/usr/bin/env python3
"""Measure DSpark acceptance length against a running ATOM server.

Acceptance length is read from the engine's own counters at /debug/mtp_stats, not
inferred from timings. The counters are cumulative since server start, so every
measurement here is a delta between a snapshot taken before the set and one taken
after -- that is what lets several question sets share one server, which matters
because loading Kimi-K3 costs minutes.

    tok/fwd = 1 + accepted_draft_tokens / forward_steps

The 1 is the bonus token the target emits on every step, so tok/fwd is 1.0 even
at zero acceptance and tops out at 1 + num_spec. This is the same scale as the
"acceptance length" column on Inferact/Kimi-K3-DSpark.

Three behaviours exist because of how the gen-3 runs failed. An aiter cross-rank
broadcast deadlock froze the engine mid-set with /health still answering 200, and
the old client held its counters and its per-prompt records in memory only, so
550 finished prompts went in the bin:

  --request-timeout  a single stuck request exits 42 instead of hanging forever
  --rotate-after     hand control back with 43 so the caller can cycle the server
  --resume           skip prompts already in <out>.partial.json

Partial results are flushed every --flush-every prompts, so a kill loses at most
that many.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

EXIT_ENGINE_STUCK = 42
EXIT_ROTATE = 43


def _get_json(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post_json(url: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _find_distribution(obj):
    """Locate the accepted-length histogram anywhere in the stats payload.

    ATOM's shape for this endpoint is not pinned by any contract we control, so
    rather than hardcode a path, find the first mapping whose keys are small
    integers -- that is the {accepted_k: count} histogram. Everything the client
    reports is derivable from it, so this is the only field that has to be found.
    """
    if isinstance(obj, dict):
        keys = list(obj.keys())
        if keys:
            try:
                as_int = [int(k) for k in keys]
            except (TypeError, ValueError):
                as_int = None
            if as_int is not None and all(0 <= k <= 64 for k in as_int) and all(
                isinstance(v, (int, float)) for v in obj.values()
            ):
                return {int(k): float(v) for k, v in obj.items()}
        for v in obj.values():
            found = _find_distribution(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_distribution(v)
            if found:
                return found
    return None


def _delta(before: dict, after: dict) -> dict:
    keys = set(before) | set(after)
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in sorted(keys)}


def _summarise(dist: dict, num_spec: int) -> dict:
    steps = sum(dist.values())
    accepted = sum(k * v for k, v in dist.items())
    return {
        "forward_steps": int(steps),
        "accepted_draft_tokens": int(accepted),
        "tok_per_fwd": (1.0 + accepted / steps) if steps else 0.0,
        "acceptance_rate": (accepted / (steps * num_spec)) if steps else 0.0,
        "accepted_length_distribution": {
            str(k): int(v) for k, v in sorted(dist.items())
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="Kimi-K3")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--num-prompts", type=int, default=0, help="0 = all")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--num-spec", type=int, default=7)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--rotate-after", type=int, default=0, help="0 = never")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush-every", type=int, default=25)
    a = ap.parse_args()

    stats_url = f"{a.base_url}/debug/mtp_stats"
    chat_url = f"{a.base_url}/v1/chat/completions"
    partial_path = a.out + ".partial.json"

    rows = []
    with open(a.questions) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if a.num_prompts:
        rows = rows[: a.num_prompts]

    done: dict[str, dict] = {}
    base_dist: dict | None = None
    if a.resume and os.path.exists(partial_path):
        with open(partial_path) as f:
            saved = json.load(f)
        done = {r["id"]: r for r in saved.get("per_prompt", [])}
        base_dist = {int(k): float(v) for k, v in
                     (saved.get("cumulative_distribution") or {}).items()}
        print(f"resuming: {len(done)} prompts already recorded", flush=True)

    try:
        raw = _get_json(stats_url, timeout=30)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"cannot read {stats_url}: {exc}", file=sys.stderr)
        return 1
    start_dist = _find_distribution(raw)
    if start_dist is None:
        print("could not find an accepted-length histogram in /debug/mtp_stats.\n"
              "raw payload follows so the parser can be fixed:\n"
              + json.dumps(raw, indent=2)[:4000], file=sys.stderr)
        return 1

    cumulative = dict(base_dist) if base_dist else {}
    snapshot = start_dist
    per_prompt = list(done.values())
    t_start = time.time()
    submitted = 0
    rotate = False

    def flush(final: bool):
        payload = {
            "label": a.label,
            "questions": a.questions,
            "model": a.model,
            "protocol": {
                "max_tokens": a.max_tokens,
                "temperature": a.temperature,
                "top_p": a.top_p,
                "num_speculative_tokens": a.num_spec,
                "turns_used": 1,
                "concurrency": 1,
            },
            "n": len(per_prompt),
            "cumulative_distribution": {str(k): int(v) for k, v in
                                        sorted(cumulative.items())},
            "per_prompt": per_prompt,
        }
        if cumulative:
            payload.update(_summarise(cumulative, a.num_spec))
            lens = [p["completion_tokens"] for p in per_prompt if p.get("completion_tokens")]
            if lens:
                ordered = sorted(lens)
                payload["completion_tokens"] = {
                    "mean": sum(lens) / len(lens),
                    "median": ordered[len(ordered) // 2],
                    "max": ordered[-1],
                    "capped": sum(1 for x in lens if x >= a.max_tokens - 16),
                }
            elapsed = time.time() - t_start
            payload["elapsed_s"] = elapsed
            if elapsed > 0:
                payload["tok_per_s"] = sum(lens) / elapsed if lens else 0.0
        with open(partial_path if not final else a.out, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    for row in rows:
        qid = str(row["id"])
        if qid in done:
            continue
        prompt = row["turns"][0]
        body = {
            "model": a.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": a.max_tokens,
            "temperature": a.temperature,
        }
        if a.top_p != 1.0:
            body["top_p"] = a.top_p
        t0 = time.time()
        try:
            resp = _post_json(chat_url, body, timeout=a.request_timeout)
        except (TimeoutError, urllib.error.URLError) as exc:
            print(f"request for {qid} did not return within "
                  f"{a.request_timeout}s ({exc}); treating the engine as stuck",
                  file=sys.stderr)
            flush(False)
            return EXIT_ENGINE_STUCK
        dt = time.time() - t0
        usage = resp.get("usage") or {}
        per_prompt.append({
            "id": qid,
            "seconds": round(dt, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
        })
        submitted += 1

        if submitted % a.flush_every == 0 or (
            a.rotate_after and submitted >= a.rotate_after
        ):
            try:
                now = _find_distribution(_get_json(stats_url, timeout=60)) or snapshot
            except (urllib.error.URLError, TimeoutError):
                now = snapshot
            step = _delta(snapshot, now)
            for k, v in step.items():
                cumulative[k] = cumulative.get(k, 0.0) + v
            snapshot = now
            payload = flush(False)
            print(f"[{a.label}] {len(per_prompt)}/{len(rows)} "
                  f"tok/fwd={payload.get('tok_per_fwd', 0):.4f} "
                  f"accept={100 * payload.get('acceptance_rate', 0):.2f}% "
                  f"steps={payload.get('forward_steps', 0)}", flush=True)
            if a.rotate_after and submitted >= a.rotate_after:
                rotate = True
                break

    try:
        now = _find_distribution(_get_json(stats_url, timeout=60)) or snapshot
        step = _delta(snapshot, now)
        for k, v in step.items():
            cumulative[k] = cumulative.get(k, 0.0) + v
    except (urllib.error.URLError, TimeoutError):
        pass

    if rotate:
        flush(False)
        print(f"[{a.label}] rotating after {submitted} prompts", flush=True)
        return EXIT_ROTATE

    payload = flush(True)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"\n=== {a.label} ===")
    print(f"prompts        : {payload['n']}")
    print(f"tok/fwd        : {payload.get('tok_per_fwd', 0):.4f}")
    print(f"acceptance     : {100 * payload.get('acceptance_rate', 0):.2f}%")
    print(f"forward steps  : {payload.get('forward_steps', 0)}")
    ct = payload.get("completion_tokens", {})
    if ct:
        print(f"completion tok : mean {ct['mean']:.0f} median {ct['median']} "
              f"max {ct['max']} capped {ct['capped']}")
    print(f"tok/s          : {payload.get('tok_per_s', 0):.2f}")
    print(f"distribution   : {payload.get('accepted_length_distribution')}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
