#!/usr/bin/env python3
"""Minimal ATOM/vLLM fp8 rollout KL reproducer.

This script deliberately avoids RLTrainer, Ray worker groups, optimizer state,
reward filtering, and training. It reproduces the first-step rollout-correction
KL shape with three isolated subprocess phases:

1. Generate token-in rollout samples with ATOM fp8.
2. Generate token-in rollout samples with vLLM fp8.
3. Re-score each generated sequence with the Lumen FP8 actor forward.

The same generated sequences can be scored with actor_mode={atom,vllm} to
separate rollout-engine differences from training-side Lumen patch differences.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _json_default(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _extract_prompt_text(row: dict[str, Any], tokenizer) -> str:
    import json as _json

    prompt_raw = row.get("prompt") or row.get("question") or row.get("input") or ""
    if isinstance(prompt_raw, list):
        prompt_text = "\n".join(m.get("content", "") for m in prompt_raw if isinstance(m, dict))
    elif isinstance(prompt_raw, str) and prompt_raw.startswith("["):
        try:
            msgs = _json.loads(prompt_raw)
            prompt_text = "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict))
        except (_json.JSONDecodeError, TypeError):
            prompt_text = prompt_raw
    else:
        prompt_text = str(prompt_raw)

    if hasattr(tokenizer, "apply_chat_template") and isinstance(prompt_raw, list):
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_raw, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return prompt_text


def select_prompt_token_ids(args: argparse.Namespace) -> list[list[int]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = load_dataset("parquet", data_files=args.dataset, split="train")
    gen = torch.Generator()
    gen.manual_seed(int(args.seed))
    perm = torch.randperm(len(ds), generator=gen).tolist()
    ids_list: list[list[int]] = []
    for idx in perm[: args.num_prompts]:
        prompt = _extract_prompt_text(ds[int(idx)], tokenizer)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ids_list.append([int(x) for x in ids])
    expanded = []
    for ids in ids_list:
        expanded.extend([ids] * args.num_generations)
    return expanded


async def _run_atom(args: argparse.Namespace, prompt_ids: list[list[int]]) -> list[dict[str, Any]]:
    from lumenrl.engine.inference.atom_ray_server import ATOMRayServer

    engine_kwargs = {
        "model": args.model,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": True,
        "trust_remote_code": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "kv_cache_dtype": "bf16",
        "online_quant_config": {"global_quant_config": args.atom_quant},
        "master_addr": _node_ip(),
        "port": _free_port(),
    }
    server = ATOMRayServer(args.model, engine_kwargs, replica_rank=0, base_seed=args.seed)
    await server.launch()
    try:
        return await server.generate_batch(prompt_ids, _sampling_params(args))
    finally:
        await server.shutdown()


async def _run_vllm(args: argparse.Namespace, prompt_ids: list[list[int]]) -> list[dict[str, Any]]:
    # Match the native vLLM fp8 runbook path: vLLM must not auto-load ATOM's
    # vLLM plugin just because ATOM is on PYTHONPATH for the ATOM phase.
    os.environ["ATOM_DISABLE_VLLM_PLUGIN"] = "1"
    from lumenrl.engine.inference.vllm_ray_server import VLLMRayServer

    engine_kwargs = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "trust_remote_code": True,
        "enable_sleep_mode": False,
        "disable_log_stats": True,
        "max_model_len": args.max_model_len,
        "quantization": "fp8_per_block",
    }
    server = VLLMRayServer(args.model, engine_kwargs, replica_rank=0, base_seed=args.seed)
    await server.launch()
    try:
        return await server.generate_batch(prompt_ids, _sampling_params(args))
    finally:
        await server.shutdown()


def _node_ip() -> str:
    return socket.gethostbyname(socket.gethostname())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "logprobs": 0,
        "seed": args.seed,
    }


def rollout_phase(args: argparse.Namespace) -> None:
    prompts = select_prompt_token_ids(args)
    started = time.time()
    if args.backend == "atom":
        results = asyncio.run(_run_atom(args, prompts))
    elif args.backend == "vllm":
        results = asyncio.run(_run_vllm(args, prompts))
    else:
        raise ValueError(args.backend)
    payload = {
        "backend": args.backend,
        "args": vars(args),
        "elapsed_s": time.time() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
    print(f"[rollout:{args.backend}] wrote {args.output} n={len(results)} elapsed={payload['elapsed_s']:.2f}s")


def configure_actor_env(actor_mode: str) -> None:
    os.environ.setdefault("LUMEN_FP8", "1")
    os.environ.setdefault("FP8_PARAM_MANAGER", "0")
    os.environ.setdefault("LUMEN_NORM", "1")
    os.environ.setdefault("LUMEN_FP8_SCALING", "blockwise2d")
    os.environ.setdefault("LUMEN_FP8_FORMAT", "fp8_e4m3")
    os.environ.setdefault("LUMEN_FP8_BLOCK_SIZE", "128")
    os.environ.setdefault("LUMEN_FP8_ATTN", "none")
    os.environ.setdefault("LUMEN_DISABLE_HF_ATTN_PATCH", "1")
    if actor_mode == "atom":
        os.environ["LUMEN_ROLLOUT"] = "ATOM"
    elif actor_mode == "vllm":
        os.environ.pop("LUMEN_ROLLOUT", None)
    else:
        raise ValueError(actor_mode)


def load_actor_model(args: argparse.Namespace):
    from lumenrl.engine.training.fsdp_backend import _apply_lumen_fp8, _load_hf_model

    configure_actor_env(args.actor_mode)
    model = _load_hf_model(args.model, torch.bfloat16)
    model = _apply_lumen_fp8(
        model,
        {
            "fp8": True,
            "lumen_norm": True,
            "rollout": "ATOM" if args.actor_mode == "atom" else "",
        },
    )
    model.eval().to("cuda")
    return model


@torch.no_grad()
def actor_response_logprobs(model, prompt_ids: list[int], response_ids: list[int]) -> list[float]:
    from lumenrl.engine.training.packing import (
        PackingContext,
        pack_sequences,
        packed_token_log_probs,
        unpack_log_probs,
    )

    seq = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device="cuda")
    mask = torch.ones_like(seq)
    packed = pack_sequences(seq, mask)
    with PackingContext(packed.cu_seqlens, packed.max_seqlen):
        out = model(
            input_ids=packed.input_ids,
            position_ids=packed.position_ids,
            attention_mask=None,
            use_cache=False,
        )
        logits = out.logits if hasattr(out, "logits") else out
        logits = logits.squeeze(0)
        flat_lp = packed_token_log_probs(
            logits,
            packed.input_ids.squeeze(0),
            packed.cu_seqlens,
            temperature=1.0,
        )
        token_lp = unpack_log_probs(flat_lp, packed.cu_seqlens, packed.seq_lens, seq.shape[1])
    start = max(0, len(prompt_ids) - 1)
    end = start + len(response_ids)
    return [float(x) for x in token_lp[0, start:end].detach().cpu().tolist()]


def score_phase(args: argparse.Namespace) -> None:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    model = load_actor_model(args)
    scored = []
    all_deltas = []
    for item in payload["results"]:
        pids = [int(x) for x in item["prompt_token_ids"]]
        rids = [int(x) for x in item["token_ids"]]
        rollout_lp = item.get("logprobs") or []
        actor_lp = actor_response_logprobs(model, pids, rids)
        n = min(len(rollout_lp), len(actor_lp), len(rids))
        delta = [float(rollout_lp[i]) - float(actor_lp[i]) for i in range(n)]
        if n:
            all_deltas.extend(delta)
        scored.append(
            {
                "prompt_len": len(pids),
                "response_len": len(rids),
                "n_logprobs": n,
                "kl_mean": sum(delta) / max(n, 1),
                "kl_max": max(delta) if delta else None,
                "kl_min": min(delta) if delta else None,
                "rollout_lp_mean": sum(rollout_lp[:n]) / max(n, 1),
                "actor_lp_mean": sum(actor_lp[:n]) / max(n, 1),
            }
        )
    summary = summarize_deltas(all_deltas)
    out = {
        "backend": payload["backend"],
        "actor_mode": args.actor_mode,
        "summary": summary,
        "per_sequence": scored,
        "source": str(args.input),
    }
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[score:{payload['backend']} actor={args.actor_mode}] {json.dumps(summary, sort_keys=True)} -> {args.output}")


def summarize_deltas(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"tokens": 0, "mean": None, "min": None, "max": None, "abs_mean": None}
    return {
        "tokens": len(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
        "abs_mean": sum(abs(x) for x in xs) / len(xs),
        "gt_0p01_frac": sum(1 for x in xs if x > 0.01) / len(xs),
    }


def orchestrate(args: argparse.Namespace) -> None:
    work = args.output_dir
    work.mkdir(parents=True, exist_ok=True)
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--seed",
        str(args.seed),
        "--num-prompts",
        str(args.num_prompts),
        "--num-generations",
        str(args.num_generations),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
    ]
    rollout_files = {}
    for backend in ("atom", "vllm"):
        out = work / f"{backend}_rollout.json"
        rollout_files[backend] = out
        subprocess.run(base + ["rollout", "--backend", backend, "--output", str(out)], check=True)

    score_files = []
    for backend, rollout_file in rollout_files.items():
        for actor_mode in ("atom", "vllm"):
            out = work / f"{backend}_scored_by_{actor_mode}.json"
            score_files.append(out)
            subprocess.run(
                base
                + [
                    "score",
                    "--input",
                    str(rollout_file),
                    "--actor-mode",
                    actor_mode,
                    "--output",
                    str(out),
                ],
                check=True,
            )

    rows = []
    for file in score_files:
        data = json.loads(file.read_text(encoding="utf-8"))
        rows.append(
            {
                "backend": data["backend"],
                "actor_mode": data["actor_mode"],
                **data["summary"],
            }
        )
    report = {"args": vars(args), "rows": rows}
    report_path = work / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[done] summary={report_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["run", "rollout", "score"])
    p.add_argument("--model", default=os.environ.get("MODEL_PATH", "/mnt/shengnxu/xysheng/working/data/models/Qwen3-8B-Base"))
    p.add_argument("--dataset", default=os.environ.get("TRAIN_FILE", "/mnt/shengnxu/xysheng/working/data/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet"))
    p.add_argument("--seed", type=int, default=10086)
    p.add_argument("--num-prompts", type=int, default=2)
    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--atom-quant", default="per_block_fp8")
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/lumenrl_atom_vllm_kl_repro"))
    p.add_argument("--backend", choices=["atom", "vllm"])
    p.add_argument("--actor-mode", choices=["atom", "vllm"], default="atom")
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "run":
        orchestrate(args)
    elif args.phase == "rollout":
        if args.backend is None or args.output is None:
            raise SystemExit("rollout requires --backend and --output")
        rollout_phase(args)
    elif args.phase == "score":
        if args.input is None or args.output is None:
            raise SystemExit("score requires --input and --output")
        score_phase(args)


if __name__ == "__main__":
    main()
