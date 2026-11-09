"""Close one RSI generation: train role LoRA, merge, serve, roll the challenger.

The trainable update is the existing Qwen3 MoE LoRA SFT trainer. It consumes
answer-masked specialist turns. Each role is merged into its own checkpoint
and served on its own endpoint, then the next generation is rolled out against
the current champion.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from micake_rsi.guards import UNTRUSTED
from micake_rsi.sft_data import ChatTokenizer, write_role_sft

logger = logging.getLogger(__name__)

MERGE_MANIFEST = "merge_manifest.json"

TrainFn = Callable[[Path], None]
MergeFn = Callable[[Path, Path, Path], None]
ServeFn = Callable[[Mapping[str, Any]], None]
RolloutFn = Callable[[Mapping[str, Any]], dict[str, Any]]


def load_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def training_config(
    train: Mapping[str, Any],
    *,
    role: str,
    base_model: str,
    data_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """SFT config for one role, composed over the pinned hardware and recipe profiles."""
    return {
        "profiles": {
            "hardware": train["hardware_profile"],
            "recipe": train["recipe_profile"],
        },
        "model": {
            "name_or_path": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "local_path": base_model,
            "revision": train["model_revision"],
        },
        "precision": {"fp8_format": "e4m3fn"},
        "distributed": {"use_fsdp": False, "expert_backend": "sonic", "timeout_seconds": 7200},
        "optimizations": {"aiter_attn": False, "fused_router": False, "moe_global_expert_layout": True},
        "data": {
            "train_path": manifest["train_path"],
            "validation_path": None,
            "max_sequence_length": int(train["max_sequence_length"]),
            "pad_token_id": 151643,
            "num_workers": 0,
            "spread_long_sequences": True,
        },
        "training": {
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "selective_gc": True,
            "moe_activation_offload": True,
            "lm_head_chunk_size": 1024,
            "max_steps": int(train["max_steps"]),
            "evaluation_interval": 0,
            "log_interval": 1,
        },
        "output": {"adapter_dir": str(data_dir / "adapter"), "resume_adapter_dir": None},
        "execution": {
            "allow_training": True,
            "data_manifest_path": manifest["manifest_path"],
            "tokenizer_sha256": train["tokenizer_sha256"],
            "data_manifest_sha256": manifest["manifest_sha256"],
        },
        "role": role,
    }


def challenger_config(
    cfg: Mapping[str, Any],
    parent_id: str,
    generation_id: str,
    served: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Next-generation experiment config pointing specialists at their merged checkpoints."""
    nxt = copy.deepcopy(dict(cfg))
    train = nxt["train"]
    nxt["rsi"]["generation_id"] = generation_id
    nxt["rsi"]["parent_generation_id"] = parent_id
    nxt["rsi"]["episodes_per_task"] = int(train["challenger_episodes"])
    for role, spec in nxt["specialists"].items():
        update = served[role]
        version = str(int(spec["binding"]["adapter_version"]) + 1)
        spec["binding"].update({
            "adapter_id": f"{role}-lora",
            "adapter_version": version,
            "policy_version": f"{role}-{generation_id}",
        })
        spec["endpoint"]["base_url"] = f"http://127.0.0.1:{update['port']}/v1"
        spec["endpoint"]["model"] = update["checkpoint"]
    return nxt


def _generation_id(parent_id: str) -> str:
    if parent_id.startswith("gen") and parent_id[3:].isdigit():
        return f"gen{int(parent_id[3:]) + 1}"
    return f"{parent_id}-challenger"


def prepare_updates(
    cfg: Mapping[str, Any],
    tokenizer: ChatTokenizer,
) -> list[dict[str, Any]]:
    """Build one training job per trainable role that has a positive-advantage turn."""
    train = cfg["train"]
    root = Path(cfg["rsi"]["root"]) / cfg["rsi"]["generation_id"]
    jobs: list[dict[str, Any]] = []
    for role, spec in cfg["specialists"].items():
        if not spec["binding"]["trainable"]:
            continue
        data_dir = root / "train" / role
        samples = load_samples(root / "rl" / f"{role}.jsonl")
        manifest = write_role_sft(
            [s for s in samples if s.get("manager_verification") not in UNTRUSTED],
            tokenizer,
            data_dir,
            min_advantage=float(train["min_advantage"]),
            max_sequence_length=int(train["max_sequence_length"]),
            tokenizer_sha256=train["tokenizer_sha256"],
            tokenizer_path=spec["endpoint"]["model"],
        )
        job: dict[str, Any] = {"role": role, "rows": manifest["artifacts"]["train"]["rows"], "data_dir": str(data_dir)}
        if job["rows"] == 0:
            job["skipped"] = True
            job["checkpoint"] = spec["endpoint"]["model"]
            logger.info("role %s has no positive-advantage turns; challenger keeps %s", role, job["checkpoint"])
        else:
            config = training_config(
                train, role=role, base_model=train["base_model"], data_dir=data_dir, manifest=manifest,
            )
            config_path = data_dir / "sft.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            job.update({"skipped": False, "config_path": str(config_path), "adapter_dir": config["output"]["adapter_dir"]})
        job["port"] = int(train["serve"][role]["port"])
        jobs.append(job)
    return jobs


def run_cycle(
    cfg: Mapping[str, Any],
    tokenizer: ChatTokenizer,
    train_role: TrainFn,
    merge_role: MergeFn,
    serve: ServeFn,
    rollout: RolloutFn,
) -> dict[str, Any]:
    """Train, serve and roll out the challenger generation.

    Args:
        cfg: Champion generation config. Its generation must already be recorded.
        tokenizer: Chat tokenizer for the specialist base model.
        train_role: Runs one role SFT config to completion.
        merge_role: Merges ``(base, adapter, output)``.
        serve: Starts one endpoint per role from ``{"role", "checkpoint", "port", "gpus"}``.
        rollout: Runs one generation and returns its manifest.

    Returns:
        The challenger generation manifest.
    """
    parent_id = cfg["rsi"]["generation_id"]
    generation_id = _generation_id(parent_id)
    jobs = prepare_updates(cfg, tokenizer)
    served: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if job["skipped"]:
            checkpoint = job["checkpoint"]
        else:
            checkpoint_dir = Path(cfg["rsi"]["root"]) / generation_id / "checkpoints" / job["role"]
            if (checkpoint_dir / MERGE_MANIFEST).exists():
                logger.info("role %s already merged at %s; skipping train and merge", job["role"], checkpoint_dir)
            else:
                train_role(Path(job["config_path"]))
                merge_role(Path(cfg["train"]["base_model"]), Path(job["adapter_dir"]), checkpoint_dir)
            checkpoint = str(checkpoint_dir)
        serve_cfg = cfg["train"]["serve"][job["role"]]
        served[job["role"]] = {
            "checkpoint": checkpoint,
            "port": job["port"],
            "gpus": serve_cfg["gpus"],
            "tp": int(serve_cfg["tp"]),
        }
    for role, update in served.items():
        serve({**update, "role": role})
    challenger = challenger_config(cfg, parent_id, generation_id, served)
    manifest = rollout(challenger)
    logger.info("challenger %s promotion: %s", generation_id, manifest["promotion"]["decision"])
    return manifest
