"""Config + setup-time guards for ``algorithm.score_centering``."""

from __future__ import annotations

import pytest

from lumenrl.algorithms.score_centering import validate_score_centering
from lumenrl.core.config import LumenRLConfig, ScoreCenteringConfig


def _sc_config(**sc) -> LumenRLConfig:
    """A config the v1 score-centering path accepts (Megatron + vLLM DAPO)."""
    cfg = LumenRLConfig()
    cfg.algorithm.name = "dapo"
    cfg.policy.training_backend = "megatron_native"
    cfg.policy.generation_backend = "vllm"
    cfg.policy.generation.vllm_cfg.calculate_log_probs = True
    cfg.policy.generation.vllm_cfg.transport = "ray_http"
    cfg.controller.ray.enabled = True
    cfg.algorithm.score_centering = ScoreCenteringConfig(enabled=True, **sc)
    return cfg


def test_defaults_are_off_with_paper_settings() -> None:
    sc = LumenRLConfig().algorithm.score_centering
    assert sc.enabled is False
    assert (sc.top_k, sc.is_mode, sc.tis_clip, sc.eps) == (32, "tis", 2.0, 1e-6)
    assert sc.rho_pin_tail == 1e-5


def test_yaml_round_trip(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "algorithm:\n  name: dapo\n  score_centering:\n    top_k: 64\n    is_mode: icepop\n"
    )
    cfg = LumenRLConfig.from_yaml(path, overrides=["algorithm.score_centering.icepop_high=4.0"])
    sc = cfg.algorithm.score_centering
    assert (sc.enabled, sc.top_k, sc.is_mode, sc.icepop_high) == (False, 64, "icepop", 4.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(top_k=0),
        dict(is_mode="sequence"),
        dict(tis_clip=0.0),
        dict(icepop_low=2.0, icepop_high=1.0),
        dict(eps=0.0),
        dict(chunk_rows=0),
        dict(rho_pin_tail=-1.0),
        dict(base_objective="ppo"),
    ],
)
def test_field_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        ScoreCenteringConfig(**kwargs)


def test_supported_config_passes() -> None:
    validate_score_centering(_sc_config())


def test_disabled_skips_all_checks() -> None:
    cfg = LumenRLConfig()  # atom + fsdp2 + grpo defaults would all be rejected
    validate_score_centering(cfg)


def _set(cfg: LumenRLConfig, dotted: str, value) -> None:
    obj = cfg
    *parents, leaf = dotted.split(".")
    for name in parents:
        obj = getattr(obj, name)
    setattr(obj, leaf, value)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("policy.generation_backend", "atom", "generation_backend"),
        ("policy.training_backend", "none", "training_backend"),
        ("algorithm.name", "ppo", "algorithm.name"),
        ("algorithm.dapo.loss_mode", "gmpo", "gmpo"),
        ("algorithm.loss_mode", "gspo", "loss_mode"),
        ("algorithm.dapo.kl_coeff", 0.1, "kl_coeff"),
        ("algorithm.distillation.enabled", True, "distillation"),
        ("policy.generation.vllm_cfg.calculate_log_probs", False, "calculate_log_probs"),
        ("policy.generation.vllm_cfg.top_p", 0.95, "top_p"),
        ("policy.generation.vllm_cfg.top_k", 50, "top_k"),
        ("policy.generation.vllm_cfg.temperature", -1.0, "temperature"),
        ("quantization.rollout_correction.rollout_is", "sequence", "rollout_is"),
        ("quantization.rollout_correction.rollout_is_batch_normalize", True, "batch_normalize"),
        ("controller.ray.enabled", False, "controller.ray.enabled"),
        ("policy.generation.vllm_cfg.transport", "fifo", "transport"),
    ],
)
def test_unsupported_combinations_fail_at_setup(field, value, match) -> None:
    cfg = _sc_config()
    _set(cfg, field, value)
    with pytest.raises(ValueError, match=match):
        validate_score_centering(cfg)


def test_grpo_and_token_tis_are_allowed() -> None:
    cfg = _sc_config()
    cfg.algorithm.name = "grpo"
    cfg.quantization.rollout_correction.rollout_is = "token"
    cfg.policy.generation.vllm_cfg.top_k = 0
    validate_score_centering(cfg)


def test_config_construction_runs_the_guard(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "policy:\n  training_backend: none\n  generation_backend: vllm\n"
        "algorithm:\n  name: dapo\n  score_centering:\n    enabled: true\n"
    )
    with pytest.raises(ValueError, match="training_backend"):
        LumenRLConfig.from_yaml(path)


def test_support_replay_allows_sampler_top_k_within_the_head() -> None:
    cfg = _sc_config(top_k=32, support_replay=True)
    cfg.policy.generation.vllm_cfg.top_k = 20
    cfg.policy.generation.vllm_cfg.top_p = 0.9
    validate_score_centering(cfg)


@pytest.mark.parametrize(
    "replay, top_k, top_p, match",
    [
        (True, 64, 1.0, "top_k"),       # support can exceed the 32-token head
        (True, 32, 1.0, "top_k"),       # no room for a filler slot (logit ties)
        (True, -1, 1.0, "support_replay"),  # nothing filtered: replay is meaningless
        (False, 20, 1.0, "support_replay"),  # filtered sampling needs replay
        (True, -1, 0.9, "top_p"),       # top-p alone can exceed the head
    ],
)
def test_support_replay_guards(replay, top_k, top_p, match) -> None:
    cfg = _sc_config(top_k=32, support_replay=replay)
    cfg.policy.generation.vllm_cfg.top_k = top_k
    cfg.policy.generation.vllm_cfg.top_p = top_p
    with pytest.raises(ValueError, match=match):
        validate_score_centering(cfg)
