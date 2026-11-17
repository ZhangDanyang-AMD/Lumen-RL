import json
from pathlib import Path

import pytest

from lumenrl.models.deepseek_v41.config import AttnMode, DSV41Config

_CONFIG = Path(__file__).parent / "fixtures" / "config.json"


@pytest.fixture(scope="module")
def released():
    return DSV41Config.from_hf(json.loads(_CONFIG.read_text()))


def _role(spec):
    return (spec.mode, spec.compress_ratio, spec.kv_source, spec.index_source, spec.candidate_source)


def test_released_config_shapes(released):
    assert (released.n_layers, released.dim, released.n_heads) == (40, 5120, 64)
    assert (released.n_routed_experts, released.n_activated_experts, released.route_scale) == (384, 6, 1.5)
    assert (released.window_size, released.index_topk, released.hc_mult) == (128, 512, 4)
    assert released.norm_eps == 1e-20
    assert released.engram_layers == ((1, 0), (14, 1))


@pytest.mark.parametrize(
    "layer,role",
    [
        (0, (AttnMode.WINDOW, 0, None, None, None)),
        (1, (AttnMode.WINDOW, 0, None, None, None)),
        (2, (AttnMode.FULL, 2, 2, 2, None)),
        (3, (AttnMode.REUSE, 2, 2, 2, None)),
        (8, (AttnMode.FULL, 2, 8, 8, None)),
        (19, (AttnMode.REUSE, 2, 14, 14, None)),
        (20, (AttnMode.FULL, 1, 20, 20, None)),
        (21, (AttnMode.REUSE, 1, 20, 20, None)),
        (24, (AttnMode.REINDEX, 1, 20, 24, 20)),
        (25, (AttnMode.REUSE, 1, 20, 24, None)),
        (36, (AttnMode.REINDEX, 1, 20, 36, 20)),
        (39, (AttnMode.REUSE, 1, 20, 36, None)),
    ],
)
def test_released_layer_roles(released, layer, role):
    assert _role(released.layer_specs()[layer]) == role


def test_only_the_decoder_head_layer_is_the_candidate_source(released):
    assert [s.layer for s in released.layer_specs() if s.is_candidate_source] == [20]


def test_encoder_slice_keeps_the_full_and_reuse_roles(released):
    cfg = released.slice([0, 1, 2, 3])

    specs = cfg.layer_specs()
    assert [s.mode for s in specs] == [AttnMode.WINDOW, AttnMode.WINDOW, AttnMode.FULL, AttnMode.REUSE]
    assert specs[3].kv_source == 2 and specs[3].index_source == 2
    assert cfg.engram_layers == ((1, 0),)
    assert cfg.engram_layout_layer_ids == (1, 14)
    assert cfg.candidate_source_layer == -1


def test_encoder_decoder_slice_renumbers_every_reference(released):
    cfg = released.slice([0, 1, 2, 3, 20, 21, 24, 25])

    roles = [_role(s) for s in cfg.layer_specs()]
    assert roles[4] == (AttnMode.FULL, 1, 4, 4, None)
    assert roles[5] == (AttnMode.REUSE, 1, 4, 4, None)
    assert roles[6] == (AttnMode.REINDEX, 1, 4, 6, 4)
    assert roles[7] == (AttnMode.REUSE, 1, 4, 6, None)
    assert cfg.candidate_source_layer == 4
    assert cfg.original_layer_ids == (0, 1, 2, 3, 20, 21, 24, 25)


@pytest.mark.parametrize("keep", [[0, 1, 3], [0, 1, 2, 21], [0, 1, 2, 3, 20, 25]])
def test_slice_refuses_to_drop_a_source_a_kept_layer_reads(released, keep):
    with pytest.raises(ValueError, match="drops"):
        released.slice(keep)


def test_a_layer_sharing_a_source_of_another_ratio_is_rejected(released):
    bad = dict(json.loads(_CONFIG.read_text())["text_config"])
    bad["compress_ratios"] = list(bad["compress_ratios"])
    bad["compress_ratios"][3] = 1

    with pytest.raises(ValueError, match="shares the KV source"):
        DSV41Config.from_hf(bad)
