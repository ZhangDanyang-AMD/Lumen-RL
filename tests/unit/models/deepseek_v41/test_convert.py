import json

import pytest
import torch
from safetensors.torch import save_file

from lumenrl.models.deepseek_v41 import ops
from lumenrl.models.deepseek_v41.convert import (
    checkpoint_name,
    dequant_fp4,
    dequant_fp8_block,
    load_released,
)
from lumenrl.models.deepseek_v41.model import DSV41Model
from tests.unit.models.deepseek_v41.test_model_vs_reference import _config, _token_map

FP8_STORED = ("attn.w", "shared_experts", "indexer.wq_b", "engram.wkv")

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _e8m0(exponents: torch.Tensor) -> torch.Tensor:
    return (exponents + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


def test_fp8_block_dequant_is_exact_and_handles_ragged_blocks():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(40, 70, generator=g) * 50).to(torch.float8_e4m3fn)
    exps = torch.randint(-8, 3, (2, 3), generator=g)

    got = dequant_fp8_block(w, _e8m0(exps))

    want = w.float() * (2.0 ** exps.float()).repeat_interleave(32, 0)[:40].repeat_interleave(32, 1)[:, :70]
    assert got.dtype == torch.bfloat16 and torch.equal(got.float(), want)


def test_fp4_dequant_reads_the_low_nibble_first():
    codes = torch.tensor([[3, 12] * 16])  # 1.5, -2.0, ...
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)

    got = dequant_fp4(packed, _e8m0(torch.tensor([[1]])))

    assert got[0, :4].tolist() == [3.0, -4.0, 3.0, -4.0]


def _quant_fp8(w):
    out, inn = w.shape
    blocks = w.float().unflatten(0, (out // 32, 32)).unflatten(-1, (inn // 32, 32)).transpose(1, 2)
    scale = ops.pow2_ceil(blocks.abs().amax(dim=(-1, -2)).clamp_min(1e-4) / 448)
    q = (blocks / scale[..., None, None]).to(torch.float8_e4m3fn).transpose(1, 2).reshape(out, inn)
    e = torch.log2(scale).round().long()
    return q, _e8m0(e)


def _quant_fp4(w):
    g = w.float().unflatten(-1, (-1, 32))
    scale = ops.pow2_ceil(g.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-126) / 6)
    vals = ops.round_e2m1((g / scale).clamp(-6, 6)).flatten(-2)
    codes = torch.tensor([E2M1.index(v) if v != 0 else 0 for v in vals.flatten().tolist()]).view(vals.shape)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
    return packed, _e8m0(torch.log2(scale.squeeze(-1)).round().long())


def _bits(t):
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _released_checkpoint(tmp_path, model):
    """Write `model`'s weights the way the release stores them; return what loading must give."""
    tensors, expect = {}, {}
    for name, t in model.state_dict().items():
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            tensors[name] = expect[name] = t.clone()
        elif ".experts." in name and ".shared_experts." not in name:
            q, s = _quant_fp4(t)
            tensors[name], tensors[name[: -len("weight")] + "scale"] = q, s
            expect[name] = dequant_fp4(q, s)
        elif name.endswith(".weight") and t.dtype == torch.bfloat16 and any(k in name for k in FP8_STORED):
            q, s = _quant_fp8(t)
            tensors[name], tensors[name[: -len("weight")] + "scale"] = q, s
            expect[name] = dequant_fp8_block(q, s)
        elif t.dtype == torch.float32 and ("hc_" in name or "attn_sink" in name or "gate.bias" in name):
            tensors[name] = expect[name] = t.clone()
        else:
            tensors[name] = t.to(torch.bfloat16)
            expect[name] = tensors[name].to(t.dtype)
    tensors["mtp.0.attn_norm.weight"] = torch.ones(4, dtype=torch.bfloat16)
    names = sorted(tensors)
    half = len(names) // 2
    files = {"model-00001.safetensors": names[:half], "model-00002.safetensors": names[half:]}
    for file, part in files.items():
        save_file({n: tensors[n].contiguous() for n in part}, str(tmp_path / file))
    index = {"weight_map": {n: f for f, part in files.items() for n in part}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return expect


@pytest.fixture
def source(tmp_path):
    torch.manual_seed(0)
    rows = (600,)
    model = DSV41Model(_config(rows), token_map=torch.tensor(_token_map()))
    with torch.no_grad():
        for t in model.state_dict().values():
            if t.dtype == torch.float8_e4m3fn:
                t.copy_(torch.randn(t.shape).to(torch.float8_e4m3fn))
            elif t.dtype == torch.float8_e8m0fnu:
                t.copy_(_e8m0(torch.randint(-4, 2, t.shape)))
            else:
                t.copy_(torch.randn(t.shape))
    return model, _released_checkpoint(tmp_path, model), tmp_path


def test_load_released_fills_every_tensor_with_the_dequantized_value(source):
    model, expect, path = source
    fresh = DSV41Model(model.cfg, token_map=torch.tensor(_token_map()))

    report = load_released(fresh, str(path))

    assert report["loaded"] == len(expect) and report["skipped_kinds"] == ["mtp"]
    for name, t in fresh.state_dict().items():
        assert t.dtype == expect[name].dtype, name
        assert torch.equal(_bits(t), _bits(expect[name])), name


def test_a_slice_reads_its_layers_from_their_released_ids(source):
    model, expect, path = source
    cfg = model.cfg.slice([0, 1, 2, 3, 4, 5])
    fresh = DSV41Model(cfg, token_map=torch.tensor(_token_map()))

    load_released(fresh, str(path))

    assert checkpoint_name("layers.4.attn.wkv.weight", cfg.original_layer_ids) == "layers.4.attn.wkv.weight"
    assert torch.equal(fresh.state_dict()["layers.3.attn.wq_a.weight"], expect["layers.3.attn.wq_a.weight"])
    assert torch.equal(fresh.state_dict()["layers.5.ffn.gate.weight"], expect["layers.5.ffn.gate.weight"])


def test_checkpoint_name_maps_slice_positions_to_released_ids():
    assert checkpoint_name("layers.4.attn.wkv.weight", (0, 1, 2, 3, 20, 21)) == "layers.20.attn.wkv.weight"
    assert checkpoint_name("head.weight", (0, 1)) == "head.weight"


def test_a_tensor_without_a_source_is_an_error(source, tmp_path):
    model, _, path = source
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"]["layers.0.attn.attn_sink"]
    index_path.write_text(json.dumps(index))

    with pytest.raises(KeyError, match="no checkpoint source"):
        load_released(DSV41Model(model.cfg, token_map=torch.tensor(_token_map())), str(path))
