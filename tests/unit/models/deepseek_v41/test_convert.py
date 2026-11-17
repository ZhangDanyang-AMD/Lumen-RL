import json

import pytest
import torch
from safetensors.torch import save_file

from lumenrl.models.deepseek_v41 import ops
from lumenrl.models.deepseek_v41.config import DSV41Config
from lumenrl.models.deepseek_v41.convert import (
    checkpoint_name,
    dequant_fp4,
    dequant_fp8_block,
    load_released,
    write_slice_checkpoint,
)
from lumenrl.models.deepseek_v41.model import DSV41Model
from tests.unit.models.deepseek_v41.test_model_vs_reference import (
    TINY,
    _config,
    _token_map,
)

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


def _halves(names):
    half = len(names) // 2
    return {"model-00001.safetensors": names[:half], "model-00002.safetensors": names[half:]}


def _released_checkpoint(tmp_path, model, shards=_halves):
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
    files = shards(sorted(tensors))
    for file, part in files.items():
        save_file({n: tensors[n].contiguous() for n in part}, str(tmp_path / file))
    index = {"weight_map": {n: f for f, part in files.items() for n in part}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return expect


ROWS = (600,)


def _random_model():
    torch.manual_seed(0)
    model = DSV41Model(_config(ROWS), token_map=torch.tensor(_token_map()))
    with torch.no_grad():
        for t in model.state_dict().values():
            if t.dtype == torch.float8_e4m3fn:
                t.copy_(torch.randn(t.shape).to(torch.float8_e4m3fn))
            elif t.dtype == torch.float8_e8m0fnu:
                t.copy_(_e8m0(torch.randint(-4, 2, t.shape)))
            else:
                t.copy_(torch.randn(t.shape))
    return model


@pytest.fixture
def source(tmp_path):
    model = _random_model()
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


def _hf_config():
    """The tiny config as the release's config.json spells it, with one draft layer."""
    t = TINY
    text = {
        "vocab_size": t["vocab_size"], "hidden_size": t["dim"], "moe_intermediate_size": t["moe_inter_dim"],
        "num_hidden_layers": t["n_layers"], "num_attention_heads": t["n_heads"],
        "n_routed_experts": t["n_routed_experts"], "n_shared_experts": t["n_shared_experts"],
        "num_experts_per_tok": t["n_activated_experts"], "scoring_func": t["score_func"],
        "routed_scaling_factor": t["route_scale"], "norm_topk_prob": True, "swiglu_limit": t["swiglu_limit"],
        "q_lora_rank": t["q_lora_rank"], "head_dim": t["head_dim"], "qk_rope_head_dim": t["rope_head_dim"],
        "rms_norm_eps": t["norm_eps"], "o_groups": t["o_groups"], "o_lora_rank": t["o_lora_rank"],
        "sliding_window": t["window_size"], "compress_ratios": [*t["compress_ratios"], 0],
        "kv_source_layer_ids": list(t["kv_source_layers"]), "index_source_layer_ids": list(t["index_source_layers"]),
        "candidate_source_layer_id": t["candidate_source_layer"], "candidate_topk_blocks": t["candidate_topk_blocks"],
        "candidate_block_size": t["candidate_block_size"], "rope_theta": t["rope_theta"],
        "compress_rope_theta": t["compress_rope_theta"],
        "rope_scaling": {
            "rope_type": "yarn", "factor": t["rope_factor"], "beta_fast": t["beta_fast"],
            "beta_slow": t["beta_slow"], "original_max_position_embeddings": t["original_seq_len"],
        },
        "index_n_heads": t["index_n_heads"], "index_head_dim": t["index_head_dim"], "index_topk": t["index_topk"],
        "hc_mult": t["hc_mult"], "hc_sinkhorn_iters": t["hc_sinkhorn_iters"], "hc_eps": t["hc_eps"],
        "engram_layer_ids": list(t["engram_layer_ids"]), "engram_num_embeddings": list(ROWS),
        "engram_max_ngram_size": t["engram_max_ngram_size"], "engram_vocab_size": t["engram_vocab_size"],
        "engram_n_heads": t["engram_n_heads"], "engram_head_dim": t["engram_head_dim"],
        "engram_pad_token_id": t["engram_pad_id"],
        "engram_compressed_vocab_size": t["engram_compressed_vocab_size"],
        "num_nextn_predict_layers": 1, "dspark_target_layer_ids": [7],
    }
    return {"model_type": "deepseek_v41", "text_config": text, "vision_config": {}}


def _four_shards(names):
    """Layer 1's Engram on its own (as the release has it), the top-level tensors, the other
    layers, the draft layer."""
    groups = {"a-engram": [], "b-top": [], "c-layers": [], "d-mtp": []}
    for n in names:
        key = (
            "a-engram" if n.startswith("layers.1.engram.") else
            "d-mtp" if n.startswith("mtp.") else
            "c-layers" if n.startswith("layers.") else "b-top"
        )
        groups[key].append(n)
    return {f"{k}.safetensors": v for k, v in groups.items()}


@pytest.fixture
def released(tmp_path):
    model = _random_model()
    src = tmp_path / "release"
    src.mkdir()
    _released_checkpoint(src, model, shards=_four_shards)
    (src / "config.json").write_text(json.dumps(_hf_config()))
    (src / "tokenizer.json").write_text("{}")
    return model, src


def test_the_hf_config_helper_is_the_tiny_config():
    assert DSV41Config.from_hf(_hf_config()) == _config(ROWS)


def test_a_sliced_checkpoint_loads_as_the_slice_of_the_release(released, tmp_path):
    model, src = released
    keep = [0, 1, 2, 4, 5, 6, 7]

    report = write_slice_checkpoint(str(src), str(tmp_path / "slice"), keep)

    dst = tmp_path / "slice"
    text = json.loads((dst / "config.json").read_text())["text_config"]
    assert text["num_hidden_layers"] == 7 and text["compress_ratios"] == [0, 0, 2, 1, 1, 1, 1]
    assert text["kv_source_layer_ids"] == [2, 3] and text["index_source_layer_ids"] == [2, 3, 5]
    assert text["candidate_source_layer_id"] == 3 and text["engram_layer_ids"] == [1]
    assert text["num_nextn_predict_layers"] == 0 and text["dspark_target_layer_ids"] == []
    assert (dst / "a-engram.safetensors").is_symlink() and (dst / "b-top.safetensors").is_symlink()
    assert not (dst / "c-layers.safetensors").is_symlink() and not (dst / "d-mtp.safetensors").exists()
    assert (dst / "tokenizer.json").is_symlink()
    assert report["linked_shards"] == 2 and report["written_shards"] == 1

    token_map = torch.tensor(_token_map())
    sliced = DSV41Model(DSV41Config.from_hf(json.loads((dst / "config.json").read_text())), token_map=token_map)
    load_released(sliced, str(dst))
    want = DSV41Model(model.cfg.slice(keep), token_map=token_map)
    load_released(want, str(src))
    assert sliced.state_dict().keys() == want.state_dict().keys()
    for name, t in want.state_dict().items():
        assert torch.equal(_bits(sliced.state_dict()[name]), _bits(t)), name


def test_a_slice_that_moves_an_engram_layer_is_refused(released, tmp_path):
    _, src = released
    with pytest.raises(ValueError, match="moves Engram layers"):
        write_slice_checkpoint(str(src), str(tmp_path / "slice"), [1, 2, 3, 4, 5, 6, 7])


def test_a_slice_that_drops_a_source_is_refused(released, tmp_path):
    _, src = released
    with pytest.raises(ValueError, match="which the slice drops"):
        write_slice_checkpoint(str(src), str(tmp_path / "slice"), [0, 1, 3, 4])
