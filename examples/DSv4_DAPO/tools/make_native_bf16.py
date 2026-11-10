"""Rewrite the dequantized DSv4 slice under the checkpoint's own tensor names.

``fp8_cast_bf16.py`` dequantizes the released FP8 checkpoint and, on the way,
renames every tensor into DeepSeek-HF form (``model.layers.0.self_attn.wq_a``).
transformers reads that; vLLM's DSv4 loader does not -- it keys on the names the
released checkpoint actually uses (``layers.0.attn.wq_a``), and dies with
``KeyError: 'layers.0.input_layernorm.weight'``.

That matters for RL, not just for loading. Serving the released FP8 checkpoint
leaves vLLM holding FP8 parameters, and the trainer's BF16 weight sync then dies
in vLLM's loader (``'Parameter' object has no attribute
'load_merged_column_weight'``). A BF16 rollout takes the trainer's BF16 weights
as they are, with no quantization on either side.

So: pure rename, no arithmetic. The mapping is the inverse of SGLang's
``remap_weight_name_to_dpsk_hf_format``.

    python3 make_native_bf16.py \
        --src <WORK_DIR>/models/dsv4-L4-bf16 \
        --dst <WORK_DIR>/models/dsv4-L4-bf16-native \
        --reference <WORK_DIR>/models/dsv4-L4-fp8
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def is_mtp(name: str) -> bool:
    """The multi-token-prediction block, which does not belong in this checkpoint.

    ``fp8_cast_bf16`` does not recognise the ``mtp.0.*`` names and leaves that
    block exactly as it found it -- ``mtp.0.e_proj.weight`` is still
    ``float8_e4m3`` next to its ``[32, 32]`` block scale. Copying it into a
    directory whose config.json no longer declares ``quantization_config`` would
    hand vLLM FP8 tensors for BF16 layers.

    Dropping it costs nothing: MTP is a separate model in vLLM
    (``DeepSeekV4MTP``, built only for speculative decoding), the trainer's
    torch_dist has 43 layers and no MTP block, so a resident copy would never be
    weight-synced anyway. The 4-layer slice that this pipeline was validated on
    had no MTP block either.
    """
    return name.startswith('mtp.')


def to_native(name: str) -> str:
    if name == 'model.embed_tokens.weight':
        return 'embed.weight'
    if name == 'lm_head.weight':
        return 'head.weight'
    if name == 'model.norm.weight':
        return 'norm.weight'
    if name.startswith('model.hc_head_'):
        return name[len('model.'):]

    assert name.startswith('model.layers.'), f'unexpected tensor {name!r}'
    n = name[len('model.'):]
    n = n.replace('.self_attn.', '.attn.').replace('.mlp.', '.ffn.')
    n = n.replace('.input_layernorm.', '.attn_norm.')
    n = n.replace('.post_attention_layernorm.', '.ffn_norm.')
    n = n.replace('.ffn.topk.tid2eid', '.ffn.gate.tid2eid')
    n = n.replace('.ffn.gate.e_score_correction_bias', '.ffn.gate.bias')
    n = n.replace('.gate_proj.', '.w1.').replace('.down_proj.', '.w2.')
    n = n.replace('.up_proj.', '.w3.')
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='bf16 HF-named checkpoint')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--reference', default=None,
                    help='the native FP8 checkpoint, to check the names against')
    cli = ap.parse_args()

    os.makedirs(cli.dst, exist_ok=True)
    with open(os.path.join(cli.src, 'model.safetensors.index.json')) as fh:
        index = json.load(fh)

    want = None
    if cli.reference:
        with open(os.path.join(cli.reference, 'model.safetensors.index.json')) as fh:
            # The reference is FP8, so drop its per-weight scales, and the MTP
            # block for the reason in is_mtp().
            want = {k for k in json.load(fh)['weight_map']
                    if not k.endswith('.scale') and not is_mtp(k)}

    new_map: dict[str, str] = {}
    dropped = 0
    shards = sorted(set(index['weight_map'].values()))
    for i, shard in enumerate(shards, 1):
        tensors = load_file(os.path.join(cli.src, shard))
        kept = {k: v for k, v in tensors.items() if not is_mtp(k)}
        dropped += len(tensors) - len(kept)
        renamed = {to_native(k): v for k, v in kept.items()}
        assert len(renamed) == len(kept), 'rename collided'
        if not renamed:
            print(f'  [{i}/{len(shards)}] {shard}: MTP only, skipped', flush=True)
            continue
        save_file(renamed, os.path.join(cli.dst, shard), metadata={'format': 'pt'})
        new_map.update({k: shard for k in renamed})
        print(f'  [{i}/{len(shards)}] {shard}: {len(renamed)} tensors', flush=True)
    if dropped:
        print(f'  dropped {dropped} MTP tensors (see is_mtp)', flush=True)

    with open(os.path.join(cli.dst, 'model.safetensors.index.json'), 'w') as fh:
        json.dump({'metadata': index.get('metadata', {}), 'weight_map': new_map}, fh, indent=1)

    # vLLM auto-detects quantization from config.json. These weights are BF16, so
    # leaving the FP8 block-quant config in place would have it build FP8 layers
    # that the BF16 weight sync cannot write to.
    with open(os.path.join(cli.src, 'config.json')) as fh:
        cfg = json.load(fh)
    cfg.pop('quantization_config', None)
    # The MTP weights are gone, so stop advertising the block. Absent is the
    # shape the validated 4-layer directory had.
    cfg.pop('num_nextn_predict_layers', None)
    cfg['torch_dtype'] = 'bfloat16'
    with open(os.path.join(cli.dst, 'config.json'), 'w') as fh:
        json.dump(cfg, fh, indent=1)
    for extra in ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json'):
        p = os.path.join(cli.src, extra)
        if os.path.exists(p):
            shutil.copy2(p, cli.dst)

    print(f'\nwrote {len(new_map)} tensors to {cli.dst}')
    if want is not None:
        got = set(new_map)
        missing, extra = sorted(want - got), sorted(got - want)
        print(f'vs the native checkpoint: {len(want & got)}/{len(want)} names match')
        for k in missing[:10]:
            print(f'  missing: {k}')
        for k in extra[:10]:
            print(f'  extra:   {k}')
        if missing or extra:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
