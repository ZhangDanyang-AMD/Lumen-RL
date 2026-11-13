"""Produce a transformers reference forward for a DSv4 slice (optional stage).

transformers 5.13.1 reads the native checkpoint directly (native->HF mapping plus
FP8 dequant), so this is an independent reference: it shares no code with miles,
with Megatron, or with the torch_dist conversion. Saves logits for a fixed token
sequence. Nothing in the DAPO run reads the output; it exists for debugging a
conversion against an independent implementation.

    python3 ref_hf_logits.py --model <WORK_DIR>/models/dsv4-L4-fp8 --out <file.pt> --seq 256
"""

import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, FineGrainedFP8Config
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M


def patched_grouped(self, x):
    # The batched form of this GEMM reads out of bounds on gfx950 (strided bmm);
    # must be installed before from_pretrained.
    input_shape, hidden_dim = x.shape[:-2], x.shape[-1]
    w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2).contiguous()
    x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1).contiguous()
    return torch.bmm(x, w).transpose(0, 1).reshape(*input_shape, self.n_groups, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='native FP8 checkpoint directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    M.DeepseekV4GroupedLinear.forward = patched_grouped

    cfg = AutoConfig.from_pretrained(args.model)
    qc = {
        k: v
        for k, v in (getattr(cfg, 'quantization_config', {}) or {}).items()
        if k not in ('quant_method', 'fmt')
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype='auto',
        device_map='auto',
        quantization_config=FineGrainedFP8Config(dequantize=True, **qc),
    )
    model.eval()
    dev = model.get_input_embeddings().weight.device
    print(f'loaded {args.model} on {dev}', flush=True)

    # Same generator/sequence the Megatron side uses.
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    tokens = torch.randint(0, cfg.vocab_size, (1, args.seq), generator=g)

    with torch.no_grad():
        logits = model(input_ids=tokens.to(dev), use_cache=False).logits

    torch.save(
        {
            'tokens': tokens.cpu(),
            'logits': logits[0].float().cpu(),
            'model': args.model,
            'seed': args.seed,
        },
        args.out,
    )
    print(f'saved {args.out}: logits {tuple(logits.shape)} '
          f'mean={logits.float().mean():.4f} std={logits.float().std():.4f}')


if __name__ == '__main__':
    main()
