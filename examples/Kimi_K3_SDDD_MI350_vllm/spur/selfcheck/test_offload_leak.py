"""Measure how much GPU memory survives the Phase A/B offload sequence.

Round 1 of batch-alternating training dies because vLLM finds only 166/288 GiB
free after _offload_draft_to_cpu(). This reproduces the offload on one GPU and
reports allocated / reserved / driver-free separately, which tells us whether
the memory is still owned by tensors or merely held by the caching allocator.
"""

import gc
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lumenrl")

from lumenrl.models.dspark import DSparkModel

HIDDEN = 7168
VOCAB = 163840
NUM_TARGET_LAYERS = 5
B = int(os.environ.get("TEST_B", 1))
T = int(os.environ.get("TEST_T", 2048))
ANCHOR_NUM = int(os.environ.get("TEST_ANCHOR", 512))

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
GIB = 2**30


def report(tag: str) -> None:
    free, total = torch.cuda.mem_get_info()
    print(
        f"{tag:<28} allocated={torch.cuda.memory_allocated() / GIB:7.2f} "
        f"reserved={torch.cuda.memory_reserved() / GIB:7.2f} "
        f"driver_free={free / GIB:7.2f} / {total / GIB:.2f} GiB"
    )


def main() -> int:
    print(f"config: B={B} T={T} anchor_num={ANCHOR_NUM}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
    torch.manual_seed(0)
    report("baseline")

    model = DSparkModel(
        hidden_dim=HIDDEN,
        vocab_size=VOCAB,
        num_heads=64,
        num_kv_heads=64,
        num_layers=5,
        num_target_layers=NUM_TARGET_LAYERS,
        block_size=7,
        ffn_dim=14336,
        rms_norm_eps=1e-5,
        rope_theta=50000.0,
        rope_scaling={
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        anchor_num=ANCHOR_NUM,
        markov_rank=256,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        mask_token_id=163837,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ).to(device=DEVICE, dtype=DTYPE)
    report("model on GPU")

    lm_head_w = torch.randn(VOCAB, HIDDEN, device=DEVICE, dtype=DTYPE) * 0.02
    embed_w = torch.randn(VOCAB, HIDDEN, device=DEVICE, dtype=DTYPE) * 0.02

    # Stand-in for BF16Optimizer: fp32 master + fp32 grads + Adam m/v.
    fp32_params = [p.detach().float().clone() for p in model.parameters()]
    fp32_grads = [torch.zeros_like(p) for p in fp32_params]
    adam_state = [
        (torch.zeros_like(p), torch.zeros_like(p)) for p in fp32_params
    ]
    report("optimizer states on GPU")

    input_ids = torch.randint(0, VOCAB, (B, T), device=DEVICE)
    aux_hidden = torch.randn(B, T, HIDDEN * NUM_TARGET_LAYERS, device=DEVICE, dtype=DTYPE)
    target_hs = torch.randn(B, T, HIDDEN, device=DEVICE, dtype=DTYPE)
    loss_mask = torch.zeros(B, T, device=DEVICE)
    loss_mask[:, T // 2:] = 1.0

    result = model(
        input_ids=input_ids,
        token_embeds=F.embedding(input_ids, embed_w),
        aux_hidden_states=aux_hidden,
        teacher_lm_head_weight=lm_head_w,
        embed_weight=embed_w,
        loss_mask=loss_mask,
        target_hidden_states=target_hs,
        ce_alpha=0.1,
        l1_alpha=0.9,
        conf_alpha=1.0,
        decay_gamma=4.0,
    )
    result["total_loss"].backward()
    del result, input_ids, aux_hidden, target_hs, loss_mask
    print(f"peak during step: {torch.cuda.max_memory_allocated() / GIB:.2f} GiB")
    report("after backward")

    # ---- what _offload_draft_to_cpu() does ----
    for p in model.parameters():
        p.grad = None
    for p in model.parameters(recurse=True):
        p.data = p.data.to("cpu")
    for b in model.buffers(recurse=True):
        b.data = b.data.to("cpu")
    for mp in fp32_params:
        mp.data = mp.data.cpu()
    for mg in fp32_grads:
        mg.data = mg.data.cpu()
    for m, v in adam_state:
        m.data = m.data.cpu()
        v.data = v.data.cpu()
    del lm_head_w, embed_w
    report("after moving to CPU")

    torch.cuda.empty_cache()
    gc.collect()
    report("after empty_cache + gc")

    torch.cuda.empty_cache()
    report("after 2nd empty_cache")

    survivors: dict[str, float] = {}
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor) or not obj.is_cuda:
                continue
        except Exception:
            continue
        key = f"{tuple(obj.shape)} {obj.dtype}"
        survivors[key] = survivors.get(key, 0.0) + obj.numel() * obj.element_size() / GIB
    print("\nCUDA tensors still alive:")
    for key, size in sorted(survivors.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {size:7.3f} GiB  {key}")

    free, total = torch.cuda.mem_get_info()
    want = 0.9 * total
    print(
        f"\nvLLM would need {want / GIB:.2f} GiB free, driver reports "
        f"{free / GIB:.2f} GiB -> {'OK' if free >= want else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
