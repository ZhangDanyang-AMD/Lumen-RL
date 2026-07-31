"""Single-GPU forward/backward check for DSparkModel with the real K3 config.

Mirrors SpecDistillTrainer._train_step_dspark without vLLM, torchrun or the
teacher cache, so shape bugs in the dual-source MLA path surface in seconds
instead of a 6-minute docker cycle.
"""

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
ANCHOR_NUM = int(os.environ.get("TEST_ANCHOR", 64))

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16


def build_model() -> DSparkModel:
    rope_scaling = {
        "type": "yarn",
        "rope_type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 32768,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
    }
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
        rope_scaling=rope_scaling,
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
    )
    return model.to(device=DEVICE, dtype=DTYPE)


def main() -> int:
    torch.manual_seed(0)
    print(f"config: B={B} T={T} anchor_num={ANCHOR_NUM}")
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"draft params: {n_params}")

    lm_head_w = torch.randn(VOCAB, HIDDEN, device=DEVICE, dtype=DTYPE) * 0.02
    embed_w = torch.randn(VOCAB, HIDDEN, device=DEVICE, dtype=DTYPE) * 0.02

    input_ids = torch.randint(0, VOCAB, (B, T), device=DEVICE)
    token_embeds = F.embedding(input_ids, embed_w)

    # separate_last_hidden=true: vLLM emits N-1 layers, the trainer re-joins the
    # last one, so fc sees all N concatenated.
    aux_hidden = torch.randn(
        B, T, HIDDEN * (NUM_TARGET_LAYERS - 1), device=DEVICE, dtype=DTYPE
    )
    target_hs = torch.randn(B, T, HIDDEN, device=DEVICE, dtype=DTYPE)
    aux_hidden = torch.cat([aux_hidden, target_hs], dim=-1)

    # Supervise the second half only, as a real sample's assistant turn would.
    loss_mask = torch.zeros(B, T, device=DEVICE)
    loss_mask[:, T // 2:] = 1.0

    result = model(
        input_ids=input_ids,
        token_embeds=token_embeds,
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

    total = result["total_loss"]
    print(
        f"total_loss={float(total):.4f} ce={float(result['ce_loss']):.4f} "
        f"tv={float(result['tv_loss']):.4f} conf={float(result['conf_loss']):.4f}"
    )
    for i, (ls, acc) in enumerate(zip(result["losses"], result["accuracies"])):
        print(f"  step {i}: loss={float(ls):.4f} acc={float(acc):.4f}")

    total.backward()
    grad_sq = 0.0
    missing = []
    for name, p in model.named_parameters():
        if p.grad is None:
            missing.append(name)
        else:
            grad_sq += float(p.grad.float().pow(2).sum())
    print(f"grad_norm={grad_sq ** 0.5:.4f}")
    if missing:
        print(f"params without grad: {len(missing)} (first: {missing[:5]})")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak GPU alloc: {peak:.1f} GiB")

    if not torch.isfinite(total):
        print("FAIL: total_loss is not finite")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
