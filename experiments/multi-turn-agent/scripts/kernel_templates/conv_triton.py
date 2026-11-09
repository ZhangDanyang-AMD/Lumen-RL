import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 4, "BLOCK_C": 64, "BLOCK_L": 128}, num_warps=4),
        triton.Config({"BLOCK_B": 4, "BLOCK_C": 128, "BLOCK_L": 64}, num_warps=4),
        triton.Config({"BLOCK_B": 2, "BLOCK_C": 128, "BLOCK_L": 128}, num_warps=8),
    ],
    key=["C", "L"],
)
@triton.jit
def _causal_conv1d_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    B_size, C, L, K,
    stride_xb, stride_xc, stride_xl,
    stride_ob, stride_oc, stride_ol,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    c_mask = offs_c < C
    l_mask = offs_l < L

    acc = tl.zeros([BLOCK_C, BLOCK_L], dtype=tl.float32)

    for ki in range(K):
        src_l = offs_l - (K - 1) + ki
        src_mask = (src_l >= 0) & (src_l < L) & l_mask
        x_ptrs = X_ptr + pid_b * stride_xb + offs_c[:, None] * stride_xc + src_l[None, :] * stride_xl
        x = tl.load(x_ptrs, mask=c_mask[:, None] & src_mask[None, :], other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs_c * K + ki, mask=c_mask, other=0.0).to(tl.float32)
        acc += x * w[:, None]

    if HAS_BIAS:
        bias = tl.load(BIAS_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        acc += bias[:, None]

    o_ptrs = OUT_ptr + pid_b * stride_ob + offs_c[:, None] * stride_oc + offs_l[None, :] * stride_ol
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=c_mask[:, None] & l_mask[None, :])


def causal_conv1d(x, weight, bias=None):
    B_size, C, L = x.shape
    C_w, _, K = weight.shape
    assert C == C_w
    output = torch.empty_like(x)
    grid = lambda meta: (B_size, triton.cdiv(C, meta["BLOCK_C"]), triton.cdiv(L, meta["BLOCK_L"]))
    _causal_conv1d_kernel[grid](
        x, weight.reshape(C, K), bias, output,
        B_size, C, L, K,
        x.stride(0), x.stride(1), x.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        HAS_BIAS=bias is not None,
    )
    return output
