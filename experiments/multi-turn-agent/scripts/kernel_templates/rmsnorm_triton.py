import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_N": 8192}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, OUT_ptr,
    N: tl.constexpr,
    eps,
    stride_x_row, stride_out_row,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_off = row * stride_x_row
    o_off = row * stride_out_row

    _sum_sq = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + x_off + cols, mask=mask, other=0.0).to(tl.float32)
        _sum_sq += x * x

    mean_sq = tl.sum(_sum_sq) / N
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + x_off + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = x * rrms * w
        tl.store(OUT_ptr + o_off + cols, out.to(tl.bfloat16), mask=mask)


def fused_rmsnorm_add(x, weight, eps=1e-5):
    M, N = x.shape
    output = torch.empty_like(x)
    _rmsnorm_kernel[(M,)](
        x, weight, output,
        N, eps,
        x.stride(0), output.stride(0),
    )
    return output
