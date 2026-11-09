import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_N": 4096}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _dynamic_quant_kernel(
    X_ptr, OUT_ptr, SCALE_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    amax = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))

    max_val = tl.max(amax)
    scale = max_val / 127.0
    scale = tl.where(scale > 0, scale, 1.0)
    tl.store(SCALE_ptr + row, scale)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        q = x / scale
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(OUT_ptr + row * stride_om + cols * stride_on, q.to(tl.bfloat16), mask=mask)


def dynamic_quantize(x):
    M, N = x.shape
    output = torch.empty_like(x)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    _dynamic_quant_kernel[(M,)](
        x, output, scale,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
    )
    return output
