#!/usr/bin/env python3
"""Triton kernel for the 3x3 Winograd F4x3 convolution on gfx950."""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _conv_3x3_winograd_f4x3_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    k_height: tl.constexpr,
    k_width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    input_batch_stride: tl.constexpr,
    input_channel_stride: tl.constexpr,
    input_row_stride: tl.constexpr,
    input_col_stride: tl.constexpr,
    weight_kh_stride: tl.constexpr,
    weight_kw_stride: tl.constexpr,
    weight_in_ch_stride: tl.constexpr,
    weight_out_ch_stride: tl.constexpr,
    bias_out_ch_stride: tl.constexpr,
    output_batch_stride: tl.constexpr,
    output_channel_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    output_col_stride: tl.constexpr,
):
    """Compute the 3x3 Winograd F4x3 convolution."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Compute output coordinates
    oh = pid_m * 4
    ow = pid_n * 3
    
    # Loop over channel tiles
    for c_tile in range(tl.cdiv(channels, 16)):
        c_start = c_tile * 16
        
        # Load weight tile (16 channels x 9 weights)
        w_vals = tl.zeros((16, 9), dtype=tl.float32)
        for kh in range(k_height):
            for kw in range(k_width):
                idx = (
                    (kh * weight_kh_stride +
                     kw * weight_kw_stride +
                     c_start * weight_in_ch_stride) &
                    (batch_size * channels * k_height * k_width - 1)
                )
                w_vals += tl.load(weight_ptr + idx, mask=(c_start + tl.arange(0, 16)) < channels)
        
        # Apply Winograd transform to weight
        w_transformed = tl.zeros((16, 9), dtype=tl.float32)
        for i in range(16):
            for j in range(9):
                acc = 0.0
                for m in range(4):
                    for n in range(4):
                        acc += w_vals[i, m * 3 + n] * tl.math.pow(-1.0, m + n)
                w_transformed[i, j] = acc
        
        # Accumulate over 4x4 input patches
        acc = tl.zeros((16, 9), dtype=tl.float32)
        for ih in range(4):
            for iw in range(4):
                h_idx = oh + ih
                w_idx = ow + iw
                
                # Check bounds
                valid = ((h_idx >= padding_h) & 
                         (w_idx >= padding_w) & 
                         (h_idx < height + padding_h) & 
                         (w_idx < width + padding_w))
                
                # Handle padding
                pad_h = h_idx - padding_h
                pad_w = w_idx - padding_w
                
                # Load input values
                val = tl.where(valid, 
                              tl.load(input_ptr + 
                                    (pad_h * input_row_stride + 
                                     pad_w * input_col_stride + 
                                     c_start * input_channel_stride)), 
                              0.0)
                
                # Apply Winograd transform to input patch
                for i in range(16):
                    for j in range(9):
                        acc[i, j] += val * tl.math.pow(-1.0, ih + iw)
        
        # Multiply transformed weight with transformed input
        result = tl.zeros((16, 9), dtype=tl.float32)
        for i in range(16):
            for j in range(9):
                result[i, j] = acc[i, j] * w_transformed[i, j]
        
        # Store results
        for i in range(16):
            if c_start + i < channels:
                for j in range(9):
                    tl.store(output_ptr + 
                           ((oh + (j // 3)) * output_row_stride + 
                            (ow + (j % 3)) * output_col_stride + 
                            (c_start + i) * output_channel_stride),
                           result[i, j])


def conv_3x3_winograd_f4x3(input_tensor, weight_tensor, bias=None, stride=(1, 1), padding=(0, 0)):
    """
    Compute the 3x3 Winograd F4x3 convolution.
    
    Args:
        input_tensor: Input tensor of shape (B, H, W, C) in NHWC layout
        weight_tensor: Weight tensor of shape (3, 3, C, O)
        bias: Bias tensor of shape (O,) or None
        stride: Stride tuple (stride_h, stride_w)
        padding: Padding tuple (padding_h, padding_w)
    
    Returns:
        Output tensor of shape (B, OH, OW, O) where OH = ceil((H + 2*padding_h - 3)/stride_h + 1)
              and OW = ceil((W + 2*padding_w - 3)/stride_w + 1)
    """
    if input_tensor.shape != (4, 512, 512, 256):
        raise ValueError("Input tensor must be of shape (4, 512, 512, 256)")
    if weight_tensor.shape != (3, 3, 256, 64):
        raise ValueError("Weight tensor must be of shape (3, 3, 256, 64)")
    if bias is not None and bias.shape != (64,):
        raise ValueError("Bias tensor must be of shape (64,)")

    # Convert to BF16 if needed
    if input_tensor.dtype != torch.bfloat16:
        input_tensor = input_tensor.to(torch.bfloat16)
    if weight_tensor.dtype != torch.bfloat16:
        weight_tensor = weight_tensor.to(torch.bfloat16)
    if bias is not None and bias.dtype != torch.bfloat16:
        bias = bias.to(torch.bfloat16)

    # Get dimensions
    B, H, W, C = input_tensor.shape
    KH, KW, _, O = weight_tensor.shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    
    # Calculate output size
    OH = math.ceil((H + 2 * padding_h - KH) / stride_h + 1)
    OW = math.ceil((W + 2 * padding_w - KW) / stride_w + 1)
    
    # Create output tensor
    output = torch.empty(B, OH, OW, O, device=input_tensor.device, dtype=torch.bfloat16)
    
    # Launch kernel
    grid = (math.ceil(OH / 4), math.ceil(OW / 3))
    
    _conv_3x3_winograd_f4x3_kernel[
        grid,
        num_warps=4,
        num_stages=3,
    ](
        input_tensor,
        weight_tensor,
        bias,
        output,
        batch_size=B,
        channels=C,
        height=H,
        width=W,
        k_height=KH,
        k_width=KW,
        out_height=OH,
        out_width=OW,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        input_batch_stride=H * W * C,
        input_channel_stride=1,
        input_row_stride=W * C,
        input_col_stride=C,
        weight_kh_stride=KW * C * O,
        weight_kw_stride=C * O,
        weight_in_ch_stride=O,
        weight_out_ch_stride=1,
        bias_out_ch_stride=1,
        output_batch_stride=OH * OW * O,
        output_channel_stride=1,
        output_row_stride=OW * O,
        output_col_stride=O,
    )
    
    return output