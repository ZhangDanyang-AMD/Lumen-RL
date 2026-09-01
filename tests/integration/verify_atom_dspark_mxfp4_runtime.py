"""ROCm integration check for DSpark MXFP4 weights and dynamic A4W4 GEMM."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from aiter import dtypes
from aiter.utility import fp4_utils
from atom.model_ops.linear import gemm_a4w4_quant
from atom.model_ops.utils import shuffle_weights
from safetensors import safe_open
from safetensors.torch import save_file

from lumenrl.quantization.atom_dspark_ptq import quantize_mxfp4_weight


def main() -> None:
    """Verify packed checkpoint I/O and dynamic MXFP4 activation GEMM."""

    torch.manual_seed(7)
    rows, output_size, input_size = 16, 256, 512
    weight = (
        torch.randn(output_size, input_size, dtype=torch.bfloat16, device="cuda")
        * 0.02
    )
    quantized, scale = quantize_mxfp4_weight(weight)

    assert quantized.dtype == dtypes.fp4x2
    assert quantized.shape == (output_size, input_size // 2)
    assert scale.dtype == dtypes.fp8_e8m0
    assert scale.shape == (output_size, input_size // 32)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "model.safetensors"
        save_file(
            {
                "linear.weight": quantized.cpu(),
                "linear.weight_scale": scale.cpu(),
            },
            checkpoint,
        )
        with safe_open(checkpoint, framework="pt", device="cuda") as loaded:
            loaded_weight = loaded.get_tensor("linear.weight")
            loaded_scale = loaded.get_tensor("linear.weight_scale")

    assert torch.equal(loaded_weight.view(torch.uint8), quantized.view(torch.uint8))
    assert torch.equal(loaded_scale.view(torch.uint8), scale.view(torch.uint8))

    # Match LinearBase.process_weights_after_loading for the default AITER path.
    weight_parameter = torch.nn.Parameter(loaded_weight, requires_grad=False)
    shuffle_weights(weight_parameter)
    loaded_weight = weight_parameter.data
    loaded_scale = fp4_utils.e8m0_shuffle(loaded_scale)
    activation = torch.randn(
        rows, input_size, dtype=torch.bfloat16, device="cuda"
    )
    actual = gemm_a4w4_quant(
        activation,
        None,
        loaded_weight,
        torch.bfloat16,
        loaded_scale,
        dtypes.fp4x2,
        None,
        output_size,
    )
    reference = activation.float().matmul(weight.float().t())

    assert actual.shape == (rows, output_size)
    assert torch.isfinite(actual).all()
    relative_mae = (actual.float() - reference).abs().mean() / reference.abs().mean()
    assert relative_mae.item() < 0.25


if __name__ == "__main__":
    main()
