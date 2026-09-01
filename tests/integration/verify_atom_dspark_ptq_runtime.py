"""ROCm integration check for ATOM-compatible DSpark PTPC checkpoint tensors."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from aiter import QuantType
from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz
from atom.quantization.quark.utils import quant_weight_online
from safetensors import safe_open
from safetensors.torch import save_file

from lumenrl.quantization.atom_dspark_ptq import quantize_ptpc_weight


def main() -> None:
    """Verify canonical quantization, safetensors I/O, and ROCm normalization."""

    weight = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-4.0, -1.0, 2.0, 3.0],
            [0.25, -0.5, 0.75, -1.0],
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    actual_weight, actual_scale = quantize_ptpc_weight(weight)
    reference_weight, reference_scale = quant_weight_online(
        weight,
        QuantType.per_Token,
        torch.float8_e4m3fn,
    )

    assert actual_weight.dtype == torch.float8_e4m3fn
    assert actual_scale.dtype == torch.float32
    assert actual_scale.shape == (weight.shape[0], 1)
    assert torch.isfinite(actual_scale).all()
    assert actual_scale[0].item() == 0
    assert (actual_scale >= 0).all()
    assert torch.equal(actual_weight.view(torch.uint8), reference_weight.view(torch.uint8))
    torch.testing.assert_close(actual_scale, reference_scale, rtol=0, atol=0)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "model.safetensors"
        save_file(
            {
                "linear.weight": actual_weight.cpu(),
                "linear.weight_scale": actual_scale.cpu(),
            },
            checkpoint,
        )
        with safe_open(checkpoint, framework="pt", device="cuda") as loaded:
            loaded_weight = loaded.get_tensor("linear.weight")
            loaded_scale = loaded.get_tensor("linear.weight_scale")

    assert torch.equal(loaded_weight.view(torch.uint8), actual_weight.view(torch.uint8))
    torch.testing.assert_close(loaded_scale, actual_scale, rtol=0, atol=0)

    before = loaded_weight.float() * loaded_scale
    fnuz_weight, fnuz_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
        loaded_weight.clone(),
        loaded_scale.clone(),
    )
    after = fnuz_weight.float() * fnuz_scale
    assert fnuz_weight.dtype == torch.float8_e4m3fnuz
    torch.testing.assert_close(after, before, rtol=0, atol=0)


if __name__ == "__main__":
    main()

