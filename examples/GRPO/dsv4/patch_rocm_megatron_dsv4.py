#!/usr/bin/env python3
"""Add DSV4 support to ROCm/Megatron-LM (rocm_dev branch).

Adds the minimal TransformerConfig fields and TransformerBlock/Layer hooks
needed by Lumen's ``get_dsv4_spec()`` to construct the DSV4 model.  The actual
DSV4 attention/compressor/indexer/HC layers come from the Lumen library — this
patch only makes Megatron's config and block infrastructure accept them.

Changes:
  1. TransformerConfig: add 'dsv4' to experimental_attention_variant Literal,
     add dsv4_* fields (hc_mult, compress_ratios, window_size, etc.),
     add dsv4_mode post_init flag.
  2. TransformerBlock: add HCHeadParams on last PP rank when dsv4_mode.
  3. TransformerLayer: add per-layer HC params (hc_attn_fn/base/scale,
     hc_ffn_fn/base/scale) when dsv4_mode.

Usage (inside container):
    python3 examples/GRPO/dsv4/patch_rocm_megatron_dsv4.py /workspace/Megatron-LM
"""

from __future__ import annotations

import os
import re
import sys


def patch_file(path: str, replacements: list[tuple[str, str]]) -> bool:
    with open(path) as f:
        content = f.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_transformer_config(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_config.py")

    with open(path) as f:
        content = f.read()
    original = content

    # 1. Add 'dsv4' to the Literal type
    content = content.replace(
        "Literal['gated_delta_net', 'dsa']",
        "Literal['gated_delta_net', 'dsa', 'dsv4']",
    )

    # 2. Add DSV4 fields after the DSA section
    # Find the right place — after the last DSA field
    dsa_marker = "    ####################\n    # DSA\n    ####################"
    if dsa_marker in content:
        # Find end of DSA fields section (next ## or blank-line-then-field)
        # Insert DSV4 fields after all dsa_ fields
        # Find last dsa_ field
        last_dsa_match = None
        for m in re.finditer(r'    dsa_\w+:.*\n(?:    """.*?"""\n)?', content):
            last_dsa_match = m
        if last_dsa_match:
            insert_pos = last_dsa_match.end()
            dsv4_fields = '''
    ####################
    # DSV4
    ####################
    dsv4_mode: bool = False
    dsv4_hc_mult: Optional[int] = None
    dsv4_hc_sinkhorn_iters: int = 20
    dsv4_hc_eps: float = 1e-6
    dsv4_compress_ratios: Optional[list] = None
    dsv4_compress_rope_theta: float = 160000.0
    dsv4_o_groups: Optional[int] = None
    dsv4_o_lora_rank: Optional[int] = None
    dsv4_n_hash_layers: int = 0
    dsv4_window_size: int = 128

'''
            content = content[:insert_pos] + dsv4_fields + content[insert_pos:]

    # 3. Add dsv4_mode = True in __post_init__ when variant == "dsv4"
    dsa_post_init = '        if self.experimental_attention_variant == "dsa":'
    if dsa_post_init in content:
        dsv4_post_init = '''        if self.experimental_attention_variant == "dsv4":
            self.dsv4_mode = True

'''
        # Insert before the dsa check
        content = content.replace(dsa_post_init, dsv4_post_init + dsa_post_init)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_transformer_block(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_block.py")

    with open(path) as f:
        content = f.read()
    original = content

    # Add HC head params creation after _build_layers() call
    build_layers_call = "        self._build_layers()"
    if build_layers_call in content and "HCHeadParams" not in content:
        hc_block = '''
        # DSV4 Hyper-Connection head params (last PP rank only)
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.models.dsv4.ops.hyper_connection import HCHeadParams
            from megatron.core import parallel_state as mpu
            if mpu.is_pipeline_last_stage():
                self.hc_head_params = HCHeadParams(self.config)
'''
        content = content.replace(
            build_layers_call,
            build_layers_call + hc_block,
        )

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_transformer_layer(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_layer.py")

    with open(path) as f:
        content = f.read()
    original = content

    # Add per-layer HC params after self.mlp assignment in __init__
    # Find a stable anchor point — after the mlp is built
    if "dsv4_mode" not in content and "self.mlp = build_module" in content:
        # Find the class __init__ and add HC params
        # Look for the end of __init__ where self.mlp is assigned
        anchor = "        self.bias_dropout_add_exec_handler = torch.enable_grad"
        if anchor in content:
            hc_layer = '''
        # DSV4 Hyper-Connection per-layer params
        if getattr(self.config, 'dsv4_mode', False):
            import torch.nn as nn
            hc_mult = self.config.dsv4_hc_mult or 4
            hc_dim = hc_mult * self.config.hidden_size
            mix_size = (2 + hc_mult) * hc_mult
            self.hc_attn_fn = nn.Parameter(torch.zeros(mix_size, hc_dim, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.zeros(mix_size, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_size, hc_dim, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.zeros(mix_size, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            for p in [self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale,
                       self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale]:
                p._keep_fp32 = True

'''
            content = content.replace(anchor, hc_layer + anchor)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_eav_specs(megatron_root: str) -> bool:
    """Add dsv4 branch to get_experimental_attention_variant_module_spec."""
    path = os.path.join(megatron_root, "megatron", "core", "models", "gpt",
                        "experimental_attention_variant_module_specs.py")
    with open(path) as f:
        content = f.read()
    original = content

    # Replace the else branch to handle dsv4 before raising.
    # Lumen's get_dsv4_spec monkey-patches this at runtime, but the Literal
    # type needs to accept 'dsv4' without erroring.
    old_else = '''    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )'''
    new_else = '''    elif config.experimental_attention_variant == "dsv4":
        # DSV4 spec is injected by Lumen's get_dsv4_spec() monkey-patch at runtime
        raise ValueError(
            "DSV4 attention variant requires Lumen's get_dsv4_spec() — "
            "call it before get_experimental_attention_variant_module_spec()"
        )
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )'''
    if old_else in content and "dsv4" not in content:
        content = content.replace(old_else, new_else)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_tp_layers(megatron_root: str) -> bool:
    """Add condition_init_method to tensor_parallel/layers.py (needed by Lumen linears)."""
    path = os.path.join(megatron_root, "megatron", "core", "tensor_parallel", "layers.py")
    with open(path) as f:
        content = f.read()
    if "def condition_init_method" in content:
        return False
    stub = '''

def condition_init_method(config, init_method):
    """Condition weight initialization on config (Lumen compatibility shim).

    Returns the init_method unchanged — Lumen's LumenColumnParallelLinear calls
    this during CPU initialization. Xavier-uniform override is not used for DSV4.
    """
    if getattr(config, "init_method_xavier_uniform", False):
        import torch.nn.init as init
        return init.xavier_uniform_
    return init_method

'''
    content += stub
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_distrib_optimizer_fp32_detach(megatron_root: str) -> bool:
    """Fix missing .detach() on FP32 param shard views in DistributedOptimizer.

    The BF16 path uses model_param.detach().view(-1)[...] but the FP32 path
    uses model_param.view(-1)[...] without .detach(). When FP32 params have
    requires_grad=True (e.g. DSV4 HC params, attn_sink, compressor APE), the
    view creates a non-leaf tensor, causing 'can't optimize a non-leaf Tensor'
    when HybridDeviceOptimizer (CPU offload) validates param groups.
    """
    path = os.path.join(megatron_root, "megatron", "core", "optimizer", "distrib_optimizer.py")
    with open(path) as f:
        content = f.read()
    old = "shard_model_param = model_param.view(-1)["
    if old not in content:
        return False
    content = content.replace(old, "shard_model_param = model_param.detach().view(-1)[")
    with open(path, "w") as f:
        f.write(content)
    return True


def main(megatron_root: str) -> None:
    results = {
        "transformer_config.py": patch_transformer_config(megatron_root),
        "transformer_block.py": patch_transformer_block(megatron_root),
        "transformer_layer.py": patch_transformer_layer(megatron_root),
        "experimental_attention_variant_module_specs.py": patch_eav_specs(megatron_root),
        "tensor_parallel/layers.py": patch_tp_layers(megatron_root),
        "optimizer/distrib_optimizer.py": patch_distrib_optimizer_fp32_detach(megatron_root),
    }
    print(f"Patched ROCm Megatron at {megatron_root}:")
    for name, ok in results.items():
        print(f"  {'PATCHED' if ok else 'skipped'}: {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <megatron-root>")
        sys.exit(1)
    main(sys.argv[1])
