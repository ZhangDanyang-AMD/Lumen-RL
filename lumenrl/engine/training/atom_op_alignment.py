"""Opt-in ATOM-shaped Qwen3 dense TP1 operators, without replacing Parameters.

Install before FSDP wrapping, after the packed-attention backend is registered.
Only instance forwards change. Packing, checkpoint names, and HF __call__ hooks
remain intact. No BF16 source/quantized operand is cached across forwards.
The adapter supports first-order training; decode KV-cache execution is rejected.
RoPE uses the public non-cache AITER op (not ATOM's cache-writing fusion).
Norm/SiLU inference forwards have explicit torch-recomputed training backwards.
The model-local core carries (MLP branch, residual) through layer __call__ and
fuses next-layer/final add+norm. FSDP must shard whole decoder layers, not the
individual projections/norms whose __call__ fusion bypasses. Output-capture,
cache, and unknown extra forward arguments are explicitly rejected.
"""

import importlib
import inspect
import logging
import os
from types import MethodType, SimpleNamespace

import torch
from torch.autograd.function import once_differentiable

logger = logging.getLogger(__name__)


class AtomOpAlignmentError(RuntimeError):
    """Requested alignment cannot be supplied; never silently use HF instead."""


def _probe(module, names):
    try:
        imported = importlib.import_module(module)
        result = tuple(getattr(imported, name) for name in names)
        if not all(callable(item) for item in result):
            raise TypeError("non-callable public API")
        return result
    except (ImportError, AttributeError, TypeError) as exc:
        raise AtomOpAlignmentError(f"Missing alignment capability {module}:{names}") from exc


def _probe_capabilities(mxfp4, gemm_callback):
    norm, norm_add, silu = _probe("aiter", ("rmsnorm2d_fwd", "rmsnorm2d_fwd_with_add", "silu_and_mul"))
    rope_fwd, rope_bwd = _probe("aiter.ops.triton.rope.rope", ("rope_cached_thd_positions_2c_fwd", "rope_cached_thd_positions_2c_bwd"))
    if gemm_callback is None:
        try:
            gemm_callback = importlib.import_module("aiter.tuned_gemm").tgemm.mm
        except (ImportError, AttributeError) as exc:
            raise AtomOpAlignmentError("Missing ATOM BF16 tgemm.mm capability") from exc
    if not callable(gemm_callback):
        raise AtomOpAlignmentError("gemm_callback must accept (input_MK, weight_NK)")
    caps = SimpleNamespace(norm=norm, norm_add=norm_add, silu=silu,
                           rope_fwd=rope_fwd, rope_bwd=rope_bwd, gemm=gemm_callback)
    if mxfp4:
        caps.qkv, caps.gate_up, caps.linear = _probe(
            "lumen.ops.quantize.linear", ("mxfp4_qkv_linear", "mxfp4_gate_up_linear", "quantized_linear"))
        (caps.quantize,) = _probe("lumen.ops.quantize.ops", ("convert_to_mxfp4_2d",))
        # Existing Lumen dispatch must be pinned by the caller; otherwise its
        # fallback chain could hide a different quantized forward implementation.
        required = {"LUMEN_MXFP4_ASM": "0", "LUMEN_MXFP4_PRESHUFFLE": "0"}
        for name, value in required.items():
            if os.environ.get(name) != value:
                raise AtomOpAlignmentError(f"Alignment requires {name}={value}")
        static_dispatch = os.environ.get("LUMEN_MXFP4_STATIC_DISPATCH", "0") == "1"
        for name in ("LUMEN_MXFP4_AUTOTUNE", "LUMEN_MXFP4_DGRAD_HADAMARD"):
            if os.environ.get(name, "0") != "0":
                raise AtomOpAlignmentError(f"Alignment requires {name}=0")
        if not static_dispatch and os.environ.get("LUMEN_MXFP4_FLYDSL", "0") != "0":
            raise AtomOpAlignmentError(
                "Alignment requires LUMEN_MXFP4_FLYDSL=0 unless "
                "LUMEN_MXFP4_STATIC_DISPATCH=1"
            )
        linear_module = importlib.import_module("lumen.ops.quantize.linear")
        # The frozen Lumen fast branch calls the chosen direct op without its
        # dequant-BF16 exception fallback. Refuse incompatible imports/versions.
        plain_dispatch_ok = (
            getattr(linear_module, "_FAST_QUANT_DISPATCH", False)
            and getattr(linear_module, "_MXFP4_ASM_ENV", None) == "0"
            and getattr(linear_module, "_MXFP4_PRESHUFFLE_ENV", None) == "0"
            and not getattr(linear_module, "_MXFP4_DGRAD_HADAMARD", True)
            and not linear_module.mxfp4_autotune.AUTOTUNE_ENABLED
            and linear_module._mxfp4_probe_backends()
            and linear_module._fast_mxfp4_gemm_fn is not None
        )
        if static_dispatch:
            static_module = getattr(linear_module, "mxfp4_static_dispatch", None)
            static_state = static_module.state() if static_module is not None else {}
            if not (
                plain_dispatch_ok
                and static_module is not None
                and static_module.ENABLED
                and static_state.get("usable")
            ):
                raise AtomOpAlignmentError(
                    "Static MXFP4 dispatch requested but its table/dispatch is "
                    f"not usable: {static_state.get('reason', 'missing module')}"
                )
        elif not plain_dispatch_ok or linear_module.flydsl_mxfp4._ENABLED:
            raise AtomOpAlignmentError(
                "Frozen Lumen direct plain fail-closed dispatch is unavailable"
            )
    return caps


def _require_packed_attention():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    interface = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
    if not callable(interface) or getattr(interface, "__name__", "") != "_varlen_attention_forward":
        raise AtomOpAlignmentError("Existing AITER packed sdpa wrapper is not registered")


def _norm_reference(x, weight, eps):
    z = x.float()
    z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
    return z.to(x.dtype) * weight


def _silu_reference(x):
    gate, up = x.float().chunk(2, dim=-1)
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


class _Recomputed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, forward, reference, *inputs):
        ctx.reference = reference
        ctx.save_for_backward(*inputs)
        return forward(*inputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        needs = ctx.needs_input_grad[2:]
        with torch.enable_grad():
            inputs = [x.detach().requires_grad_(need) for x, need in zip(ctx.saved_tensors, needs)]
            outputs = ctx.reference(*inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            differentiable = [x for x, need in zip(inputs, needs) if need]
            if not differentiable:
                return (None, None, *([None] * len(inputs)))
            active = [(out, grad) for out, grad in zip(outputs, grad_outputs) if out.requires_grad]
            grads = iter(torch.autograd.grad([out for out, _ in active], differentiable,
                                            [grad for _, grad in active]))
        return (None, None, *(next(grads) if need else None for need in needs))


class _BF16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, gemm):
        ctx.save_for_backward(x, weight)
        y = gemm(x.reshape(-1, x.shape[-1]).contiguous(), weight.contiguous())
        expected = (x.numel() // x.shape[-1], weight.shape[0])
        if not isinstance(y, torch.Tensor) or tuple(y.shape) != expected or y.dtype != x.dtype:
            raise AtomOpAlignmentError("BF16 GEMM callback violated output shape/dtype contract")
        return y.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1])
        # Keep the original torch BF16 backward GEMMs; only forward selects OPUS.
        # Upcasting entire operands would force a slower FP32 compute path.
        dx = (dy2 @ weight).to(x.dtype).view_as(x) if ctx.needs_input_grad[0] else None
        dw = (dy2.t() @ x.reshape(-1, x.shape[-1])).to(weight.dtype) if ctx.needs_input_grad[1] else None
        return dx, dw, None


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, caps):
        # Decoder FSDP input casting must not round ATOM's fp32 frequencies.
        if cos.dtype != torch.float32 or sin.dtype != torch.float32:
            raise AtomOpAlignmentError("ATOM RoPE requires FP32 cos/sin; disable decoder FSDP input casting")
        if cos.requires_grad or sin.requires_grad:
            raise AtomOpAlignmentError("Trainable rotary frequencies are unsupported")
        positions = torch.arange(q.shape[0], device=q.device, dtype=torch.long)
        ctx.save_for_backward(cos, sin, positions)
        ctx.caps = caps
        return caps.rope_fwd(q, k, cos, sin, positions, 0, True, False)

    @staticmethod
    @once_differentiable
    def backward(ctx, dq, dk):
        cos, sin, positions = ctx.saved_tensors
        gq, gk = ctx.caps.rope_bwd(dq.contiguous(), dk.contiguous(), cos, sin, positions, 0, True, False)
        return gq, gk, None, None, None


def _norm(module, x, residual=None):
    caps = module._atom_ops
    eps = module.variance_epsilon
    shape = x.shape
    if residual is None:
        def forward(a, w):
            return caps.norm(a.reshape(-1, shape[-1]).contiguous(), w, eps,
                             use_model_sensitive_rmsnorm=1).view(shape)
        return _Recomputed.apply(forward, lambda a, w: _norm_reference(a, w, eps), x, module.weight)

    def forward(a, r, w):
        flat = a.reshape(-1, shape[-1]).contiguous()
        out, residual_out = torch.empty_like(flat), torch.empty_like(flat)
        caps.norm_add(out, flat, r.reshape(-1, shape[-1]).contiguous(), residual_out,
                      w, eps, use_model_sensitive_rmsnorm=1)
        return out.view(shape), residual_out.view(shape)

    def reference(a, r, w):
        added = a + r
        return _norm_reference(added, w, eps), added

    return _Recomputed.apply(forward, reference, x, residual, module.weight)


def _norm_forward(self, hidden_states):
    return _norm(self, hidden_states)


def _project(owner, x, names):
    modules = [getattr(owner, name) for name in names]
    # Resolve current attributes inside forward, after FSDP's pre-forward
    # unshard/mixed-precision hooks. Never capture Parameter objects at install.
    weights = [module.weight for module in modules]
    if x.dtype != torch.bfloat16 or any(w.dtype != x.dtype or w.device != x.device for w in weights):
        raise AtomOpAlignmentError("Alignment requires unsharded BF16 operands on one device")
    if any(w.ndim != 2 or w.shape[1] != x.shape[-1] for w in weights):
        raise AtomOpAlignmentError("Invalid unsharded linear shape")
    caps = owner._atom_ops
    if owner._atom_mxfp4:
        if any(w.shape[0] % 32 or w.shape[1] % 32 for w in weights):
            raise AtomOpAlignmentError("MXFP4 source rows/columns must be divisible by 32")
        with torch.no_grad():
            packed, scale = caps.quantize(torch.cat(weights, dim=0), block_size=32,
                                          use_sr=False, shuffle_data=False)
        if len(weights) == 3:
            y = caps.qkv(x, *weights, packed, scale, block_size=32,
                         tensor_id="qkv.weight")
        elif len(weights) == 2:
            y = caps.gate_up(x, *weights, packed, scale, block_size=32,
                             tensor_id="gate_up.weight")
        else:
            y = caps.linear(x, weights[0], scaling_type="mxfp4", block_size=32,
                            fp8_weight_cache=packed, fp8_weight_scale=scale,
                            tensor_id=f"{names[0]}.weight")
    else:
        y = _BF16Linear.apply(x, torch.cat(weights, dim=0), caps.gemm)
    biases = [module.bias for module in modules]
    if any(b is not None for b in biases):
        bias = torch.cat([b if b is not None else w.new_zeros(w.shape[0]) for b, w in zip(biases, weights)])
        y = y + bias
    return y


def _linear_forward(self, x):
    # _project uses an owner/member list; keep the original module as its own
    # owner without registering a recursive nn.Module reference.
    xweight = self.weight
    if (x.dtype != torch.bfloat16 or xweight.dtype != x.dtype or x.device != xweight.device
            or xweight.ndim != 2 or xweight.shape[1] != x.shape[-1]):
        raise AtomOpAlignmentError("Expected unsharded BF16 linear operands on one device")
    if self._atom_mxfp4:
        caps = self._atom_ops
        if x.dtype != torch.bfloat16 or xweight.dtype != x.dtype:
            raise AtomOpAlignmentError("Alignment requires BF16 linear operands")
        with torch.no_grad():
            packed, scale = caps.quantize(xweight, block_size=32, use_sr=False, shuffle_data=False)
        y = caps.linear(
            x, xweight, scaling_type="mxfp4", block_size=32,
            fp8_weight_cache=packed, fp8_weight_scale=scale,
            tensor_id=getattr(self, "_atom_tensor_id", "weight"),
        )
    else:
        y = _BF16Linear.apply(x, xweight, self._atom_ops.gemm)
    return y if self.bias is None else y + self.bias


def _attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    if past_key_values is not None:
        raise AtomOpAlignmentError("Alignment supports training/prefill, not cache-writing decode")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    qkv = _project(self, hidden_states, ("q_proj", "k_proj", "v_proj"))
    sizes = [self.q_proj.weight.shape[0], self.k_proj.weight.shape[0], self.v_proj.weight.shape[0]]
    q, k, v = qkv.split(sizes, dim=-1)
    shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    q, k, v = self.q_norm(q.view(shape)), self.k_norm(k.view(shape)), v.view(shape)
    batch, seq = hidden_states.shape[:2]
    cos, sin = position_embeddings
    if cos.shape not in ((batch, seq, self.head_dim), (1, seq, self.head_dim)) or sin.shape != cos.shape:
        raise AtomOpAlignmentError("Expected full Qwen3 rotary frequencies with batch or broadcast batch")
    cos, sin = cos.expand(batch, -1, -1), sin.expand(batch, -1, -1)
    # Frequencies have already been gathered by each batch's position_ids.
    # A per-token table + identity indices preserves packed resets/batch offsets.
    cos = cos[..., : self.head_dim // 2].reshape(batch * seq, self.head_dim // 2).contiguous()
    sin = sin[..., : self.head_dim // 2].reshape_as(cos).contiguous()
    q, k = _RoPE.apply(q.reshape(batch * seq, -1, self.head_dim),
                       k.reshape(batch * seq, -1, self.head_dim), cos, sin, self._atom_ops)
    q, k = q.view(shape), k.view(shape)
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, None)
    if not callable(interface):
        raise AtomOpAlignmentError("Registered packed attention backend is unavailable")
    out, weights = interface(self, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                             attention_mask, dropout=0.0 if not self.training else self.attention_dropout,
                             scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
    return self.o_proj(out.reshape(*hidden_states.shape[:-1], -1).contiguous()), weights


def _mlp_forward(self, x):
    packed = _project(self, x, ("gate_proj", "up_proj"))

    def forward(a):
        out = a.new_empty((*a.shape[:-1], a.shape[-1] // 2))
        self._atom_ops.silu(out, a.contiguous())
        return out

    return self.down_proj(_Recomputed.apply(forward, _silu_reference, packed))


@torch.no_grad()
def _rotary_forward(self, x, position_ids):
    # Preserve fp32 cache values before any BF16 rounding, as ATOM does.
    inv = self.inv_freq.float().to(x.device)[None, :, None]
    with torch.autocast(device_type=x.device.type, enabled=False):
        freqs = (inv.expand(position_ids.shape[0], -1, 1) @ position_ids[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling


def _decoder_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                     use_cache=False, position_embeddings=None, *, residual=None, **kwargs):
    if use_cache or past_key_values is not None:
        raise AtomOpAlignmentError("KV-cache decode is outside the training adapter")
    if kwargs:
        raise AtomOpAlignmentError(f"Unsupported decoder forward arguments: {sorted(kwargs)}")
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = _norm(self.input_layernorm, hidden_states, residual)
    hidden_states, _ = self.self_attn(hidden_states, position_embeddings=position_embeddings,
                                      attention_mask=attention_mask)
    hidden_states, residual = _norm(self.post_attention_layernorm, hidden_states, residual)
    return self.mlp(hidden_states), residual


def _model_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                   inputs_embeds=None, use_cache=None, **kwargs):
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast

    # Replacing core forward deliberately bypasses HF's capture_outputs
    # decorator: tuple layer outputs are an internal carry, not hidden_states.
    for name in ("output_hidden_states", "output_attentions"):
        requested = kwargs.pop(name, getattr(self.config, name, False))
        if requested:
            raise AtomOpAlignmentError(f"{name} capture is unsupported by the residual-carry adapter")
    if kwargs.pop("return_dict", True) is not True:
        raise AtomOpAlignmentError("Core forward returns BaseModelOutputWithPast, not tuple mode")
    if kwargs:
        raise AtomOpAlignmentError(f"Unsupported model forward arguments: {sorted(kwargs)}")
    if use_cache is None:
        use_cache = getattr(self.config, "use_cache", False)
    if use_cache or past_key_values is not None:
        raise AtomOpAlignmentError("KV-cache decode is outside the training adapter")
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if inputs_embeds.ndim != 3:
        raise AtomOpAlignmentError("Expected inputs_embeds with shape [batch, sequence, hidden]")
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
    if position_ids.ndim != 2 or position_ids.shape[1] != inputs_embeds.shape[1]:
        raise AtomOpAlignmentError("Expected position_ids [batch or 1, sequence]")
    if isinstance(attention_mask, dict):
        causal_mask_mapping = attention_mask
        if set(causal_mask_mapping) != {"full_attention"}:
            raise AtomOpAlignmentError("Expected only the full_attention causal-mask mapping")
    else:
        from lumenrl.engine.training.packing import get_packing_context

        if attention_mask is None and get_packing_context() is not None:
            # Packed varlen forward (training forward AND the packed old_log_prob
            # forward): sequence isolation comes from cu_seqlens and the shared
            # kernel's causal=True. A dense HF causal mask is neither needed nor
            # accepted by the shared alignment attention, which is fail-closed on
            # masked/non-packed paths. Keep the mapped slot None so the packed
            # backend still receives its (unmasked, causal) contract.
            causal_mask_mapping = {"full_attention": None}
        else:
            causal_mask_mapping = {"full_attention": create_causal_mask(
                config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                past_key_values=None, position_ids=position_ids)}
    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
    hidden_states, residual = inputs_embeds, None
    for decoder_layer in self.layers:
        if getattr(decoder_layer, "gradient_checkpointing", False) and decoder_layer.training:
            checkpoint = getattr(decoder_layer, "_gradient_checkpointing_func", None)
            if getattr(checkpoint, "keywords", {}).get("use_reentrant") is not False:
                raise AtomOpAlignmentError("Residual carry requires explicit non-reentrant checkpointing")
        # Do not call .forward directly: FSDP pre/post hooks and the existing
        # non-reentrant GradientCheckpointingLayer __call__ own this boundary.
        hidden_states, residual = decoder_layer(
            hidden_states, residual=residual, attention_mask=causal_mask_mapping["full_attention"],
            position_embeddings=position_embeddings, position_ids=position_ids, use_cache=False)
    hidden_states, _ = _norm(self.norm, hidden_states, residual)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)


def install_atom_op_alignment(model, mxfp4: bool, gemm_callback=None):
    """Install fail-closed Qwen3 dense TP1 alignment; return a removable handle.

    ``gemm_callback(input_MK, weight_NK) -> output_MN`` is forward-only;
    the adapter retains native torch first-order BF16 linear gradients.
    MXFP4 linears instead retain Lumen's existing MXFP4 autograd recipe.
    Call once before FSDP, not after sharding. No GPU work occurs at install.
    """
    if type(mxfp4) is not bool:
        raise AtomOpAlignmentError("mxfp4 must be boolean")
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "qwen3":
        raise AtomOpAlignmentError("Only Qwen3 dense is supported")
    if getattr(config, "pretraining_tp", 1) != 1 or getattr(config, "hidden_act", None) != "silu":
        raise AtomOpAlignmentError("Only dense TP1 SiLU is supported")
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise AtomOpAlignmentError("Install the existing packed AITER sdpa backend first")
    if hasattr(model, "_atom_alignment_handle"):
        raise AtomOpAlignmentError("Alignment is already installed")
    modules = list(model.modules())
    layers = [m for m in modules if type(m).__name__ == "Qwen3DecoderLayer"]
    if not layers or len(layers) != config.num_hidden_layers:
        raise AtomOpAlignmentError("Expected complete unwrapped Qwen3 decoder layers")
    actions = []
    for layer in layers:
        attention, mlp = layer.self_attn, layer.mlp
        if type(attention).__name__ != "Qwen3Attention" or type(mlp).__name__ != "Qwen3MLP":
            raise AtomOpAlignmentError("Unsupported Qwen3 layer structure")
        linears = (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj,
                   mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        if any(type(linear) is not torch.nn.Linear or hasattr(linear, "parametrizations") for linear in linears):
            raise AtomOpAlignmentError("Expected original, unparametrized HF linear modules")
        if attention.sliding_window is not None or attention.attention_dropout != 0:
            raise AtomOpAlignmentError("Only zero-dropout full causal attention is supported")
        signature = inspect.signature(attention.forward)
        if list(signature.parameters)[:4] != ["hidden_states", "position_embeddings", "attention_mask", "past_key_values"]:
            raise AtomOpAlignmentError(f"Unsupported Qwen3 attention signature: {signature}")
        for norm in (layer.input_layernorm, layer.post_attention_layernorm, attention.q_norm, attention.k_norm):
            if type(norm).__name__ != "Qwen3RMSNorm":
                raise AtomOpAlignmentError("Expected original Qwen3RMSNorm")
            actions.append((norm, _norm_forward, False))
        actions.extend(((attention, _attention_forward, mxfp4), (mlp, _mlp_forward, mxfp4),
                        (attention.o_proj, _linear_forward, mxfp4), (mlp.down_proj, _linear_forward, mxfp4),
                        (layer, _decoder_forward, False)))
    core = getattr(model, "model", model)
    if type(core).__name__ != "Qwen3Model" or not hasattr(core, "norm") or not hasattr(model, "lm_head"):
        raise AtomOpAlignmentError("Expected Qwen3 causal LM with original core, final norm and lm_head")
    core_signature = inspect.signature(core.forward)
    if list(core_signature.parameters)[:6] != ["input_ids", "attention_mask", "position_ids", "past_key_values", "inputs_embeds", "use_cache"]:
        raise AtomOpAlignmentError(f"Unsupported Qwen3 model forward signature: {core_signature}")
    if getattr(core, "has_sliding_layers", False) or any(t != "full_attention" for t in config.layer_types):
        raise AtomOpAlignmentError("Only full causal Qwen3 layers are supported")
    rotary = getattr(core, "rotary_emb", None)
    rope_parameters = getattr(config, "rope_parameters", {})
    if rotary is None or type(rotary).__name__ != "Qwen3RotaryEmbedding" or rope_parameters.get("rope_type") != "default":
        raise AtomOpAlignmentError("Only default Qwen3 RoPE is supported")
    actions.extend(((core.norm, _norm_forward, False), (model.lm_head, _linear_forward, False),
                    (rotary, _rotary_forward, False), (core, _model_forward, False)))
    for module, _, _ in actions:
        if "_atom_ops" in module.__dict__ or "_atom_mxfp4" in module.__dict__:
            raise AtomOpAlignmentError("Module already has adapter state")
    for parameter in model.parameters():
        if hasattr(parameter, "placements"):
            raise AtomOpAlignmentError("Install alignment before FSDP/DTensor sharding")
    _require_packed_attention()
    caps = _probe_capabilities(mxfp4, gemm_callback)
    saved = [(module, module.__dict__.get("forward")) for module, _, _ in actions]
    tensor_ids = []
    if mxfp4:
        for layer in layers:
            for module, name in (
                (layer.self_attn.o_proj, "o_proj.weight"),
                (layer.mlp.down_proj, "down_proj.weight"),
            ):
                tensor_ids.append((module, module.__dict__.get("_atom_tensor_id")))
                module._atom_tensor_id = name
    for module, forward, quantized in actions:
        module._atom_ops, module._atom_mxfp4 = caps, quantized
        module.forward = MethodType(forward, module)
    handle = SimpleNamespace(capabilities={"norm": "AITER model-sensitive; cross-layer/final add+norm; torch recompute bwd",
                                          "silu": "AITER + fp32 torch recompute bwd",
                                          "rope": "AITER public cached fwd/bwd, no KV write",
                                          "attention": "existing sdpa registry/packing untouched",
                                          "linear": "Lumen MXFP4" if mxfp4 else "BF16 callback"})

    def remove():
        for module, previous in tensor_ids:
            if previous is None:
                module.__dict__.pop("_atom_tensor_id", None)
            else:
                module._atom_tensor_id = previous
        for module, previous in saved:
            if previous is None:
                del module.forward
            else:
                module.forward = previous
            del module._atom_ops
            del module._atom_mxfp4
        del model._atom_alignment_handle

    handle.remove = remove
    model._atom_alignment_handle = handle
    logger.info("ATOM operator alignment installed: %s", handle.capabilities)
    return handle
