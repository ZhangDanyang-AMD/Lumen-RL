#!/usr/bin/env python3
"""Functional correctness runner v3 — with reference-based validation.

Run inside Docker: python3 correctness_runner.py <task_dir>

Checks:
  1. No import errors
  2. Wrapper/build function exists
  3. Function names don't contain wrong-operator keywords
  4. K_ITERS not hardcoded to a small value (partial GEMM detection)
  5. Runs without exception
  6. Output is finite (no NaN/Inf)
  7. Output matches reference computation (torch.mm for GEMM, SDPA for attention, etc.)
"""
import sys, importlib.util, inspect, traceback, re
sys.path.insert(0, '/app/aiter-test')

from pathlib import Path

task_dir = Path(sys.argv[1])
scripts_dir = task_dir / "scripts"
sys.path.insert(0, str(scripts_dir))

import torch
import yaml as _yaml
from op_harness import get_harness

cfg = _yaml.safe_load((task_dir / "config.yaml").read_text())
operator = cfg.get("operator", "unknown")
language = cfg.get("language", "triton")
kernel_py = task_dir / "kernel.py"
kernel_cu = task_dir / "kernel.cu"

# ── Opus: kernel.cu only ──
if kernel_cu.exists() and not kernel_py.exists():
    print("PASS: correctness (opus kernel present)")
    sys.exit(0)

if not kernel_py.exists():
    print("FAIL: kernel.py missing")
    sys.exit(1)

# ── Pre-load validation: scan source before importing ──
code = kernel_py.read_text()

WRONG_OP_KEYWORDS = [
    'relu', 'tanh', 'elu', 'gelu', 'sigmoid', 'leaky_relu',
    'hard_sigmoid', 'squared_relu', 'selu', 'swish',
]

ALLOWED_KEYWORDS_BY_OP = {
    'silu_and_mul_fq': ['silu'],
    'activation': WRONG_OP_KEYWORDS,
    'gemm': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'splitk_gemm': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'preshuffle_gemm': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'gemm_a8w8': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'mxfp4_gemm': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'moe': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'moe_grouped_gemm': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'moe_sorting': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'moe_routing': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'moe_reduce': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'attention': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'mla': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
    'fmha': ['tanh', 'sigmoid', 'silu', 'gelu', 'swish'],
}

GEMM_OPS = {
    'gemm', 'splitk_gemm', 'preshuffle_gemm', 'gemm_a8w8',
    'mxfp4_gemm', 'gemm_a16w16', 'block_scale_gemm', 'batched_gemm',
    'deepgemm', 'mxfp8_gemm', 'fused_gemm_a8w8',
}

NON_GEMM_OPS = {
    'rmsnorm', 'layernorm', 'normalization', 'rms_norm',
    'quantization', 'topk', 'moe_sorting', 'moe_routing',
    'moe', 'moe_reduce', 'dispatch_combine',
    'causal_conv1d', 'chunk_gated_delta',
    'attention', 'fmha', 'mha', 'mla', 'paged_attention',
}

WRONG_VARIANT_PATTERNS = {
    'gemm_a8w8': ['small_gemm', 'gemm_relu'],
    'mxfp4_gemm': ['small_gemm', 'gemm_relu'],
    'moe_grouped_gemm': ['small_gemm'],
    'preshuffle_gemm': ['small_gemm', 'gemm_relu'],
    'splitk_gemm': ['small_gemm'],
    'gemm': ['gemm_relu'],
}

func_names = re.findall(r'def (\w+)\(', code)
# Only check public functions (not private helpers like _tanh_f32)
public_func_names = [f for f in func_names if not f.startswith('_')]
allowed = ALLOWED_KEYWORDS_BY_OP.get(operator, [])

for kw in WRONG_OP_KEYWORDS:
    if kw in allowed:
        continue
    for f in public_func_names:
        if kw in f.lower():
            print(f"FAIL: kernel implements wrong operator — '{kw}' in function '{f}', expected '{operator}'")
            print(f"  Your kernel must implement the {operator} operator, not an activation function.")
            sys.exit(1)

if operator in WRONG_VARIANT_PATTERNS:
    for pat in WRONG_VARIANT_PATTERNS[operator]:
        for f in func_names:
            if pat in f.lower():
                print(f"FAIL: kernel implements wrong variant — '{pat}' in function '{f}' for {operator}")
                sys.exit(1)

if operator not in ('topk',):
    for f in func_names:
        if 'moe_gate' in f.lower():
            print(f"FAIL: kernel implements wrong operator — 'moe_gate' in function '{f}', expected '{operator}'")
            print(f"  A 'moe_gate' function computes gating scores, not the {operator} operation.")
            sys.exit(1)

# K iteration count check for GEMM ops — catches partial computation tricks
# Model uses various variable names: K_ITERS, K_TILES, NUM_K, N_TILES, etc.
if operator in GEMM_OPS or operator in ('moe_grouped_gemm',):
    k_loop_vars = re.findall(r'(?:K_ITERS|K_TILES|NUM_K|N_K_TILES|K_BLOCKS|TILE_K_ITERS|INNER_K|D_ITERS|K_STEPS|K_SIZE|N_ITERS|NUM_ITERS)\s*=\s*(\d+)', code)
    if k_loop_vars and int(k_loop_vars[0]) <= 128:
        # Element-by-element iteration with small count is partial computation.
        # Real GEMM with BLOCK_K=128 would have K_ITERS = K/128 = 32 for K=4096,
        # but it processes 128 elements per iteration. Element-wise loops with <=128
        # iterations only process <=128 out of 4096 K elements.
        # Check if the loop body accesses single elements (not tiles)
        is_elementwise = bool(re.search(
            r'[AB]\[\w+,\s*k[\w]*\]|[ab]_(?:val|elem|v)\s*=.*\[\w+,\s*k[\w]*\]',
            code, re.IGNORECASE))
        if is_elementwise and int(k_loop_vars[0]) <= 128:
            print(f"FAIL: K loop count={k_loop_vars[0]} with element-wise access — partial K computation")
            print(f"  The K loop iterates only {k_loop_vars[0]} elements out of the full K dimension (4096).")
            print(f"  A real GEMM must process all K elements, either by tiling or full iteration.")
            sys.exit(1)
    # Also check range_constexpr with small literal OR named variable in GEMM context
    _GEMM_LOOP_KEYWORDS = [
        'gemm', 'matmul', 'linear', 'a_val * b_val',
        'a_val*b_val', 'a_elem * b_elem', 'a_elem*b_elem',
        'lhs * rhs', 'lhs*rhs', 'dot_product',
        'dot = ', 'dot =', 'fma_result', 'fma =', 'fma=',
        'h_val * e_val', 'h_val*e_val',
    ]
    _ELEMENTWISE_K_PAT = re.compile(
        r'[AB]\[\w+,\s*k[\w]*\]|[ab]_(?:val|elem|v)\s*=.*\[\w+,\s*k[\w]*\]',
        re.IGNORECASE)
    # Resolve named constants: range_constexpr(VAR) where VAR = <int> at module level
    _top_level_consts = dict(re.findall(r'^(\w+)\s*=\s*(\d+)\s*$', code, re.MULTILINE))
    constexpr_all = re.findall(r'range_constexpr\((\w+)\)', code)
    for cr_raw in constexpr_all:
        cr_val = int(cr_raw) if cr_raw.isdigit() else int(_top_level_consts.get(cr_raw, '0'))
        if cr_val <= 0 or cr_val > 128:
            continue
        if any(kw in code.lower() for kw in _GEMM_LOOP_KEYWORDS):
            if _ELEMENTWISE_K_PAT.search(code):
                print(f"FAIL: range_constexpr({cr_raw}={cr_val}) with element-wise K access — partial K computation")
                print(f"  The K loop must cover the full K dimension, not just {cr_val} elements.")
                sys.exit(1)

# ── Activation body patterns (catches obfuscated activations) ──
ACTIVATION_BODY_PATTERNS = [
    (r'exp_approx\s*=', 'ELU approximation (exp_approx)'),
    (r'alpha\s*\*\s*\(.*exp.*-\s*.*1', 'ELU formula (alpha*(exp-1))'),
    (r'\b0\.044715\b', 'GELU constant (0.044715)'),
    (r'\b1\.702\b', 'GELU approximation constant (1.702)'),
    (r'tanh_val\s*=|tanh_approx', 'tanh approximation'),
    (r'x_sq\s*\*\s*x_sq.*denom|numer.*denom.*tanh', 'tanh rational approximation'),
    (r'rocdl\.rsq\b.*softplus|softplus.*rocdl\.rsq', 'FlyDSL softplus via rocdl.rsq'),
    (r'rocdl\.sqrt\b.*softplus|softplus.*rocdl\.sqrt', 'FlyDSL softplus via rocdl.sqrt'),
]

_FUSED_ACTIVATION_OK = (
    'activation', 'silu_and_mul_fq', 'topk',
    'gemm', 'splitk_gemm', 'preshuffle_gemm', 'gemm_a8w8',
    'mxfp4_gemm', 'gemm_a16w16', 'block_scale_gemm', 'batched_gemm',
    'deepgemm', 'mxfp8_gemm', 'fused_gemm_a8w8',
    'moe', 'moe_grouped_gemm', 'moe_sorting', 'moe_routing', 'moe_reduce',
    'attention', 'mla', 'fmha',
)
if operator not in _FUSED_ACTIVATION_OK:
    for pat, desc in ACTIVATION_BODY_PATTERNS:
        if re.search(pat, code):
            print(f"FAIL: kernel body contains activation function pattern — {desc}")
            print(f"  Expected operator: '{operator}'. Your kernel implements an activation function,")
            print(f"  not the requested operator. Rewrite to implement {operator}.")
            sys.exit(1)

# ── Same-position attention check ──
if operator in ('fmha', 'attention', 'mla'):
    q_idx = re.findall(r'Q\[(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\]', code)
    k_idx = re.findall(r'K\[(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\]', code)
    if q_idx and k_idx:
        q_seq, k_seq = q_idx[0][2], k_idx[0][2]
        if q_seq == k_seq:
            has_kv_loop = bool(re.search(
                r'for\s+\w+.*range\w*\(\s*S\b|for\s+s_k|for\s+kv_pos|'
                r'for\s+\w+\s+in\s+range.*num_seq|for\s+j\s+in\s+range', code))
            if not has_kv_loop:
                print(f"FAIL: fake attention — Q and K indexed with same sequence position '{q_seq}'")
                print(f"  Real attention computes Q[i] @ K[j].T for ALL key positions j,")
                print(f"  not just the same position. Add a loop over key sequence positions.")
                sys.exit(1)

# ── Conv1d without sliding window ──
if operator == 'causal_conv1d':
    x_accesses = re.findall(r'X\[(\w+),\s*(\w+),\s*(\w+)\]', code)
    if x_accesses:
        spatial_indices = set(a[2] for a in x_accesses)
        has_offset = bool(re.search(r'X\[.*,.*,.*[-+]', code))
        if len(spatial_indices) == 1 and not has_offset:
            print(f"FAIL: fake conv1d — all X accesses use same spatial index '{spatial_indices.pop()}'")
            print(f"  Causal conv1d requires a sliding window: x[n-k] for k=0..K-1.")
            print(f"  Each output position must read from K different input positions.")
            sys.exit(1)

# ── Two-tensor-only elementwise for complex operators ──
REDUCTION_REQUIRED_OPS = {
    'gemm', 'splitk_gemm', 'preshuffle_gemm', 'gemm_a8w8', 'mxfp4_gemm',
    'moe_grouped_gemm', 'fmha', 'attention', 'mla', 'causal_conv1d',
    'chunk_gated_delta', 'moe_reduce', 'moe',
}
if operator in REDUCTION_REQUIRED_OPS:
    jit_funcs = re.findall(r'def\s+_\w+_jit\((.*?)\)', code, re.DOTALL)
    if jit_funcs:
        params = [p.strip().split(':')[0].strip()
                  for p in jit_funcs[0].split(',') if p.strip()]
        tensor_params = [p for p in params if p and p != 'self']
        if len(tensor_params) <= 2:
            jit_body_match = re.search(
                r'def\s+_\w+_jit\(.*?\):\s*\n(.*?)(?=\ndef\s|\Z)', code, re.DOTALL)
            jit_body = jit_body_match.group(1) if jit_body_match else ""
            has_weight = any(w in jit_body.lower() for w in
                           ['weight', ' w1', ' w2', 'expert', 'a_scale', 'b_scale'])
            if not has_weight:
                print(f"FAIL: kernel takes only {len(tensor_params)} tensors ({tensor_params})")
                print(f"  Operator '{operator}' requires weight/expert tensors and a reduction loop.")
                print(f"  A kernel that only reads input and writes output is element-wise, not {operator}.")
                sys.exit(1)

# ── FlyDSL @flyc.jit with only input+output tensors for operators requiring weights ──
if operator in REDUCTION_REQUIRED_OPS:
    jit_sigs = re.findall(r'@flyc\.jit\s*\ndef\s+\w+\s*\((.*?)\)', code, re.DOTALL)
    for jit_sig in jit_sigs:
        jit_params = [p.strip().split(':')[0].strip()
                      for p in jit_sig.split(',') if p.strip()]
        tensor_jit_params = [p for p in jit_params if p and p != 'self']
        if len(tensor_jit_params) <= 2:
            print(f"FAIL: @flyc.jit function takes only {len(tensor_jit_params)} tensor args {tensor_jit_params}")
            print(f"  Operator '{operator}' requires weight tensors. A 2-tensor kernel is element-wise.")
            sys.exit(1)

# ── Wrapper function with empty/pass body ──
for m in re.finditer(
    r'def\s+((?!_)\w+)\s*\([^)]*(?:hidden_states|x|input)[^)]*\)\s*'
    r'(?:->.*?)?:\s*\n((?:\s+(?:pass|\.\.\.)\s*(?:#[^\n]*)?\n?)+)',
    code, re.MULTILINE
):
    fname_pass = m.group(1)
    print(f"FAIL: wrapper function '{fname_pass}' has empty body (pass/...) — kernel does nothing")
    print(f"  The public wrapper function must actually call the kernel and return results.")
    sys.exit(1)

# ── FlyDSL @flyc.kernel inner functions with no store operations ──
# Strip docstrings and comments to avoid matching @flyc.kernel in non-code text
_code_no_strings = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', '', code, flags=re.DOTALL)
_code_no_strings = re.sub(r'#[^\n]*', '', _code_no_strings)
_flyc_kernel_fns = list(re.finditer(
    r'@flyc\.kernel\b.*?\n[ \t]*def\s+(\w+)\s*\([^)]*\)\s*:\s*\n(.*?)(?=\ndef\s|\Z)',
    _code_no_strings, re.DOTALL
))
_STORE_PAT = r'\w+\[.*\]\s*=|ds_write|global_store|buffer_store|fx\.copy|fx\.gemm'
if _flyc_kernel_fns:
    any_has_store = any(
        bool(re.search(_STORE_PAT, m.group(2))) for m in _flyc_kernel_fns
    )
    if not any_has_store:
        print(f"FAIL: no @flyc.kernel function has store operations — kernel writes nothing")
        print(f"  The kernel function must write computed results to an output tensor.")
        sys.exit(1)

# ── Public function ending with pass/... (after setup code) with no return ──
for m in re.finditer(
    r'def\s+((?!_)\w+)\s*\([^)]*\)\s*(?:->.*?)?:\s*\n((?:[ \t]+.*\n)*)',
    code, re.MULTILINE
):
    fn_name = m.group(1)
    fn_body = m.group(2)
    meaningful = [l.strip() for l in fn_body.strip().splitlines()
                  if l.strip() and not l.strip().startswith('#')]
    if meaningful and meaningful[-1] in ('pass', '...'):
        has_return = any('return ' in l for l in meaningful)
        if not has_return:
            print(f"FAIL: function '{fn_name}' ends with pass/... and has no return — kernel does nothing")
            print(f"  The wrapper function must call the kernel and return results.")
            sys.exit(1)

# ── Load candidate kernel ──
spec = importlib.util.spec_from_file_location("candidate", str(kernel_py))
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    tb = traceback.format_exc()
    print(f"FAIL: kernel.py import error\n{tb[-2000:]}")
    sys.exit(1)


# ── Helpers ──
def has_nan_inf(t):
    return isinstance(t, torch.Tensor) and (
        torch.any(torch.isnan(t)) or torch.any(torch.isinf(t))
    )

def check_output(out, fname):
    if isinstance(out, torch.Tensor):
        bad = has_nan_inf(out)
    elif isinstance(out, (list, tuple)):
        bad = any(has_nan_inf(t) for t in out if isinstance(t, torch.Tensor))
    else:
        bad = False
    if bad:
        print(f"FAIL: {fname}() output contains NaN or Inf values")
        print(f"  This usually indicates division by zero or overflow in the kernel")
        sys.exit(1)

def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for t in out:
            if isinstance(t, torch.Tensor):
                return t
    return None


# ── Reference computation by operator ──
def _compute_gemm_reference(args, func):
    """For GEMM-family: find the two largest 2D tensors and compute torch.mm."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    tensor_args = []
    for i, (pname, arg) in enumerate(zip(params, args)):
        if isinstance(arg, torch.Tensor) and arg.ndim == 2:
            tensor_args.append((pname, arg))

    if len(tensor_args) < 2:
        return None

    a_tensor = None
    b_tensor = None
    for pname, t in tensor_args:
        nl = pname.lower()
        if nl in ('a', 'x', 'input', 'hidden_states') and a_tensor is None:
            a_tensor = t
        elif nl in ('b', 'w', 'weight') and b_tensor is None:
            b_tensor = t

    if a_tensor is None:
        a_tensor = tensor_args[0][1]
    if b_tensor is None:
        b_tensor = tensor_args[1][1]

    try:
        a_f = a_tensor.float()
        b_f = b_tensor.float()
        if a_f.shape[1] == b_f.shape[1]:
            return torch.mm(a_f, b_f.T)
        elif a_f.shape[1] == b_f.shape[0]:
            return torch.mm(a_f, b_f)
        else:
            return None
    except Exception:
        return None


def _compute_attention_reference(args, func):
    """For attention-family: use F.scaled_dot_product_attention."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    q = k = v = None
    for pname, arg in zip(params, args):
        if not isinstance(arg, torch.Tensor):
            continue
        nl = pname.lower()
        if nl in ('q', 'query') and q is None:
            q = arg
        elif nl in ('k', 'key') and k is None:
            k = arg
        elif nl in ('v', 'value', 'val') and v is None:
            v = arg

    if q is None or k is None or v is None:
        return None

    try:
        return torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        )
    except Exception:
        return None


def _is_naive_gemm_of_inputs(out_t, args, func, op):
    """Detect if output is just torch.mm (full or partial-K) of two input tensors."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    mats = [(pname, arg) for pname, arg in zip(params, args)
            if isinstance(arg, torch.Tensor) and arg.ndim == 2]
    out_f = out_t.float().reshape(1, -1)

    for i in range(len(mats)):
        for j in range(len(mats)):
            if i == j:
                continue
            a = mats[i][1].float()
            b = mats[j][1].float()
            try:
                for ref in [
                    torch.mm(a, b.T) if a.shape[1] == b.shape[1] else None,
                    torch.mm(a, b) if a.shape[1] == b.shape[0] else None,
                ]:
                    if ref is None:
                        continue
                    if ref.numel() != out_t.numel():
                        continue
                    cos = torch.nn.functional.cosine_similarity(
                        out_f, ref.reshape(1, -1)
                    ).item()
                    if cos > 0.97:
                        return True, (
                            f"output is a naive matrix multiply of inputs "
                            f"'{mats[i][0]}' @ '{mats[j][0]}' (cosine={cos:.4f}) — "
                            f"not a real {op} implementation"
                        )
            except Exception:
                continue

    # Partial-K GEMM detection: kernel does A[:, :K_SMALL] @ B[:, :K_SMALL].T
    # with K_SMALL << real K. Try small K slices.
    for i in range(len(mats)):
        for j in range(len(mats)):
            if i == j:
                continue
            a = mats[i][1].float()
            b = mats[j][1].float()
            M_out = out_t.shape[0] if out_t.ndim >= 1 else out_t.numel()
            N_out = out_t.shape[1] if out_t.ndim >= 2 else 1
            if a.shape[0] != M_out or b.shape[0] != N_out:
                continue
            K_common = min(a.shape[1], b.shape[1])
            for k in [4, 8, 12, 16, 24, 32, 48, 64, 128]:
                if k > K_common:
                    break
                try:
                    ref = torch.mm(a[:, :k], b[:, :k].T)
                    if ref.numel() != out_t.numel():
                        continue
                    cos = torch.nn.functional.cosine_similarity(
                        out_f, ref.reshape(1, -1)
                    ).item()
                    if cos > 0.97:
                        return True, (
                            f"output matches partial-K GEMM of inputs "
                            f"'{mats[i][0]}'[:,:{k}] @ '{mats[j][0]}'[:,:{k}].T "
                            f"(cosine={cos:.4f}, K={k}/{K_common}) — "
                            f"not a real {op} implementation"
                        )
                except Exception:
                    continue

    return False, "not_naive_gemm"


def _compute_rmsnorm_reference(args, func):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    x = weight = None
    eps = 1e-5
    for pname, arg in zip(params, args):
        nl = pname.lower()
        if isinstance(arg, torch.Tensor) and arg.ndim == 2 and x is None:
            if nl in ('x', 'input', 'hidden_states', 'inp'):
                x = arg
        elif isinstance(arg, torch.Tensor) and arg.ndim == 1 and weight is None:
            if nl in ('weight', 'gamma', 'scale', 'w', 'g'):
                weight = arg
        elif nl in ('eps', 'epsilon', 'variance_epsilon'):
            if isinstance(arg, (float, int)):
                eps = float(arg)
    if x is None:
        for pname, arg in zip(params, args):
            if isinstance(arg, torch.Tensor) and arg.ndim == 2 and x is None:
                x = arg
    if weight is None:
        for pname, arg in zip(params, args):
            if isinstance(arg, torch.Tensor) and arg.ndim == 1 and weight is None:
                weight = arg
    if x is None:
        return None
    x_f = x.float()
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    ref = x_f * rms
    if weight is not None:
        ref = ref * weight.float()
    return ref


def _compute_topk_reference(args, func):
    """TopK: torch.topk(scores, k)."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    scores = None
    k = 2
    for pname, arg in zip(params, args):
        nl = pname.lower()
        if isinstance(arg, torch.Tensor) and arg.ndim == 2 and scores is None:
            if nl in ('scores', 'logits', 'x', 'input', 'hidden_states', 'gate_output', 'router_logits'):
                scores = arg
        elif isinstance(arg, int) and nl in ('top_k', 'k', 'topk', 'num_experts_per_tok'):
            k = arg
    if scores is None:
        for pname, arg in zip(params, args):
            if isinstance(arg, torch.Tensor) and arg.ndim == 2 and scores is None:
                scores = arg
    if scores is None:
        return None
    try:
        vals, _ = torch.topk(scores.float(), k, dim=-1)
        return vals
    except Exception:
        return None


def _check_moe_sorting_output(out, args, func, op):
    """Verify sorting output is a valid permutation."""
    out_t = _first_tensor(out)
    if out_t is None:
        return False, "kernel produced no tensor output"
    if out_t.dtype not in (torch.int32, torch.int64):
        return False, (
            f"sorting output has dtype {out_t.dtype} — moe_sorting must output "
            f"integer indices (int32/int64), not floating-point values"
        )
    vals = out_t.cpu().long().flatten()
    n = vals.numel()
    if n == 0:
        return True, "empty_output"
    if vals.min() < 0:
        return False, f"sorting output has negative indices (min={vals.min().item()})"
    uniq = vals.unique().numel()
    if uniq < n * 0.5:
        return False, f"sorting output has too many duplicates ({uniq}/{n} unique) — not a valid permutation"
    return True, "valid_sorting_output"


def _check_moe_routing_output(out, args, func, op):
    """Verify routing output: weights in [0,1] summing to ~1, indices in valid range."""
    out_tensors = []
    if isinstance(out, (list, tuple)):
        out_tensors = [t for t in out if isinstance(t, torch.Tensor)]
    elif isinstance(out, torch.Tensor):
        out_tensors = [out]
    if not out_tensors:
        return False, "kernel produced no tensor output"
    for t in out_tensors:
        if t.dtype in (torch.float32, torch.float16, torch.bfloat16) and t.ndim == 2:
            w = t.float()
            if w.min().item() < -0.05:
                return False, f"routing weights negative: min={w.min().item():.4f}"
            row_sums = w.sum(dim=-1)
            max_dev = (row_sums - 1.0).abs().max().item()
            if max_dev > 0.2 and w.shape[-1] > 1:
                return False, f"routing weights don't sum to ~1: max_deviation={max_dev:.4f}"
        elif t.dtype in (torch.int32, torch.int64) and t.ndim == 2:
            if t.min().item() < 0:
                return False, f"routing expert indices negative: min={t.min().item()}"
    return True, "valid_routing_output"


def _compute_quantization_check(out, args, func, op):
    """For quantization: dequantized output should approximate input."""
    out_t = _first_tensor(out)
    if out_t is None:
        return False, "kernel produced no tensor output"
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    x = None
    for pname, arg in zip(params, args):
        nl = pname.lower()
        if isinstance(arg, torch.Tensor) and arg.ndim == 2:
            if nl in ('x', 'input', 'hidden_states', 'inp'):
                x = arg
                break
    if x is None:
        for pname, arg in zip(params, args):
            if isinstance(arg, torch.Tensor) and arg.ndim == 2 and x is None:
                x = arg
    if x is None:
        return False, "could not find input tensor for quantization check"
    x_f = x.float()
    out_f = out_t.float()
    if out_f.shape != x_f.shape:
        if out_f.numel() != x_f.numel():
            if out_f.numel() < x_f.numel() * 0.5:
                return False, (
                    f"quantization output shape {tuple(out_t.shape)} too small for "
                    f"input shape {tuple(x.shape)} — not a real quantization kernel"
                )
            return False, (
                f"quantization output shape {tuple(out_t.shape)} does not match "
                f"input shape {tuple(x.shape)} — cannot verify quantization correctness"
            )
        out_f = out_f.reshape(x_f.shape)
    cos = torch.nn.functional.cosine_similarity(
        out_f.reshape(1, -1), x_f.reshape(1, -1)
    ).item()
    if cos < 0.85:
        return False, (
            f"quantized output does not approximate input (cosine={cos:.4f}) — "
            f"expected dequant(quant(x)) ≈ x"
        )
    return True, f"quantization_cos={cos:.4f}"


def _compute_causal_conv1d_reference(args, func):
    """Causal conv1d: F.conv1d with left-padding."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    x = weight = None
    for pname, arg in zip(params, args):
        nl = pname.lower()
        if isinstance(arg, torch.Tensor) and arg.ndim == 3 and x is None:
            if nl in ('x', 'input', 'hidden_states'):
                x = arg
        elif isinstance(arg, torch.Tensor) and weight is None:
            if nl in ('weight', 'conv_weight', 'w'):
                weight = arg
    if x is None:
        for pname, arg in zip(params, args):
            if isinstance(arg, torch.Tensor) and arg.ndim == 3 and x is None:
                x = arg
    if weight is None:
        for pname, arg in zip(params, args):
            if isinstance(arg, torch.Tensor) and arg.ndim == 3 and weight is None:
                weight = arg
    if x is None or weight is None:
        return None
    try:
        B, C, L = x.shape
        K = weight.shape[-1]
        x_padded = torch.nn.functional.pad(x.float(), (K - 1, 0))
        return torch.nn.functional.conv1d(x_padded, weight.float(), groups=C)
    except Exception:
        return None


def _check_chunk_gated_delta_sensitivity(out, args, func, harness, op):
    """Verify chunk_gated_delta output changes when gating signal is zeroed."""
    out_t = _first_tensor(out)
    if out_t is None:
        return False, "kernel produced no tensor output"
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    gate_idx = None
    for i, (pname, arg) in enumerate(zip(params, args)):
        if pname.lower() in ('g', 'gate', 'beta', 'decay') and isinstance(arg, torch.Tensor):
            gate_idx = i
            break
    if gate_idx is None:
        return False, "no gate parameter found — chunk_gated_delta must accept a gate/decay tensor"
    try:
        torch.manual_seed(42)
        args_zero = list(harness.make_args(func))
        args_zero[gate_idx] = torch.zeros_like(args_zero[gate_idx])
        out_zero = func(*args_zero)
        torch.cuda.synchronize()
        t_zero = _first_tensor(out_zero)
        if t_zero is not None and t_zero.shape == out_t.shape:
            if torch.allclose(out_t.float(), t_zero.float(), rtol=0.01, atol=0.01):
                return False, (
                    "output unchanged when gate/decay zeroed — "
                    f"kernel ignores gating signal, not a real {op}"
                )
    except Exception:
        pass
    return True, "gate_sensitivity_ok"


def _check_reference(out, ref, op, rtol=None, atol=None, cos_thresh=None):
    """Compare candidate output against reference. Returns (pass, message)."""
    out_t = _first_tensor(out)
    if out_t is None:
        return False, "kernel produced no tensor output"
    if ref is None:
        return True, "no_ref"

    out_f = out_t.float()
    ref_f = ref.float()

    if out_f.shape != ref_f.shape:
        if out_f.numel() != ref_f.numel():
            return False, f"shape mismatch: got {tuple(out_f.shape)}, expected {tuple(ref_f.shape)}"
        out_f = out_f.reshape(ref_f.shape)

    is_gemm = op in GEMM_OPS or op == 'moe_grouped_gemm'
    rtol = rtol if rtol is not None else (0.05 if is_gemm else 0.15)
    atol = atol if atol is not None else (0.02 if is_gemm else 0.05)
    cos_thresh = cos_thresh if cos_thresh is not None else (0.99 if is_gemm else 0.95)
    rel_thresh = 0.1 if is_gemm else 0.2

    if torch.allclose(out_f, ref_f, rtol=rtol, atol=atol):
        return True, "matches_reference"

    # If allclose fails, check cosine similarity (catches scale issues but right direction)
    cos_sim = torch.nn.functional.cosine_similarity(
        out_f.reshape(1, -1), ref_f.reshape(1, -1)
    ).item()
    if cos_sim > cos_thresh:
        # For GEMM ops, reject constant-scaled copies of the reference (e.g. mm(A,B)*alpha)
        if is_gemm:
            nonzero = ref_f.abs() > 1e-8
            if nonzero.sum() > 100:
                ratios = out_f[nonzero] / ref_f[nonzero]
                ratio_std = ratios.std().item()
                ratio_mean = ratios.mean().item()
                if ratio_std < 0.01 * abs(ratio_mean) and abs(ratio_mean - 1.0) > 0.05:
                    return False, (
                        f"output is constant-scaled reference (scale={ratio_mean:.4f}, "
                        f"cosine={cos_sim:.4f}) — kernel applies arbitrary alpha, not a real {op}"
                    )
        return True, f"cosine_sim={cos_sim:.4f}"

    # Check relative error statistics
    rel_err = (out_f - ref_f).abs() / (ref_f.abs() + 1e-8)
    median_rel = rel_err.median().item()
    max_rel = rel_err.max().item()

    if median_rel < rel_thresh:
        return True, f"median_rel_err={median_rel:.4f}"

    return False, (
        f"output differs from reference (cosine_sim={cos_sim:.4f}, "
        f"median_rel_err={median_rel:.4f}, max_rel_err={max_rel:.4f})"
    )


def _is_trivial_output(out_t, args, func):
    """Detect trivial outputs: all-zeros, identity copy, or element-wise activation of any input.
    Returns (is_trivial, reason) — applies to ALL non-GEMM, non-attention operators."""
    out_f = out_t.float()

    # Check all-zeros
    if out_f.abs().max().item() < 1e-8:
        return True, "output is all zeros (trivial kernel)"

    # Check all-ones
    if (out_f - 1.0).abs().max().item() < 1e-6:
        return True, "output is all ones (trivial kernel)"

    # Find all 2D tensor inputs to compare against
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    input_tensors = []
    for pname, arg in zip(params, args):
        if isinstance(arg, torch.Tensor) and arg.shape == out_t.shape:
            input_tensors.append((pname, arg))

    for pname, input_t in input_tensors:
        inp_f = input_t.float()

        # Identity: output == input (with bf16 tolerance)
        if torch.allclose(out_f, inp_f, rtol=1e-2, atol=1e-2):
            return True, f"output is identical to input '{pname}' (identity copy, not a real kernel)"

        # Negation: output == -input
        if torch.allclose(out_f, -inp_f, rtol=1e-2, atol=1e-2):
            return True, f"output is negation of input '{pname}' (trivial kernel)"

        # Absolute value: output == |input|
        if torch.allclose(out_f, inp_f.abs(), rtol=1e-2, atol=1e-2):
            return True, f"output is abs(input '{pname}') (trivial kernel)"

        # Element-wise square: output == x * x
        sq_ref = inp_f * inp_f
        if torch.allclose(out_f, sq_ref, rtol=1e-2, atol=1e-2):
            return True, f"output is element-wise square of input '{pname}' (x*x, trivial kernel)"

        # Element-wise cube: output == x * x * x
        cube_ref = inp_f * inp_f * inp_f
        if torch.allclose(out_f, cube_ref, rtol=1e-2, atol=1e-2):
            return True, f"output is element-wise cube of input '{pname}' (x³, trivial kernel)"

        # Softplus: output == log(1 + exp(x))
        softplus_ref = torch.nn.functional.softplus(inp_f)
        if torch.allclose(out_f, softplus_ref, rtol=1e-2, atol=1e-2):
            return True, f"output matches softplus('{pname}'), not a real kernel"

        # Approximate softplus: (x + sqrt(x^2+4))/2 or similar FlyDSL rocdl approximations
        softplus_cos = torch.nn.functional.cosine_similarity(
            out_f.reshape(1, -1), softplus_ref.reshape(1, -1)
        ).item()
        if softplus_cos > 0.98:
            return True, f"output resembles softplus('{pname}') (cosine={softplus_cos:.4f}), not a real kernel"

        # ReLU: max(input, 0) — check both torch.relu and manual clamp
        relu_ref = torch.relu(inp_f)
        if torch.allclose(out_f, relu_ref, rtol=1e-2, atol=1e-2):
            return True, f"output matches relu('{pname}'), not a real kernel"

        # Leaky ReLU
        lrelu_ref = torch.nn.functional.leaky_relu(inp_f, 0.01)
        if torch.allclose(out_f, lrelu_ref, rtol=1e-2, atol=1e-2):
            return True, f"output matches leaky_relu('{pname}'), not a real kernel"

        # Tanh
        tanh_ref = torch.tanh(inp_f)
        if torch.allclose(out_f, tanh_ref, rtol=1e-2, atol=1e-2):
            return True, f"output matches tanh('{pname}'), not a real kernel"

        # Sigmoid
        sig_ref = torch.sigmoid(inp_f)
        if torch.allclose(out_f, sig_ref, rtol=1e-2, atol=1e-2):
            return True, f"output matches sigmoid('{pname}'), not a real kernel"

        # ELU
        elu_ref = torch.nn.functional.elu(inp_f)
        if torch.allclose(out_f, elu_ref, rtol=1e-2, atol=1e-2):
            return True, f"output matches elu('{pname}'), not a real kernel"

        # GELU
        gelu_ref = torch.nn.functional.gelu(inp_f)
        if torch.allclose(out_f, gelu_ref, rtol=1e-2, atol=1e-2):
            return True, f"output matches gelu('{pname}'), not a real kernel"

        # SiLU / Swish
        silu_ref = torch.nn.functional.silu(inp_f)
        if torch.allclose(out_f, silu_ref, rtol=1e-2, atol=1e-2):
            return True, f"output matches silu('{pname}'), not a real kernel"

        # Scaled copy: output = alpha * input (constant scaling)
        # If ratio is constant across all elements, it's just scaling
        nonzero = inp_f.abs() > 1e-8
        if nonzero.sum() > 100:
            ratios = out_f[nonzero] / inp_f[nonzero]
            if ratios.std().item() < 0.01 * ratios.abs().mean().item():
                ratio_val = ratios.mean().item()
                if abs(ratio_val - 1.0) > 0.01:
                    return True, f"output is constant-scaled copy of input '{pname}' (scale={ratio_val:.4f}), not a real kernel"

        # Fake SiLU: output ≈ x * f(x) where f(x) is NOT real sigmoid
        nonzero_mask = inp_f.abs() > 0.1
        if nonzero_mask.sum() > 100:
            ratio = out_f[nonzero_mask] / inp_f[nonzero_mask]
            real_sigmoid = torch.sigmoid(inp_f[nonzero_mask])
            sigmoid_err = (ratio - real_sigmoid).abs()
            if sigmoid_err.median().item() > 0.1 and ratio.std().item() > 0.01:
                real_silu = torch.nn.functional.silu(inp_f)
                if not torch.allclose(out_f, real_silu, rtol=0.05, atol=0.05):
                    if ratio.min().item() > -0.5 and ratio.max().item() < 1.5:
                        return True, f"output resembles x*f(x) but f(x) is not sigmoid — fake SiLU/activation on '{pname}'"

    return False, "nontrivial_output"


def _structural_check_moe(out, args, func, op):
    """For MoE operators: verify output isn't a trivial element-wise function.
    Also check that output actually depends on weight/expert tensors, not just input."""
    out_t = _first_tensor(out)
    if out_t is None:
        return False, "kernel produced no tensor output"

    trivial, reason = _is_trivial_output(out_t, args, func)
    if trivial:
        return False, reason

    # MoE output should differ from a simple element-wise function of hidden_states.
    # Run kernel with zeroed-out weight tensors — if output is the same, weights aren't used.
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    try:
        torch.manual_seed(42)
        args_zeroed = list(harness.make_args(func))
        for i, (pname, arg) in enumerate(zip(params, args_zeroed)):
            nl = pname.lower()
            if isinstance(arg, torch.Tensor) and nl in (
                'w1', 'w2', 'w3', 'weight', 'gate_up_weight', 'down_weight',
                'topk_weights', 'expert_weights',
            ):
                args_zeroed[i] = torch.zeros_like(arg)
        out_zeroed = func(*args_zeroed)
        torch.cuda.synchronize()
        t_zeroed = _first_tensor(out_zeroed)
        if t_zeroed is not None and out_t.shape == t_zeroed.shape:
            if torch.allclose(out_t.float(), t_zeroed.float(), rtol=1e-2, atol=1e-2):
                return False, (
                    f"output does not change when weight tensors are zeroed — "
                    f"kernel ignores weights and is not a real {op} implementation"
                )
    except Exception:
        pass

    return True, "structural_ok"


def _structural_check_conv1d(out, args, func, op):
    """For causal_conv1d: verify output actually uses the convolution weight."""
    out_t = _first_tensor(out)
    if out_t is None:
        return False, "kernel produced no tensor output"

    trivial, reason = _is_trivial_output(out_t, args, func)
    if trivial:
        return False, reason

    # Zero-out conv weight and re-run — output should change
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    try:
        torch.manual_seed(42)
        args_zeroed = list(harness.make_args(func))
        for i, (pname, arg) in enumerate(zip(params, args_zeroed)):
            nl = pname.lower()
            if isinstance(arg, torch.Tensor) and nl in ('weight', 'conv_weight', 'b'):
                args_zeroed[i] = torch.zeros_like(arg)
        out_zeroed = func(*args_zeroed)
        torch.cuda.synchronize()
        t_zeroed = _first_tensor(out_zeroed)
        if t_zeroed is not None and out_t.shape == t_zeroed.shape:
            if torch.allclose(out_t.float(), t_zeroed.float(), rtol=1e-2, atol=1e-2):
                return False, (
                    "output does not change when conv weight is zeroed — "
                    "kernel ignores weight and is not a real convolution"
                )
    except Exception:
        pass

    return True, "structural_ok"


def _structural_check_recurrence(out, args, func, op):
    """For chunk_gated_delta: verify output is not a trivial element-wise function."""
    out_t = _first_tensor(out)
    if out_t is None:
        return False, "kernel produced no tensor output"

    trivial, reason = _is_trivial_output(out_t, args, func)
    if trivial:
        return False, reason

    # Zero-out k and v — output should change for real recurrence
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    try:
        torch.manual_seed(42)
        args_zeroed = list(harness.make_args(func))
        for i, (pname, arg) in enumerate(zip(params, args_zeroed)):
            nl = pname.lower()
            if isinstance(arg, torch.Tensor) and nl in ('k', 'v'):
                args_zeroed[i] = torch.zeros_like(arg)
        out_zeroed = func(*args_zeroed)
        torch.cuda.synchronize()
        t_zeroed = _first_tensor(out_zeroed)
        if t_zeroed is not None and out_t.shape == t_zeroed.shape:
            if torch.allclose(out_t.float(), t_zeroed.float(), rtol=1e-2, atol=1e-2):
                return False, (
                    "output does not change when k/v tensors are zeroed — "
                    "kernel ignores key/value inputs and is not a real recurrence"
                )
    except Exception:
        pass

    return True, "structural_ok"


def _structural_check_sorting(out, args, func, op):
    """For sorting/routing/topk: verify output isn't trivial AND depends on routing inputs."""
    out_t = _first_tensor(out)
    if out_t is None:
        return False, "kernel produced no tensor output"

    trivial, reason = _is_trivial_output(out_t, args, func)
    if trivial:
        return False, reason

    if out_t.dtype in (torch.int32, torch.int64):
        if out_t.min().item() < 0:
            return False, f"sorting output contains negative indices (min={out_t.min().item()})"

    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    _ROUTING_PARAMS = {
        'topk_weights', 'topk_ids', 'expert_ids', 'token_ids',
        'sorted_token_ids', 'expert_weights',
        'w1', 'w2', 'w3', 'weight', 'gate_up_weight', 'down_weight',
    }
    try:
        torch.manual_seed(42)
        args_zeroed = list(harness.make_args(func))
        zeroed_any = False
        for i, (pname, arg) in enumerate(zip(params, args_zeroed)):
            nl = pname.lower()
            if isinstance(arg, torch.Tensor) and nl in _ROUTING_PARAMS:
                args_zeroed[i] = torch.zeros_like(arg)
                zeroed_any = True
        if zeroed_any:
            out_zeroed = func(*args_zeroed)
            torch.cuda.synchronize()
            t_zeroed = _first_tensor(out_zeroed)
            if t_zeroed is not None and out_t.shape == t_zeroed.shape:
                if torch.allclose(out_t.float(), t_zeroed.float(), rtol=1e-2, atol=1e-2):
                    return False, (
                        f"output does not change when routing/weight tensors are zeroed — "
                        f"kernel ignores routing inputs and is not a real {op} implementation"
                    )
    except Exception:
        pass

    return True, "structural_ok"


def _structural_check_gemm(out, args, func, op):
    """For GEMM ops: verify output depends on weight/B tensor."""
    out_t = _first_tensor(out)
    if out_t is None:
        return False, "kernel produced no tensor output"

    trivial, reason = _is_trivial_output(out_t, args, func)
    if trivial:
        return False, reason

    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    try:
        torch.manual_seed(42)
        args_zeroed = list(harness.make_args(func))
        for i, (pname, arg) in enumerate(zip(params, args_zeroed)):
            nl = pname.lower()
            if isinstance(arg, torch.Tensor) and nl in (
                'b', 'w', 'weight', 'b_scale', 'w1', 'w2',
            ):
                args_zeroed[i] = torch.zeros_like(arg)
        out_zeroed = func(*args_zeroed)
        torch.cuda.synchronize()
        t_zeroed = _first_tensor(out_zeroed)
        if t_zeroed is not None and out_t.shape == t_zeroed.shape:
            if torch.allclose(out_t.float(), t_zeroed.float(), rtol=1e-2, atol=1e-2):
                return False, (
                    f"output does not change when weight/B tensor is zeroed — "
                    f"kernel ignores weights and is not a real {op} implementation"
                )
    except Exception:
        pass

    return True, "structural_ok"


# ── Deterministic double-run check ──
def _double_run_check(fname, func, harness):
    """Run kernel twice with identical inputs but fresh output buffers.
    If outputs differ, the kernel isn't writing to the output buffer (uses uninitialized memory)."""
    try:
        torch.manual_seed(42)
        args1 = harness.make_args(func)
        out1 = func(*args1)
        torch.cuda.synchronize()
        t1 = _first_tensor(out1)

        torch.manual_seed(42)
        args2 = harness.make_args(func)
        out2 = func(*args2)
        torch.cuda.synchronize()
        t2 = _first_tensor(out2)

        if t1 is None or t2 is None:
            return False, "kernel produced no tensor output — cannot verify correctness"
        if t1.shape != t2.shape:
            return False, f"kernel output shape changed between identical runs ({tuple(t1.shape)} vs {tuple(t2.shape)})"
        if not torch.allclose(t1.float(), t2.float(), rtol=1e-2, atol=1e-2):
            return False, (
                "output differs between two identical runs — kernel does not write to output buffer "
                "(likely uses torch.empty without writing, producing uninitialized garbage)"
            )
        return True, "deterministic"
    except Exception as exc:
        return False, f"kernel raised exception during double-run check: {exc}"


# ── Run the kernel and validate ──
def run_and_validate(fname, func, harness):
    """Run kernel, check output, compare against reference."""

    # Step 0: double-run determinism check (catches uninitialized output buffers)
    det_ok, det_msg = _double_run_check(fname, func, harness)
    if not det_ok:
        print(f"FAIL: {fname}() {det_msg}")
        print(f"  Your kernel must actually write computed results to the output tensor.")
        sys.exit(1)
    print(f"  Determinism check: {det_msg}")

    torch.manual_seed(42)
    try:
        args = harness.make_args(func)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"FAIL: could not build test inputs for {fname}\n{tb[-1000:]}")
        print(f"  Hint: wrapper signature must match op_harness shapes for operator='{operator}'")
        sys.exit(1)

    try:
        out = func(*args)
        torch.cuda.synchronize()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"FAIL: {fname}() raised an exception:\n{tb[-2000:]}")
        sys.exit(1)

    check_output(out, fname)

    # Reference-based validation by operator family
    op = operator.lower().replace('-', '_')

    # Universal trivial-output check for ALL operators
    # (catches identity, zeros, relu, tanh, sigmoid, etc. regardless of operator)
    out_t = _first_tensor(out)
    if out_t is not None:
        trivial, reason = _is_trivial_output(out_t, args, func)
        if trivial:
            print(f"FAIL: {fname}() {reason}")
            print(f"  Your kernel must implement the actual {operator} computation.")
            sys.exit(1)
        print(f"  Trivial-output check: {reason}")

    # ── Anti-GEMM check for non-GEMM operators ──
    if op in NON_GEMM_OPS and out_t is not None:
        is_fake, fake_reason = _is_naive_gemm_of_inputs(out_t, args, func, op)
        if is_fake:
            print(f"FAIL: {fname}() {fake_reason}")
            print(f"  Your kernel must implement the actual {operator} computation, not a matrix multiply.")
            sys.exit(1)
        print(f"  Anti-GEMM check: {fake_reason}")

    # ── Operator-specific reference checks ──
    if op in GEMM_OPS or op == 'moe_grouped_gemm':
        ref = _compute_gemm_reference(args, func)
        if ref is not None:
            ok, msg = _check_reference(out, ref, op)
            if not ok:
                print(f"FAIL: {fname}() {msg}")
                print(f"  Your kernel output does not match torch.mm reference for operator='{operator}'.")
                print(f"  This means your kernel is computing the wrong thing.")
                sys.exit(1)
            print(f"  Reference check: {msg}")
        if op == 'moe_grouped_gemm':
            ok, msg = _structural_check_moe(out, args, func, op)
            if not ok:
                print(f"FAIL: {fname}() {msg}")
                print(f"  moe_grouped_gemm must use expert weight tensors, not just hidden_states.")
                sys.exit(1)
            print(f"  MoE structural check: {msg}")
        if op in GEMM_OPS:
            ok, msg = _structural_check_gemm(out, args, func, op)
            if not ok:
                print(f"FAIL: {fname}() {msg}")
                print(f"  GEMM kernel must use weight/B tensor, not just the input.")
                sys.exit(1)
            print(f"  GEMM structural check: {msg}")

    elif op in ('attention', 'fmha', 'mha', 'mla', 'paged_attention'):
        ref = _compute_attention_reference(args, func)
        if ref is not None:
            ok, msg = _check_reference(out, ref, op)
            if not ok:
                print(f"FAIL: {fname}() {msg}")
                print(f"  Your kernel output does not match scaled_dot_product_attention reference.")
                sys.exit(1)
            print(f"  Reference check: {msg}")
        else:
            sig_attn = inspect.signature(func)
            attn_params = [p.lower() for p in sig_attn.parameters.keys()]
            has_qkv = any(p in ('q', 'query') for p in attn_params) and \
                       any(p in ('k', 'key') for p in attn_params)
            if not has_qkv:
                print(f"FAIL: {fname}() has no Q/K parameters — not an attention kernel")
                print(f"  Attention kernels must accept Q, K, V tensors.")
                sys.exit(1)

    elif op in ('rmsnorm', 'layernorm', 'normalization', 'rms_norm'):
        ref = _compute_rmsnorm_reference(args, func)
        if ref is not None:
            ok, msg = _check_reference(out, ref, op, rtol=0.05, atol=0.01, cos_thresh=0.98)
            if not ok:
                print(f"FAIL: {fname}() {msg}")
                print(f"  Your kernel output does not match RMSNorm reference.")
                sys.exit(1)
            print(f"  RMSNorm reference check: {msg}")

    elif op == 'topk':
        ref = _compute_topk_reference(args, func)
        if ref is not None:
            ok, msg = _check_reference(out, ref, op, rtol=0.05, atol=0.01, cos_thresh=0.99)
            if not ok:
                print(f"FAIL: {fname}() {msg}")
                print(f"  Your kernel output does not match torch.topk reference.")
                sys.exit(1)
            print(f"  TopK reference check: {msg}")
        ok, msg = _structural_check_sorting(out, args, func, op)
        if not ok:
            print(f"FAIL: {fname}() {msg}")
            sys.exit(1)
        print(f"  Structural check: {msg}")

    elif op == 'moe_sorting':
        ok, msg = _check_moe_sorting_output(out, args, func, op)
        if not ok:
            print(f"FAIL: {fname}() {msg}")
            sys.exit(1)
        print(f"  Sorting output check: {msg}")
        ok, msg = _structural_check_sorting(out, args, func, op)
        if not ok:
            print(f"FAIL: {fname}() {msg}")
            sys.exit(1)
        print(f"  Structural check: {msg}")

    elif op == 'moe_routing':
        ok, msg = _check_moe_routing_output(out, args, func, op)
        if not ok:
            print(f"FAIL: {fname}() {msg}")
            print(f"  Routing output must have valid weights (in [0,1], sum ~1) and valid expert indices.")
            sys.exit(1)
        print(f"  Routing output check: {msg}")
        ok, msg = _structural_check_sorting(out, args, func, op)
        if not ok:
            print(f"FAIL: {fname}() {msg}")
            sys.exit(1)
        print(f"  Structural check: {msg}")

    elif op == 'quantization':
        ok, msg = _compute_quantization_check(out, args, func, op)
        if not ok:
            print(f"FAIL: {fname}() {msg}")
            print(f"  Quantized output must approximately recover the input when dequantized.")
            sys.exit(1)
        print(f"  Quantization check: {msg}")

    elif op == 'causal_conv1d':
        ref = _compute_causal_conv1d_reference(args, func)
        if ref is not None:
            ok, msg = _check_reference(out, ref, op, rtol=0.05, atol=0.02, cos_thresh=0.98)
            if not ok:
                print(f"FAIL: {fname}() {msg}")
                print(f"  Your kernel output does not match F.conv1d causal reference.")
                sys.exit(1)
            print(f"  Conv1d reference check: {msg}")
        ok, msg = _structural_check_conv1d(out, args, func, op)
        if not ok:
            print(f"FAIL: {fname}() {msg}")
            sys.exit(1)
        print(f"  Structural check: {msg}")

    elif op == 'chunk_gated_delta':
        ok, msg = _check_chunk_gated_delta_sensitivity(out, args, func, harness, op)
        if not ok:
            print(f"FAIL: {fname}() {msg}")
            sys.exit(1)
        print(f"  Gate sensitivity check: {msg}")
        ok, msg = _structural_check_recurrence(out, args, func, op)
        if not ok:
            print(f"FAIL: {fname}() {msg}")
            sys.exit(1)
        print(f"  Structural check: {msg}")

    elif op in ('moe', 'moe_reduce', 'dispatch_combine'):
        ok, msg = _structural_check_moe(out, args, func, op)
        if not ok:
            print(f"FAIL: {fname}() {msg}")
            print(f"  Your kernel must implement the actual {operator} computation, not a simple activation.")
            sys.exit(1)
        print(f"  Structural check: {msg}")

    print(f"PASS: correctness ({fname} for operator='{operator}')")
    sys.exit(0)


# ── FlyDSL: try build_*/launch_*/compile_* pattern first ──
if language == "flydsl":
    build_fns = [
        (n, o) for n, o in inspect.getmembers(mod, inspect.isfunction)
        if callable(o) and re.match(r'(build_|get_flydsl_|compile_|launch_)', n)
    ]
    if build_fns:
        fname, build_fn = build_fns[0]
        harness = get_harness(operator)
        sig = inspect.signature(build_fn)
        kwargs = {}
        for pname, param in sig.parameters.items():
            if hasattr(harness, 'build_kwargs') and pname in harness.build_kwargs:
                kwargs[pname] = harness.build_kwargs[pname]
            elif param.default is not inspect.Parameter.empty:
                kwargs[pname] = param.default
        try:
            result = build_fn(**kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"FAIL: {fname}() build raised exception:\n{tb[-2000:]}")
            sys.exit(1)

        if callable(result):
            run_and_validate(fname, result, harness)
        else:
            print(f"PASS: correctness ({fname} returned non-callable, build succeeded)")
            sys.exit(0)

# ── Standard wrapper function path (triton / flydsl fallback) ──
def _is_jit(o): return hasattr(o, "fn") and hasattr(o, "run")
wrappers = [
    (n, o) for n, o in inspect.getmembers(mod, inspect.isfunction)
    if not n.startswith("_")
    and getattr(o, "__module__", "") == "candidate"
    and not _is_jit(o)
    and n != "main"
]

if not wrappers:
    print("FAIL: kernel.py has no public wrapper function")
    if language == "flydsl":
        print("  Hint: define a build_*() function that compiles/returns your FlyDSL kernel,")
        print("        or a public wrapper function that calls it.")
    else:
        print("  Hint: define a wrapper function (not starting with _) that calls your @triton.jit kernel")
    sys.exit(1)

harness = get_harness(operator)
fname, func = wrappers[0]
run_and_validate(fname, func, harness)
