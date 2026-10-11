"""Kernel validation: reject PyTorch replacement in wrapper."""

PYTORCH_OPS = [
    "torch.argsort", "torch.sort(", "torch.gather", "torch.cumsum",
    "torch.matmul", "torch.mm(", "torch.bmm(", "torch.einsum",
    "torch.nn.functional", "F.linear", "F.conv",
    ".mm(", ".matmul(", ".bmm(",
]


def validate_kernel(content, original_path=None):
    """Validate kernel content.

    Returns (ok: bool, content_or_error: str).
    Checks:
    1. Must contain @triton.jit (no full PyTorch replacement)
    2. Wrapper must not use PyTorch compute ops
    3. Triton kernel must not be empty (pass-only)
    """
    if "@triton.jit" not in content:
        return False, "Kernel must contain @triton.jit. Do NOT replace with PyTorch ops."

    # Check PyTorch compute ops in wrapper (code after last @triton.jit)
    parts = content.split("@triton.jit")
    wrapper = parts[-1] if len(parts) > 1 else ""
    for op in PYTORCH_OPS:
        if op in wrapper:
            return False, f"Wrapper must not use {op}. All computation must be in the Triton kernel."

    # Check Triton kernel is not empty
    jit_body = parts[1].split("def ")[0] if len(parts) > 1 and "def " in parts[1] else parts[1] if len(parts) > 1 else ""
    stripped = jit_body.replace(" ", "").replace("\n", "").replace("#", "")
    if "tl.load" not in jit_body and "tl.store" not in jit_body:
        # Kernel has no load/store — likely empty shell
        if "pass" in jit_body and "tl." not in jit_body.replace("tl.constexpr", ""):
            return False, "Triton kernel is empty (pass-only). Write actual kernel logic with tl.load/tl.store."

    return True, content
