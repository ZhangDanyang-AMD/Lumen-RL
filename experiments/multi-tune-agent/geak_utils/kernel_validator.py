"""Kernel validation: reject PyTorch replacement in wrapper."""

PYTORCH_OPS = [
    "torch.argsort", "torch.sort(", "torch.gather", "torch.cumsum",
    "torch.matmul", "torch.mm(", "torch.bmm(", "torch.einsum",
    "torch.nn.functional", "F.linear", "F.conv",
    ".mm(", ".matmul(", ".bmm(",
]

OPUS_MARKERS = [
    "__global__", "__device__", "__host__", "#include",
    "hipLaunchKernelGGL", "HIP_KERNEL_NAME",
    "threadIdx", "blockIdx", "blockDim",
]


def validate_kernel(content, original_path=None, language=None):
    """Validate kernel content.

    Returns (ok: bool, content_or_error: str).
    Checks vary by language:
    - triton: must contain @triton.jit, no PyTorch wrapper cheating
    - opus: must contain HIP/C++ markers, no PyTorch ops
    - flydsl: must be valid Python AST, no PyTorch compute ops
    """
    if len(content.strip()) < 10:
        return False, "Kernel content is too short."

    if language == "opus":
        has_hip = any(marker in content for marker in OPUS_MARKERS)
        if not has_hip:
            return False, (
                "OPUS kernel must contain HIP/C++ kernel code "
                "(e.g., __global__, __device__, #include, threadIdx). "
                "Do NOT write Python/Triton code for OPUS tasks."
            )
        for op in PYTORCH_OPS:
            if op in content:
                return False, f"OPUS kernel must not use {op}."
        return True, content

    if language == "flydsl":
        try:
            import ast
            ast.parse(content)
        except SyntaxError as e:
            return False, f"FlyDSL kernel must be valid Python: {e}"
        for op in PYTORCH_OPS:
            if op in content:
                return False, f"FlyDSL kernel must not use {op}."
        return True, content

    # triton (default) — original validation
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
    if "tl.load" not in jit_body and "tl.store" not in jit_body:
        if "pass" in jit_body and "tl." not in jit_body.replace("tl.constexpr", ""):
            return False, "Triton kernel is empty (pass-only). Write actual kernel logic with tl.load/tl.store."

    return True, content
