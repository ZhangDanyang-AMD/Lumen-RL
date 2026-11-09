"""Kernel validation: enforce language-specific constraints and block cross-language cheating."""

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

# Markers that indicate Python/Triton code being used in wrong-language tasks
TRITON_MARKERS = ["@triton.jit", "import triton", "triton.language"]
PYTHON_IMPORT_MARKERS = ["import torch", "from torch", "import triton", "from triton"]


def validate_kernel(content, original_path=None, language=None):
    """Validate kernel content. Enforces language-specific constraints.

    Returns (ok: bool, content_or_error: str).

    Triton:
      - Must contain @triton.jit
      - Must NOT have PyTorch compute ops in wrapper
      - Must NOT be written in HIP C++ (no __global__ in Python file)

    OPUS (kernel.cu):
      - Must contain HIP/C++ markers (__global__, threadIdx, etc.)
      - Must NOT have PyTorch ops
      - Must NOT be a Python/Triton file masquerading as HIP C++

    FlyDSL:
      - Must be valid Python AST
      - Must NOT have PyTorch compute ops
      - Must NOT be a Triton kernel (@triton.jit forbidden)
      - Must NOT be HIP C++ code in a .py file
    """
    if len(content.strip()) < 10:
        return False, "Kernel content is too short."

    # ── OPUS ──────────────────────────────────────────────────────────────────
    if language == "opus":
        # Must have HIP/C++ markers
        has_hip = any(marker in content for marker in OPUS_MARKERS)
        if not has_hip:
            return False, (
                "OPUS kernel (kernel.cu) must contain HIP/C++ kernel code "
                "(e.g., __global__, __device__, #include, threadIdx). "
                "Do NOT write Python/Triton code for OPUS tasks. "
                "Write a proper HIP C++ kernel with __global__ functions."
            )
        # Must NOT be Python/Triton code pretending to be HIP
        if "import triton" in content or "@triton.jit" in content:
            return False, (
                "OPUS kernel must be HIP C++, not Triton Python. "
                "Remove all Python/Triton code and write __global__ HIP C++ kernels."
            )
        for op in PYTORCH_OPS:
            if op in content:
                return False, f"OPUS kernel must not use {op}."
        return True, content

    # ── FLYDSL ────────────────────────────────────────────────────────────────
    if language == "flydsl":
        # Must be valid Python
        try:
            import ast
            ast.parse(content)
        except SyntaxError as e:
            return False, f"FlyDSL kernel must be valid Python: {e}"
        # Must NOT be a Triton kernel
        if "@triton.jit" in content or "import triton" in content:
            return False, (
                "FlyDSL kernel must NOT use Triton (@triton.jit is forbidden). "
                "This is a FlyDSL task — write FlyDSL DSL code, not Triton. "
                "The sandbox rejects any kernel with import triton or @triton.jit."
            )
        # Must NOT use HIP C++ constructs in Python
        if "__global__" in content or "__device__" in content:
            return False, (
                "FlyDSL kernel is Python code — do not use HIP C++ (__global__, __device__). "
                "Write FlyDSL DSL code."
            )
        for op in PYTORCH_OPS:
            if op in content:
                return False, f"FlyDSL kernel must not use {op}."
        return True, content

    # ── TRITON (default) ──────────────────────────────────────────────────────
    if "@triton.jit" not in content:
        return False, (
            "Triton kernel must contain @triton.jit. "
            "Do NOT replace the kernel with PyTorch ops or HIP C++ code. "
            "Write a proper @triton.jit decorated kernel function."
        )

    # Must NOT be HIP C++ masquerading as Python
    if "__global__" in content and "import triton" not in content:
        return False, (
            "Triton kernel.py must be Python — do not write HIP C++ (__global__) in kernel.py. "
            "For HIP kernels, use kernel.cu (OPUS language)."
        )

    # Check PyTorch compute ops in wrapper (code after last @triton.jit)
    parts = content.split("@triton.jit")
    wrapper = parts[-1] if len(parts) > 1 else ""
    for op in PYTORCH_OPS:
        if op in wrapper:
            return False, (
                f"Wrapper must not use {op}. "
                "All computation must be in the @triton.jit kernel. "
                "The wrapper may only set up tensors and launch the kernel."
            )

    # Check Triton kernel is not empty
    jit_body = parts[1].split("def ")[0] if len(parts) > 1 and "def " in parts[1] else parts[1] if len(parts) > 1 else ""
    if "tl.load" not in jit_body and "tl.store" not in jit_body:
        if "pass" in jit_body and "tl." not in jit_body.replace("tl.constexpr", ""):
            return False, (
                "Triton kernel is empty (pass-only). "
                "Write actual kernel logic using tl.load, tl.store, tl.dot, etc."
            )

    return True, content
