"""Check the source-built vLLM actually works against primus's own torch.

Run inside the primus container with the NFS tree on the path:

    PYTHONPATH=/home/xysheng/vllm_primus/site python3 ~/4node/verify_vllm_primus.py

`import vllm` succeeding proves nothing -- the pure-python half imports fine even
when every compiled extension failed on a libtorch symbol, which is the exact
failure mode a cross-torch port produces. So the extensions are imported one by
one, and DeepSeek-V4 is checked separately because "vLLM works" and "vLLM can
load DSv4" are different claims.
"""

import importlib
import importlib.metadata as md
import os
import sys

# vLLM 0.26 split the extensions, so the old four-name checklist is wrong twice:
#   * `_moe_C` no longer exists -> `_moe_C_stable_libtorch`
#   * importing `_C` alone registers NOTHING; `rms_norm` / `rotary_embedding` come
#     from `_C_stable_libtorch` (measured: hasattr is False after `_C`, True after)
# Checking the old names reports false failures and sends you hunting a libtorch
# symbol problem that does not exist.
EXTENSIONS = [
    "vllm._C",
    "vllm._C_stable_libtorch",
    "vllm._moe_C_stable_libtorch",
    "vllm._rocm_C",
    "vllm.cumem_allocator",
]

failures = []


def check(label, fn):
    try:
        result = fn()
        print(f"  OK    {label}" + (f"  {result}" if result else ""))
        return True
    except Exception as exc:  # noqa: BLE001 - report, do not abort
        print(f"  FAIL  {label}: {type(exc).__name__}: {str(exc)[:180]}")
        failures.append(label)
        return False


print("=== versions")
import torch

print(f"  torch      {torch.__version__}")
import vllm

print(f"  vllm       {vllm.__version__}")
print(f"  vllm from  {os.path.dirname(vllm.__file__)}")
try:
    print(f"  metadata   {md.version('vllm')}")
except Exception as exc:  # noqa: BLE001
    print(f"  metadata   NOT FOUND ({type(exc).__name__})")
print(f"  python     {'.'.join(str(v) for v in sys.version_info[:3])}")

print("\n=== compiled extensions")
for mod in EXTENSIONS:
    check(mod, lambda m=mod: importlib.import_module(m) and None)

def need(cond, what):
    if not cond:
        raise RuntimeError(f"{what} not registered")


print("\n=== ops actually registered")
check("torch.ops._moe_C.topk_softmax",
      lambda: need(hasattr(torch.ops._moe_C, "topk_softmax"), "topk_softmax"))
check("torch.ops._C.rms_norm",
      lambda: need(hasattr(torch.ops._C, "rms_norm"), "rms_norm"))

print("\n=== DeepSeek-V4")

check("vllm.models.deepseek_v4.DeepseekV4ForCausalLM",
      lambda: __import__("vllm.models.deepseek_v4", fromlist=["DeepseekV4ForCausalLM"])
      .DeepseekV4ForCausalLM and None)

import vllm.models.deepseek_v4 as dsv4

root = os.path.dirname(dsv4.__file__)
backends = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and not d.startswith("__"))
print(f"  backends   {backends}")
if "amd" not in backends:
    failures.append("deepseek_v4/amd missing")

from vllm.model_executor.models import registry as _reg  # noqa: E402

entries = {}
for name in dir(_reg):
    if name.endswith("_MODELS") and isinstance(getattr(_reg, name), dict):
        entries.update(getattr(_reg, name))
hits = sorted(k for k, v in entries.items() if "v4" in str(v).lower())
print(f"  registry   {len(entries)} models, DSv4 entries: {hits}")
if "DeepseekV4ForCausalLM" not in entries:
    failures.append("DeepseekV4ForCausalLM not in registry")

print("\n=== RCCL still primus's own (RDMA depends on 2.28.9)")
import glob


def rccl():
    libs = glob.glob("/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/librccl.so")
    if not libs:
        raise RuntimeError("librccl.so not found where primus keeps it")
    import subprocess

    out = subprocess.run(["strings", libs[0]], capture_output=True, text=True).stdout
    ver = [l for l in out.splitlines() if l.startswith("RCCL version ")]
    return ver[0] if ver else "version string not found"


check("librccl", rccl)

print()
if failures:
    print(f"VERDICT: {len(failures)} FAILED -> {failures}")
    sys.exit(1)
print("VERDICT: vllm usable on primus (extensions + DSv4 + RCCL intact)")
