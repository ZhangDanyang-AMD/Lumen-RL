"""Import amdsmi before torch can, in every interpreter that uses this tree.

primus keeps the amdsmi python bindings out of sys.path, so vllm.platforms'
ROCm probe raises ModuleNotFoundError and vLLM silently becomes
UnspecifiedPlatform -- the rollout actor then dies with the unrecognisable
"RuntimeError: Device string must not be empty". Putting the SDK's
share/amd_smi on PYTHONPATH fixes that but breaks something worse:

``torch.cuda.device_count()`` prefers amdsmi over HIP (the ROCm analogue of the
nvml-based count), and torch guards its own ``import amdsmi`` with a ctypes hook
that redirects libamd_smi.so to whichever copy the loader finds first
(ROCm/amdsmi#72). On this image that redirect makes
``amdsmi_get_processor_handles()`` return an empty list, so every process that
imports torch first reports **0 GPUs**: Ray's raylet comes up with no GPU in its
resource list and all 8 actors hang on "No available node types can fulfill
resource request {'GPU': 1.0}", and torchrun children divide by zero on
``rank % torch.cuda.device_count()``.

Importing amdsmi first binds the bindings to their own libamd_smi.so; torch then
reuses the already-imported module and counts 8. Measured on primus v26.4:

    import amdsmi; import torch   -> device_count 8, RocmPlatform
    import torch;  import amdsmi  -> device_count 0, UnspecifiedPlatform

sitecustomize is the only hook that runs early enough, since .pth files are not
processed for plain PYTHONPATH entries. This shadows /usr/lib/python3.12's
sitecustomize, whose entire content is the apport hook, so that is repeated
below.
"""

try:
    import amdsmi  # noqa: F401
except Exception:  # noqa: BLE001 -- never let this break interpreter startup
    pass

try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()
