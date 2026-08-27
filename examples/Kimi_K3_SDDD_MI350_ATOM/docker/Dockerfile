# Teacher inference on ATOM instead of vLLM, for K3 DSpark on-policy distillation.
#
# vLLM's K3 decode path faults the GPU every 15-20 batches at B=64 (KDA
# recurrent-state slot handling when requests hit the max_tokens limit), which
# is shorter than the 50-batch round this training needs. ATOM decodes K3
# reliably and reaches hidden states through forward hooks, so generation and
# extraction share one engine instead of restarting between sweeps.
#
# The base ships aiter, mooncake, transformers, datasets, safetensors and
# triton_kernels, so this image only adds the LumenRL side. Notably it ships no
# vLLM — every vLLM import in lumenrl sits behind a backend branch and must stay
# that way.
#
# /!\ Pinned by digest, not `latest`. `latest` is a rolling tag and it has moved
# twice already in ways that broke this recipe. The image built from `latest` on
# 2026-08-24 landed on atom 0.1.6rc1.dev323 with torch 2.10.0+rocm7.2.4 — 48
# commits newer than the validated ATOM on an *older* torch/ROCm — and on that
# stack the teacher deadlocks: all eight TP ranks hang forever in a 2-element
# NCCL BROADCAST on the second extract_hidden request, which surfaces as
# "timed out waiting for response to cmd=extract_hidden after 600.0s" with zero
# training steps completed. The digest below is the combination the K3
# regeneration runbook validated over 60+ hours of continuous ATOM inference.
#
#   atom  0.1.6rc1.dev275+g83e71001e   torch 2.13.0+rocm7.14.0   HIP 7.14.60850
#
# Build from the repository root:
#   docker build -f examples/Kimi_K3_SDDD_MI350_ATOM/docker/Dockerfile \
#                -t kimi_k3_dspark_atom:latest .

FROM rocm/atom-dev@sha256:2f8bd4206ad15d014ae48115eae1ee9f1db83781848a8542de7177cfbd4ac914

SHELL ["/bin/bash", "-c"]

# Fail the build, not the run, if the base ever stops being that stack. A
# mismatch here is not cosmetic: see the deadlock described above.
RUN python3 - <<'PY'
import subprocess, torch
from importlib.metadata import version
import atom

expected = {
    "atom": "0.1.6rc1.dev275+g83e71001e",
    "torch": "2.13.0+rocm7.14.0",
    "hip": "7.14.60850",
    "commit": "83e71001e94602ee8e1d04581809bded49c3f08b",
}
# atom.__version__ raises AttributeError on this build; importlib.metadata is the
# only way to read it.
observed = {
    "atom": version("atom"),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "commit": subprocess.run(
        ["git", "-C", "/app/ATOM", "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip(),
}
bad = {k: (v, observed[k]) for k, v in expected.items() if observed[k] != v}
if bad:
    raise SystemExit(
        "base image is not the validated stack: "
        + ", ".join(f"{k}: expected {e}, got {g}" for k, (e, g) in bad.items())
    )
# third_party/ATOM is deliberately kept off PYTHONPATH; if it leaks in it
# shadows /app/ATOM and the run uses an ATOM nobody validated.
assert atom.__file__.startswith("/app/ATOM/"), atom.__file__
print("base stack OK:", observed)
PY

# ---- LumenRL's runtime dependencies -----------------------------------------
# Installed with --no-deps on purpose. Letting pip resolve these freely pulls
# torch 2.13 plus the whole NVIDIA CUDA stack over the ROCm build (accelerate
# asks for torch>=2.0 and pip happily "upgrades"), and constraining torch
# instead sends the resolver into ray's dependency graph until it gives up with
# resolution-too-deep. So the transitive deps that are genuinely missing are
# named explicitly below; everything else these packages want is already in the
# base image.
#
# Only what lumenrl actually imports. Deliberately absent: numba (nothing
# imports it, and it drags numpy pins along) and ray[default]'s extras (plain
# ray satisfies every `import ray`).
RUN TORCH_BEFORE=$(python3 -c "import torch; print(torch.__version__)") && \
    NUMPY_BEFORE=$(python3 -c "import numpy; print(numpy.__version__)") && \
    python3 -m pip install --no-cache-dir --no-deps \
        ray \
        omegaconf \
        accelerate \
        wandb \
        pyzmq && \
    # antlr4 is pinned to 4.9 because omegaconf 2.3's generated parser fails to
    # deserialize its grammar against 4.13. jsonschema is what `import ray` trips
    # on. opentelemetry is what `import wandb` trips on from 0.28 onward, where
    # wandb.analytics grew a hard dependency on it -- it is not optional and not
    # gated on WANDB_MODE, so it fails at import even though this recipe never
    # logs to wandb. All pure Python, so none of them can disturb torch, and the
    # guard below fails the build loudly if that stops being true.
    python3 -m pip install --no-cache-dir \
        "antlr4-python3-runtime==4.9.3" \
        jsonschema \
        opentelemetry-api \
        opentelemetry-sdk && \
    TORCH_AFTER=$(python3 -c "import torch; print(torch.__version__)") && \
    NUMPY_AFTER=$(python3 -c "import numpy; print(numpy.__version__)") && \
    if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ] || [ "$NUMPY_BEFORE" != "$NUMPY_AFTER" ]; then \
        echo "FATAL: pinned package moved: torch $TORCH_BEFORE -> $TORCH_AFTER, numpy $NUMPY_BEFORE -> $NUMPY_AFTER" >&2; \
        exit 1; \
    fi && \
    python3 -c "import omegaconf, zmq, ray, wandb, accelerate; print('deps import clean')" && \
    echo "preserved torch=$TORCH_AFTER numpy=$NUMPY_AFTER"

# ---- Lumen: quantized training kernels for the draft model -------------------
COPY third_party/Lumen /root/Lumen
RUN cd /root/Lumen && python3 -m pip install --no-deps -e . && \
    echo "Lumen installed"

# ---- LumenRL ----------------------------------------------------------------
COPY . /root/lumenrl
RUN cd /root/lumenrl && python3 -m pip install --no-deps -e . && \
    echo "LumenRL installed"

# ---- torchspec shim ---------------------------------------------------------
# ATOM's configure_hidden_states() imports its Mooncake plumbing from
# `torchspec`, a package rocm/atom-dev does not ship. The shim maps those two
# import paths onto lumenrl.transfer so teacher and trainer serialise hidden
# states through one implementation. Installed as a real package rather than
# patched into ATOM, so ATOM stays pristine.
RUN SITE=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cp -r /root/lumenrl/examples/Kimi_K3_SDDD_MI350_ATOM/docker/torchspec "$SITE/" && \
    python3 -c "\
from torchspec.config.mooncake_config import MooncakeConfig; \
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore; \
print('torchspec shim resolves to', MooncakeConfig.__module__, EagleMooncakeStore.__mro__[1].__module__)"

# ---- Verify the pieces that decide whether a run can start ------------------
# There is no GPU during a build, and aiter's import shells out to rocminfo, so
# anything downstream of it (atom.rollout.model_runner_ext, and our subclass of
# it) can only be imported at run time. Those are checked by source inspection
# here and for real by the smoke test.
RUN python3 - <<'PY'
import ast, importlib, importlib.util, inspect, pathlib, py_compile

for mod in ("atom", "atom.rollout.async_engine", "lumenrl",
            "mooncake.store", "torchspec",
            "lumenrl.transfer.eagle_mooncake_store"):
    importlib.import_module(mod)
    print(f"  OK  {mod}")

# lumen pulls in aiter, whose import shells out to rocminfo, so it can only be
# imported once a GPU is visible. Confirm it is at least installed and locatable.
for mod in ("lumen", "aiter"):
    assert importlib.util.find_spec(mod) is not None, f"{mod} is not installed"
    print(f"  OK  {mod} (installed; import deferred to runtime, needs a GPU)")

# ATOM dropped capture_mode from this signature; the worker must not pass it.
from atom.rollout.async_engine import AsyncLLMEngine
params = list(inspect.signature(AsyncLLMEngine.configure_hidden_states).parameters)
assert params == ["self", "aux_layer_ids", "mooncake_config"], params
assert hasattr(AsyncLLMEngine, "generate_hidden_states")
print("  OK  ATOM hidden-state API")

# The buffer-sizing helper defaults to Eagle3's 3 aux layers; every caller here
# passes num_aux_layers explicitly, so the parameter must still exist.
from lumenrl.transfer.eagle_mooncake_store import calculate_eagle3_buffer_size
assert "num_aux_layers" in inspect.signature(calculate_eagle3_buffer_size).parameters
print("  OK  buffer sizing accepts num_aux_layers")

# Our runner overrides these two ATOM methods. If a future ATOM renames them the
# overrides would quietly stop taking effect: capture would never park (wasting
# Mooncake bandwidth) and every TP rank would write. Check by source, since
# importing the module needs a GPU.
src = pathlib.Path(importlib.util.find_spec(
    "atom.rollout.model_runner_ext").origin).read_text()
tree = ast.parse(src)
atom_methods = {
    n.name for cls in tree.body if isinstance(cls, ast.ClassDef)
    for n in cls.body if isinstance(n, ast.FunctionDef)
}
for name in ("configure_hidden_states", "_store_hidden_states", "run_model"):
    assert name in atom_methods, f"ATOM no longer defines {name}: {sorted(atom_methods)}"
print("  OK  ATOM runner methods present")

py_compile.compile(
    "/root/lumenrl/lumenrl/engine/inference/atom_runner_ext.py", doraise=True)
print("  OK  runner subclass compiles")

# vLLM is absent by design; assert that importing lumenrl does not need it.
assert importlib.util.find_spec("vllm") is None, "vllm unexpectedly present"
print("  OK  no vllm dependency")
PY

WORKDIR /root/lumenrl

ENV PYTHONUNBUFFERED=1 \
    GLOG_minloglevel=3 \
    MOONCAKE_LOG_LEVEL=FATAL \
    AITER_LOG_LEVEL=WARNING
