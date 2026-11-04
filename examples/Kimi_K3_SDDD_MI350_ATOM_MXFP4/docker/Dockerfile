# Kimi-K3 DSpark MXFP4 training image.
#
# The base image is produced from Dockerfile.mxfp4-atom in this directory. It
# already contains the pinned ATOM runtime, Lumen's rebuilt AITER fork, and the
# current Lumen source. This final layer installs Lumen-RL and its ATOM/Mooncake
# adapter.
#
# Build only through the helper, from the Lumen-RL checkout:
#   bash examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/docker/build_mxfp4_atom.sh
#
ARG MXFP4_LUMEN_IMAGE="kimi_k3_dspark_mxfp4_lumen:latest"
FROM ${MXFP4_LUMEN_IMAGE}

SHELL ["/bin/bash", "-c"]

# Fail the build, not the run, if the base ever stops being the validated ATOM
# stack. A mismatch is not cosmetic: on a newer ATOM (0.1.6rc1.dev323, torch
# 2.10.0+rocm7.2.4) all eight teacher TP ranks hung in an NCCL broadcast on the
# second extract_hidden request, before any training step completed.
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

# The parent must be the completed Lumen MXFP4 image, not raw ATOM.
RUN test -s /etc/lumen/build-manifest.txt && \
    test -s /etc/lumen/aiter-build-manifest.txt && \
    grep -Fq "class MXFP4QuantWeightSelectionConfig" \
        /workspace/Lumen/lumen/quantize/config.py && \
    grep -Fq "invalidate_mxfp4_weight_caches" \
        /workspace/Lumen/lumen/quantize/__init__.py

# The pinned ATOM runtime has MXFP4 Linear/GEMM support, but its standalone
# DSpark proposer inherits the target model's quantization config. A quantized
# Draft must instead allocate/load from its own config.json.
RUN python3 - <<'PY'
from pathlib import Path

path = Path("/app/ATOM/atom/spec_decode/dspark_proposer.py")
source = path.read_text(encoding="utf-8")
import_anchor = "from atom.config import CompilationLevel"
replacement_import = "from atom.config import CompilationLevel, QuantizationConfig"
assert source.count(import_anchor) == 1
source = source.replace(import_anchor, replacement_import)

config_anchor = """        draft_atom_config.hf_config = draft_hf
        draft_atom_config.compilation_config = copy.copy(self.config.compilation_config)"""
config_replacement = """        draft_atom_config.hf_config = draft_hf
        if getattr(draft_hf, "quantization_config", None) is not None:
            draft_atom_config.quant_config = QuantizationConfig(draft_hf)
            draft_atom_config.online_quant_config = None
            logger.info(
                "DSpark draft: using checkpoint-local quantization config (%s).",
                draft_atom_config.quant_config.quant_method,
            )
        draft_atom_config.compilation_config = copy.copy(self.config.compilation_config)"""
assert source.count(config_anchor) == 1
path.write_text(
    source.replace(config_anchor, config_replacement),
    encoding="utf-8",
)
patched = path.read_text(encoding="utf-8")
assert "QuantizationConfig(draft_hf)" in patched
assert "draft_atom_config.online_quant_config = None" in patched

# ATOM's TP-group reuse plugin constructs AITER CustomAllreduce IPC pools around
# vLLM's group. On this ROCm stack those pools receive VMM-backed pointers that
# hipIpcGetMemHandle rejects before TP=8 startup. Reuse vLLM's already-created
# PyNccl/RCCL communicator instead; this path needs correctness and compatibility
# more than the optional custom/quick IPC collective.
tp_group_path = Path("/app/ATOM/atom/plugin/vllm/tp_group_reuse.py")
tp_group_source = tp_group_path.read_text(encoding="utf-8")
signal_anchor = "        _setup_ca_comm_signal(adapter, tensor_model_parallel_size)\n"
assert tp_group_source.count(signal_anchor) == 1
custom_flag_anchor = (
    "            self.use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE\n"
)
assert tp_group_source.count(custom_flag_anchor) == 1
quick_ar_anchor = """            self.qr_comm = (
                QuickAllReduce(group=self.cpu_group, device=self.device)
                if self.world_size > 1
                else None
            )
"""
assert tp_group_source.count(quick_ar_anchor) == 1
ca_required_anchor = """    if device_communicator.ca_comm is None or device_communicator.ca_comm.disabled:
        logger.warning(
            "ATOM tp_group_reuse: aiter ca_comm not available on vLLM's TP group, "
            "caller will fall back to standard aiter distributed initialization "
            "(e.g., via aiter.init_dist_env(...))"
        )
        return None
"""
assert tp_group_source.count(ca_required_anchor) == 1
enable_anchor = "        set_custom_all_reduce(True)\n"
assert tp_group_source.count(enable_anchor) == 1
tp_group_source = tp_group_source.replace(
    signal_anchor,
    "        # PyNccl/RCCL needs no auxiliary CustomAllreduce signal.\n",
)
tp_group_source = tp_group_source.replace(
    custom_flag_anchor,
    "            self.use_custom_allreduce = False\n",
)
tp_group_source = tp_group_source.replace(
    quick_ar_anchor,
    "            self.qr_comm = None\n",
)
tp_group_source = tp_group_source.replace(
    ca_required_anchor,
    '    logger.info("ATOM tp_group_reuse: using vLLM PyNccl/RCCL communicator")\n',
)
tp_group_source = tp_group_source.replace(
    enable_anchor,
    "        set_custom_all_reduce(False)\n",
)
tp_group_path.write_text(
    tp_group_source,
    encoding="utf-8",
)
patched_tp_group = tp_group_path.read_text(encoding="utf-8")
assert signal_anchor not in patched_tp_group
assert "self.use_custom_allreduce = False" in patched_tp_group
assert "set_custom_all_reduce(False)" in patched_tp_group
print("ATOM standalone DSpark checkpoint-local quantization patch: OK")
print("ATOM TP-group reuse uses vLLM PyNccl/RCCL only: OK")
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
        "ray==2.56.1" \
        "omegaconf==2.3.1" \
        "accelerate==1.14.0" \
        "wandb==0.28.1" \
        "pyzmq==26.4.0" && \
    # antlr4 is pinned to 4.9 because omegaconf 2.3's generated parser fails to
    # deserialize its grammar against 4.13. jsonschema is what `import ray` trips
    # on. Pin its small dependency closure as well; allowing the resolver to
    # choose latest previously upgraded OpenTelemetry past LMCache's supported
    # range. WandB 0.28.1 needs no OpenTelemetry package when disabled here.
    python3 -m pip install --no-cache-dir --no-deps \
        "antlr4-python3-runtime==4.9.3" \
        "jsonschema==4.26.0" \
        "jsonschema-specifications==2025.9.1" \
        "referencing==0.37.0" \
        "rpds-py==2026.6.3" && \
    TORCH_AFTER=$(python3 -c "import torch; print(torch.__version__)") && \
    NUMPY_AFTER=$(python3 -c "import numpy; print(numpy.__version__)") && \
    if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ] || [ "$NUMPY_BEFORE" != "$NUMPY_AFTER" ]; then \
        echo "FATAL: pinned package moved: torch $TORCH_BEFORE -> $TORCH_AFTER, numpy $NUMPY_BEFORE -> $NUMPY_AFTER" >&2; \
        exit 1; \
    fi && \
    echo "preserved torch=$TORCH_AFTER numpy=$NUMPY_AFTER" && \
    python3 - <<'PY'
from importlib.metadata import version

import accelerate
import jsonschema
import omegaconf
import ray
import wandb
import zmq

expected = {
    "accelerate": "1.14.0",
    "antlr4-python3-runtime": "4.9.3",
    "jsonschema": "4.26.0",
    "jsonschema-specifications": "2025.9.1",
    "omegaconf": "2.3.1",
    "pyzmq": "26.4.0",
    "ray": "2.56.1",
    "referencing": "0.37.0",
    "rpds-py": "2026.6.3",
    "wandb": "0.28.1",
}
for package, wanted in expected.items():
    actual = version(package)
    assert actual == wanted, f"{package}: expected {wanted}, got {actual}"
print("deps import clean:", expected)
PY

# ---- LumenRL ----------------------------------------------------------------
COPY . /root/lumenrl
RUN cd /root/lumenrl && python3 -m pip install --no-deps -e . && \
    echo "LumenRL installed"

# ---- torchspec shim ---------------------------------------------------------
# ATOM's configure_hidden_states() imports its Mooncake plumbing from
# `torchspec`, a package rocm/atom-dev does not ship. The shim maps those two
# import paths onto lumenrl.transfer so teacher and trainer serialise hidden
# states through one implementation. Installed as a real package rather than
# patched into ATOM, so ATOM's own sources carry only the two asserted patches
# above.
RUN SITE=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cp -r /root/lumenrl/examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/docker/torchspec "$SITE/" && \
    python3 -c "\
from torchspec.config.mooncake_config import MooncakeConfig; \
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore; \
print('torchspec shim resolves to', MooncakeConfig.__module__, EagleMooncakeStore.__mro__[1].__module__)"

# ---- Verify the pieces that decide whether a run can start ------------------
# There is no GPU during a build, and aiter's import shells out to rocminfo, so
# anything downstream of it (atom.rollout.model_runner_ext, and our subclass of
# it) can only be imported at run time. Those are checked by source inspection
# here and for real when a run starts.
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

# Provenance only. Every RUN after an ARG sees it as an environment variable and
# a changed LABEL invalidates the layers after it, so this stays last to keep a
# per-build value such as BUILD_DATE from re-running the steps above.
ARG MXFP4_LUMEN_IMAGE
ARG MXFP4_LUMEN_IMAGE_ID="unknown"
ARG BUILD_DATE="unknown"
ARG LUMEN_RL_REVISION="unknown"
ARG LUMEN_RL_SOURCE_STATE="unknown"

LABEL org.opencontainers.image.description="Kimi-K3 DSpark MXFP4 training with Lumen-RL and ATOM"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL com.amd.lumenrl.mxfp4.base.image="${MXFP4_LUMEN_IMAGE}"
LABEL com.amd.lumenrl.mxfp4.base.image.id="${MXFP4_LUMEN_IMAGE_ID}"
LABEL org.opencontainers.image.revision="${LUMEN_RL_REVISION}"
LABEL com.amd.lumenrl.source.state="${LUMEN_RL_SOURCE_STATE}"

RUN printf '%s\n' \
        "lumenrl_revision=${LUMEN_RL_REVISION}" \
        "lumenrl_source_state=${LUMEN_RL_SOURCE_STATE}" \
        "lumen_base=${MXFP4_LUMEN_IMAGE}" \
        "lumen_base_image_id=${MXFP4_LUMEN_IMAGE_ID}" \
        "build_date=${BUILD_DATE}" \
        > /etc/lumen/lumenrl-build-manifest.txt

WORKDIR /root/lumenrl

ENV PYTHONUNBUFFERED=1 \
    GLOG_minloglevel=3 \
    MOONCAKE_LOG_LEVEL=FATAL \
    AITER_LOG_LEVEL=WARNING

ENTRYPOINT []
CMD ["bash"]
