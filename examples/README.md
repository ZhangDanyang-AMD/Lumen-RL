# LumenRL Examples Runbook

Reproduce the DAPO math RL training examples in this directory from scratch on a
**fresh 8-GPU AMD machine**.

> Chinese version: [README_cn.md](README_cn.md)

Every example shares one entrypoint (`lumenrl.trainer.main`, the Ray controller) and
one launch script (`examples/DAPO/run_dapo.sh`). All differences are expressed through
the config plus environment variables: **8 training actors and 8 co-located rollout
replicas (TP=1) inside a single Ray-driver process, with train->rollout weights synced
over ZMQ CUDA-IPC**.

On the algorithm side: clip-higher + dual-clip + token-mean policy loss, GRPO with
per-uid group normalization, dynamic sampling via `filter_groups`, an overlong reward
buffer, and TIS rollout correction.

**The whole thing in one line**: set the path variables -> clone the repos -> start the
container and install dependencies -> (FP8 only) apply the patch -> download models and
data -> smoke -> launch the long run with `docker exec -d`.

> ⚡ **Would rather not build from source?** Examples 1–7 have a fast path that uses the
> **published container image**: the software stack is already pinned with its kernels
> baked in, each example is a single command, and the launcher compares the metrics
> against reference values to print PASS/FAIL.
> See [**8. Running the seven examples from the release image**](docs/08-release.md) —
> three commands from `docker pull` to a verdict, and chapters 1–4 of this page can be
> skipped entirely. You need the from-scratch path below only to change the source,
> swap models, or run two nodes (example 8).

---

## Verified examples

| # | Example | Model | Training backend | Rollout / precision | GPU | Runtime | Switch |
|---|---|---|---|---|---|---|---|
| 1 | 8B BF16 baseline | Qwen3-8B-Base | Lumen FSDP2, BF16 | vLLM / BF16 | 8x MI355X (gfx950) / 8x MI325X (gfx942) | `vllm/vllm-openai-rocm:v0.23.0` | `MODE=bf16` |
| 2 | 8B FP8 rollout | Qwen3-8B-Base | Lumen FSDP2, BF16 | vLLM / `fp8_per_block` | same | same | `MODE=fp8` |
| 3 | 8B FP8 E2E | Qwen3-8B-Base | Lumen FSDP2, **FP8 blockwise2d** | vLLM / `fp8_per_block` | same | same | `MODE=fp8 TRAIN_FP8=1` |
| 4 | 8B ATOM FP8 | Qwen3-8B-Base | Lumen FSDP2, **FP8 blockwise2d** | **ATOM** / `per_block_fp8` | same | same | `MODE=atomfp8 TRAIN_FP8=1` |
| 5 | 8B ATOM BF16 | Qwen3-8B-Base | Lumen FSDP2, BF16 (pure BF16, no Lumen norm patch) | **ATOM** / BF16 | same | same | `MODE=atombf16` |
| 6 | MoE FSDP2 | Qwen3-30B-A3B-Base | Lumen FSDP2, BF16 | vLLM / BF16 | same | same | `MODE=bf16` + MoE config |
| 7 | MoE Megatron EP=8 | Qwen3-30B-A3B-Base | **Megatron-Native**, TP=PP=CP=1, EP=8, DP=8 | vLLM / BF16 | same | same | `MODE=bf16` + Megatron config |
| 8 | MoE 2-node RDMA | Qwen3-30B-A3B | **Megatron-Native**, TP=4, EP=8 | vLLM TP=2 x 4 / BF16 | 2x 8x MI308X (gfx942) | `qwen3-30b-a3b:rollout` + `trainer` | [Disaggregated guide](docs/07-disaggregated-rdma.md) |

Examples 1-7 have been run on **8x MI355X** and **8x MI325X**: smoke plus long run,
exit 0, no traceback, no OOM, no `HSA_STATUS`, weight-sync coverage assertions all
passing, and memory back to the ~298 MB/card idle baseline afterwards.

Example 8 runs on **2x 8x MI308X** with disaggregated training and inference: Megatron
trainer on node 1, vLLM rollout on node 2, connected via RCCL/RoCE GPU Direct RDMA
weight sync (9-rank process group). See the
[full deployment guide](docs/07-disaggregated-rdma.md).

> **Example 5 is the BF16 control for example 4**: same ATOM rollout engine, same
> no-eager level=3 + sleep2, with only the rollout's online quantization and the training
> side's FP8 turned off.

> **You cannot run two training backends on the same cards at once**, and you cannot
> share a node with someone else — the engine budgets KV cache as a fraction of the whole
> card. Confirm memory is at the idle baseline before starting.

> **The two training backends cannot share a checkpoint directory**; the formats differ.

---

## Preflight

Four checks, each of which has cost someone an hour. Run them after
[Dependencies](docs/02-dependencies.md) and before the first smoke.

```bash
# 1. The image must be v0.23.0. A newer vllm/vllm-openai-rocm tag on the machine
#    is not a substitute, and `:latest` is often an older vLLM.
sudo docker exec "$CONTAINER" bash -lc \
  'python3 -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"'
#    expect: 0.23.0 5.12.0

# 2. Every card at the ~298 MB idle baseline. Anything higher is a co-tenant or
#    an orphan from an interrupted run -- see Troubleshooting before launching.
sudo docker exec "$CONTAINER" bash -lc 'rocm-smi --showmeminfo vram | grep -i used'

# 3. Source installs must win over the image's wheels (PYTHONPATH is required).
sudo docker exec "$CONTAINER" bash -lc \
  'export PYTHONPATH="$RL_ROOT/Lumen-RL:$AITER_DIR:$LUMEN_DIR";
   python3 -c "import aiter, lumenrl, lumen;
from aiter import flash_attn_varlen_func; print(aiter.__file__)"'
#    expect a path under $RL_ROOT/aiter/

# 4. Example 7 only: the TE layer spec must build, not just import megatron.core.
sudo docker exec "$CONTAINER" bash -lc \
  'python3 -c "from megatron.core.models.gpt.gpt_layer_specs import \
get_gpt_layer_with_transformer_engine_spec as s; print(type(s()).__name__)"'
#    expect ModuleSpec
```

**Running several examples in one session:** clear the compile caches between any two
ATOM runs of different precision, or example 5 after example 4 dies in AOTAutograd
(see [Troubleshooting](docs/06-troubleshooting.md)) — torch's inductor cache is not
scoped per run.

```bash
sudo docker exec "$CONTAINER" bash -lc \
  'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
```

Per-example extras that are easy to miss: examples 2, 3 and 4 need the
[RMSNorm patch](docs/02-dependencies.md#27-vllm-aiter-rmsnorm-patch-required-for-examples-2-3-4)
reapplied after any `docker rm`; examples 4 and 5 need the
[ATOM JIT precompile](docs/02-dependencies.md#28-atom-jit-precompilation-required-for-examples-4-5);
examples 6 and 7 need `MODEL_PATH` pointed at the MoE model explicitly plus
`SCRATCH_ROOT` exported, and example 7 needs TransformerEngine built from source.

---

## Documentation

| Step | Document | What it covers |
|---|---|---|
| 1 | [Environment setup](docs/01-env-setup.md) | Path variables, cloning repos, starting the container |
| 2 | [Dependencies](docs/02-dependencies.md) | pip installs, vLLM patch, ATOM JIT precompile, import chain verification |
| 3 | [Models and data](docs/03-data.md) | Download from HuggingFace / ModelScope, prompt filtering |
| 4 | [Launching](docs/04-launching.md) | Configs, smoke runs, long runs, health criteria, monitoring |
| 5 | [Multi-node RDMA](docs/05-multinode-rdma.md) | RDMA pre-checks, launch/checkpoint verification, baselines (two-node only) |
| 6 | [Troubleshooting](docs/06-troubleshooting.md) | All known failure modes and fixes |
| 7 | [Disaggregated 2-node RDMA](docs/07-disaggregated-rdma.md) | Full deployment: Megatron trainer + vLLM rollout, RDMA weight sync, from-zero setup |
| 8 | [**Running from the release image**](docs/08-release.md) | ⚡ **Fast path, replaces steps 1–4**: `docker pull` plus one command per example for 1–7, with node clearing, predictable logs, metric checking (`--check`), and reference values with tolerances |

Steps 1–7 build **from source**; step 8 uses the **published image**. Both run the same
examples (1–7 in step 8 are 1–7 in the table above), so the metrics are directly
comparable. Example 8, the two-node deployment, is covered only by step 7.
