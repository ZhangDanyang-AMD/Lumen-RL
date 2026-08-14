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
