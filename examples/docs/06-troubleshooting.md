> [Examples README](../README.md) > Troubleshooting

# 6. Troubleshooting

**FP8 training diverges** (entropy ~0.04 / `grad_norm` 1e4+ / `rollout_corr/kl` 1e4+):
essentially only two causes — `FP8_PARAM_MANAGER` was not set to 0 (it conflicts with
native FSDP2's fp32 master weights), or the [vLLM RMSNorm patch](02-dependencies.md#27-vllm-aiter-rmsnorm-patch-required-for-examples-2-3-4) was never applied
(a new container needs it again).

**Falling back on memory (OOM)**:
- lower `policy.max_response_length=8192` + `max_total_sequence_length=9216` +
  `max_token_len_per_gpu=9216`;
- or lower `train_global_batch_size` / `gen_batch_size`;
- **do not** enable `fsdp_cfg.param_offload` / `optimizer_offload` on the Ray path; it
  fails with `parameters should be materialized on CPU`.

**Example 7's OOM is counterintuitive**: at `max_tokens_per_gpu: 22528` it dies in the
actor backward around step 14, but **the crash is not about peak allocated memory**
(~130 GB either way) — it is **fragmentation**. ROCm has no `expandable_segments`, and
roughly 7 bins per step each filled to 22.5k tokens repeatedly allocate and free huge
blocks, leaving reserved 42 GB above allocated. Capping bins at 8192 collapses the
fragmentation gap to 4-11 GB and drops peak reserved from 177 GB to 134 GB. **So that
8192 must not be casually raised back.**

**`weight sync (colocate-ipc) left N/M rollout parameters untouched`**: the sync missed
parameters. The exception lists the first 8 names; if they look like
`...experts.w13_weight` / `w2_weight`, the fused MoE routing did not take effect — either
the code is not up to date, or a vLLM/transformers upgrade invalidated the layout
assumption.

**ATOM rollout degradation** (with `MODE=atomfp8` / `atombf16`: `filter_groups: kept 0/96`
plus `Rollout reward: accuracy=0.0000` plus many `finished with reason max` and no `eos`
in the log): generation has broken down. Check first whether the plain RMSNorm in ATOM's
`atom/model_ops/layernorm.py` passes `use_model_sensitive_rmsnorm=1`; a misalignment
shows up first as an elevated `rollout_corr/kl` (~0.007 instead of ~0.004).
**Example 5 localizes this faster**: with quantization off, a still-elevated kl means the
problem is in the ATOM/training alignment rather than in FP8.

**Do not set `TORCHDYNAMO_DISABLE` by hand.** The script keeps it globally at `=1`
(dynamo off for training actors). The dynamo that examples 4 and 5 need for no-eager
level=3 rollout is injected by `ATOMReplicaManager` through `runtime_env` when it creates
the ATOM Ray actors, so `TORCHDYNAMO_DISABLE=0` applies **to the rollout processes only**.
A top-level `export TORCHDYNAMO_DISABLE=0` makes the training actors inherit it too,
which is pure side effect on the training side.

**Two more prerequisites for examples 4 and 5** (`MODE=atomfp8` / `atombf16` set both
automatically; do not drop them when overriding by hand):
`ATOM_ISOLATE_TORCH_COMPILE_CACHE=1` (otherwise 8 single-card replicas write the same
torch compile cache concurrently and trigger a `FileNotFoundError` in Inductor's
`write_atomic -> rename`), and `enable_sleep_mode=true` with `sleep_level=2` (releasing
KV cache / weights / CUDA graph after rollout, without which the training backward
easily hits `HSA_STATUS_ERROR_OUT_OF_RESOURCES`).

**Memory not released after a run finishes or is interrupted** (`rocm-smi` still shows
~90 GB per card while `ps` shows no trainer): `run_dapo.sh` cleans up processes **before**
starting, not afterwards, so ATOM EngineCore's `spawn_main` children (and their inductor
compile workers) become orphans still holding memory. When cleaning up manually, write
the patterns as `spawn[_]main` and so on, or `pkill -f` matches its own command line and
kills itself:

```bash
sudo docker exec "$CONTAINER" bash -lc '
  pkill -9 -f "compile_[w]orker"   || true
  pkill -9 -f "spawn[_]main"       || true
  pkill -9 -f "resource[_]tracker" || true
  sleep 8; rocm-smi --showmeminfo vram | grep -i used | head -3'
```

**Changing `DATA_ROOT` requires an unconditional assignment.** The
`docker run -e DATA_ROOT=...` in the container setup baked the value into the container
environment, so `export DATA_ROOT="${DATA_ROOT:-/new/disk}"` in a wrapper **has no effect**
(the variable already exists) and checkpoints silently go back to the old disk. Write
`export DATA_ROOT=/new/disk` directly.

**`logger.info` inside the vLLM worker does not reach the driver log.** That is why the
`weight sync coverage` line is invisible on example 7. This is **not** evidence that the
assertion did not run — to know whether it fired, check whether it raised.

For multi-node RDMA troubleshooting, see [Multi-node RDMA](05-multinode-rdma.md).
