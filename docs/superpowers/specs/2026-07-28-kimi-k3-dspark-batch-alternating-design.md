# Kimi K3 DSpark Batch-Alternating SDDD Design

## Problem

K3 (~1TB MoE model, 93 layers, 896 experts) requires TP=8 for vLLM inference, consuming all 8 GPUs on an MI350 node. This prevents the existing 4+4 disaggregated architecture (4 train / 4 infer). We need a new execution mode where 8 GPUs alternate between prefill hidden state extraction and draft training.

## Architecture: Batch-Alternating Loop

```
for round in range(total_rounds):
    # Phase A: Prefill (rank 0 only, vLLM subprocess TP=8)
    if rank == 0:
        vllm.start()                              # ~5-10 min for K3 TP=8
        round_dir = f"{cache_dir}/round_{round}"
        for i, batch in enumerate(next_N_batches(dataset)):
            hidden_states = vllm.extract_hidden_states(batch)  # Mooncake receive
            write_to_disk(round_dir, i, hidden_states)         # bin files + meta.pt
        vllm.shutdown()
    barrier()
    torch.cuda.empty_cache()

    # Phase B: Train (all ranks, composable replicate)
    load_draft_to_gpu()
    for i in range(N):
        teacher_data = load_from_disk(round_dir, i)  # read from NVMe
        loss = dspark_train_step(teacher_data)
        loss.backward()
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    offload_draft_to_cpu()
    cleanup(round_dir)
    barrier()
```

### Hidden State Transfer Pipeline

The vLLM → Mooncake path is reused from the existing 4+4 pipeline:

1. **vLLM subprocess** runs prefill with `extract_hidden_states` speculative mode
2. **MooncakeHiddenStatesConnector** (inside vLLM) stores hidden states to Mooncake TCP
3. **EagleMooncakeStore** (training process, rank 0) reads hidden states from Mooncake via `_teacher_inference_rank0()`

Steps 1-3 are identical to Kimi K25 SDDD (ATOM backend) and GPT-OSS (vLLM backend). The difference is what happens **after** rank 0 receives the hidden states: instead of the streaming `_ShmWriterThread`/`_ShmLoaderThread` pipeline (designed for concurrent prefetch+train), batch-alternating mode writes all batches to NVMe disk cache directories, then reads them sequentially during Phase B. See "Disk Cache Budget" section below for details.

### Prefill Mode (not Generate)

The dataset (`kimi-mtp-dataset`) already contains complete sequences with responses. The teacher does a **prefill-only forward pass** to extract hidden states — no autoregressive generation needed.

```yaml
teacher:
  generate_mode: prefill   # NOT generate — dataset has responses
```

This uses `_teacher_forward_vllm()` → `extract_hidden` command path, which is faster than `generate_extract` and avoids sampling/decoding overhead.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequential_mode` | `null` | `null` = existing 4+4, `batch_alternating` = new mode |
| `cache_batches` | 200 | Batches to prefill per round before switching to training |
| `cache_dir` | `/tmp/teacher_cache` | Directory for disk-cached hidden states (NVMe recommended) |

### Batch Size and Gradient Accumulation

DSpark paper uses effective batch_size=512. K3 TP=8 vLLM prefills one sequence at a time (~30s each). We set:

```yaml
train_global_batch_size: 8    # sequences per teacher prefill call (8 × 30s ≈ 4 min)
train_micro_batch_size: 1     # sequences per draft forward pass
cache_batches: 200            # steps per round (200 × 8 = 1,600 sequences cached)
cache_dir: /tmp/teacher_cache # NVMe disk cache (NOT /dev/shm — SHM capped at 100GB)
```

Effective batch_size=512 via gradient accumulation: gradients accumulated over `512 / 8 = 64` micro-steps before `optimizer.step()`. Each round produces `200 / 64 ≈ 3` optimizer updates.

Total prefill calls: `477K / 8 ≈ 59,630`. With `cache_batches=200`: `59,630 / 200 ≈ 298` rounds. vLLM startup overhead: `298 × 10 min ≈ 50 hours`.

### Disk Cache Budget

Hidden states are cached to NVMe disk (not SHM) to avoid the 100GB /dev/shm limit.

Per batch (train_global_batch_size=8, seq_len=8192):
- `hidden_states`: [8, 8192, 4×7168] × bf16 = 3.75 GB (4 aux layers, separate_last_hidden=true)
- `last_hidden_states`: [8, 8192, 7168] × bf16 = 0.94 GB
- `token_embeds`: [8, 8192, 7168] × bf16 = 0.94 GB
- `input_ids` + `attention_mask` + `loss_mask`: ~1.5 MB
- **Total per batch: ~5.6 GB**

200 batches × 5.6 GB = **~1.1 TB** on disk per round. Each round's cache is cleaned up after Phase B completes.

NVMe I/O performance: sequential read ~3-6 GB/s. Loading one batch (~5.6 GB) takes ~1-2s, well within the ~5s/step training time. Use `torch.Tensor.from_file()` or `numpy.memmap` for zero-copy reads where possible.

**Implementation**: Phase A writes each batch to `{cache_dir}/round_{r}/batch_{i}/` using the same `*.bin` + `meta.pt` format as the existing SHM slots. Phase B reads them sequentially. Each round's cache directory is cleaned up after Phase B completes.

### GPU Memory Transitions

**MI350 per-GPU budget: 288 GB HBM**

Phase A (vLLM TP=8):
- K3 model (MXFP4 MoE): ~125 GB per GPU
- KV cache (FP8): ~100-130 GB per GPU
- vLLM overhead: ~10-20 GB per GPU
- **Total: ~260 GB per GPU**

Phase B (Draft training):
- DSpark model (BF16): ~1.3 GB per GPU (replicated, not FSDP2)
- BF16Optimizer (FP32 masters + Adam m+v): ~5.2 GB per GPU
- Activations + gradients: ~1-2 GB per GPU
- **Total: ~8 GB per GPU**

Phase A → B transition:
1. `teacher_engine.shutdown()` kills vLLM subprocess → GPU VRAM freed
2. `torch.cuda.empty_cache()` + `gc.collect()`
3. Load draft model + optimizer from CPU → GPU (~8 GB)

Phase B → A transition:
1. `_offload_draft_to_cpu()`: copy model + optimizer to CPU tensors
2. `torch.cuda.empty_cache()` + `gc.collect()`
3. `teacher_engine.start()` relaunches vLLM subprocess

**CPU memory during Phase A**: Draft model + optimizer CPU copies (~50 GB total). MI350 has 1.5TB+ system RAM — well within budget.

**CPU memory during Phase B**: Draft on GPU (~8 GB/GPU). Hidden states on NVMe disk, read one batch at a time (~5.6 GB peak page cache). Well within budget.

### vLLM Startup Cost

K3 TP=8 startup is ~5-10 minutes (model loading from /dev/shm + MXFP4 quantization + KV cache allocation). With `cache_batches=200` and `train_global_batch_size=8`:

- Each round: 200 batches × ~4 min/batch ≈ 13 hours prefill + ~10 min startup
- Training: 200 steps with gradient accumulation ≈ 17 min
- vLLM overhead per round: 10 min / 13 hours ≈ 1.3%
- Total rounds: 59,630 / 200 ≈ 298 rounds
- Total vLLM startup overhead: 298 × 10 min ≈ 50 hours

**Tuning**: `cache_batches` trades disk space for startup overhead. Increase to reduce rounds; each batch uses ~5.6 GB disk (cleaned per round).

### Draft Model Wrapping: Composable Replicate (not FSDP2)

DSpark draft model is ~1.3 GB — well below the 80 GB threshold in `_can_skip_distributed_wrapping()`. The trainer automatically selects `torch.distributed.tensor.parallel.replicate()` (composable replicate) instead of FSDP2. Each GPU holds a full replica; gradients are all-reduced across ranks.

## DSpark Draft Model

### Components (trainable vs frozen)

| Component | Shape | Trainable |
|-----------|-------|-----------|
| `embed_tokens` | [163840, 7168] | Frozen (from K3) |
| `lm_head` | [163840, 7168] | Frozen (from K3) |
| `fc` | [7168, 5×7168] | Yes — 5-layer hidden state fusion |
| `hidden_norm` | [7168] | Yes — post-fusion RMSNorm |
| `layers.{0-4}` | MLA attention + FFN | Yes — 5-layer parallel backbone |
| `norm` | [7168] | Yes — final RMSNorm |
| `markov_head.markov_w1` | [163840, 256] | Yes — Markov embedding |
| `markov_head.markov_w2` | [256, 163840] | Yes — Markov projection |
| `confidence_head.proj` | [7168+256, 1] | Yes — acceptance predictor |

### Forward Pass

1. **Sample anchors**: 512 random positions per sequence from loss_mask>0 tokens
2. **Construct input**: anchor token embedding + (block_size-1) mask token embeddings
3. **Construct attention mask**: each draft query sees target context + own block prefix
4. **Fuse target hidden states**: `H_ctx = RMSNorm(fc(concat(H_l2, H_l23, H_l47, H_l71, H_l89)))`
5. **Parallel backbone forward**: 5 layers produce all 7 positions at once
6. **Markov head**: teacher-forced transition bias added to base logits
7. **Confidence head**: predict acceptance probability per position

### Loss Function

`L = 0.1 × L_ce + 0.9 × L_tv + 1.0 × L_conf`

Each weighted by position decay `w_k = exp(-k / 4.0)`:

- **L_ce**: Cross-entropy with ground-truth next tokens
- **L_tv**: Total variation distance `|softmax(draft) - softmax(target)|₁` (target logits reconstructed via frozen `lm_head(last_hidden_states)`)
- **L_conf**: BCE between predicted confidence and actual acceptance rate `1 - 0.5 × TV`

## Code Changes

### 1. `lumenrl/core/config.py` — Config additions

Add to `SpecDistillConfig`:
```python
sequential_mode: Optional[str] = None   # "batch_alternating"
cache_batches: int = 200
cache_dir: str = "/tmp/teacher_cache"
ce_loss_alpha: float = 0.1
l1_loss_alpha: float = 0.9
confidence_loss_alpha: float = 1.0
```

Add DSpark-specific fields to `DraftModelConfig`:
```python
markov_rank: int = 0
markov_head_type: str = "vanilla"
enable_confidence_head: bool = False
confidence_head_with_markov: bool = False
mask_token_id: int = 0
block_size: int = 7
```

### 2. `lumenrl/models/dspark.py` — New DSpark model

New file implementing:
- `DSparkModel` — main model class
- `VanillaMarkov` — Markov head
- `AcceptRatePredictor` — confidence head
- `compute_dspark_loss()` — CE + TV + BCE loss

### 3. `lumenrl/trainer/spec_distill_trainer.py` — Batch-alternating loop

Add:
- `_batch_alternating_train()` — outer loop with two distinct phases per round:
  - Phase A (rank 0 only): start vLLM → call `_teacher_inference_rank0()` for N batches → write each to `{cache_dir}/round_{r}/batch_{i}/` (NVMe disk) → shutdown vLLM
  - Phase B (all ranks): load draft to GPU → iterate over cached batch directories → run `_train_step_dspark()` per batch with gradient accumulation → offload draft to CPU
  - Does NOT reuse the streaming `_ShmWriterThread`/`_ShmLoaderThread` pipeline (those assume concurrent prefetch+train). Instead uses direct file I/O per batch (same `*.bin` + `meta.pt` format, written to disk).
- `_offload_draft_to_cpu()` / `_load_draft_to_gpu()` — CPU offload for GPU memory transitions
- `_train_step_dspark()` — DSpark training step (anchor sampling, Markov, confidence)
- Modify `setup()` to handle `draft_type == "dspark"` (build DSparkModel, skip FSDP2 wrapping → auto-selects replicate)
- Modify `train()` to dispatch to `_batch_alternating_train()` when `sequential_mode == "batch_alternating"`

### 4. Config files update

Update `examples/Kimi_K3_SDDD_MI350_vllm/configs/train.yaml`:
```yaml
algorithm:
  spec_distill:
    sequential_mode: batch_alternating
    cache_batches: 200
    cache_dir: /tmp/teacher_cache
    draft_type: dspark
    loss_type: dspark
  teacher:
    generate_mode: prefill
    transport: mooncake

mooncake:
  protocol: tcp
  device_name: ""
  global_segment_size: 16GB
  local_buffer_size: 4GB
```

## Testing

### Smoke test

Use `configs/smoke_test.yaml` with `cache_batches: 5`:
1. vLLM starts with TP=8 → prefills 5 batches → writes to disk cache
2. vLLM shuts down, draft model loads to GPU
3. Draft model trains 5 steps on cached hidden states
4. Verify loss decreases, no NaN/OOM

### Validation checklist

- [ ] vLLM starts and produces hidden states at correct layers via Mooncake
- [ ] Disk cache files written with correct shapes: `hidden_states [bs, T, 4*7168]`, `last_hidden_states [bs, T, 7168]`
- [ ] vLLM subprocess fully exits (no leaked GPU memory)
- [ ] Draft model loads to GPU after vLLM shutdown (composable replicate, not FSDP2)
- [ ] DSpark forward pass produces correct loss components
- [ ] Markov head teacher-forced logit bias is correct
- [ ] Confidence head produces valid probabilities
- [ ] CPU offload/reload works correctly between phases
- [ ] Checkpoint save/load across rounds works
- [ ] Multi-round alternation completes without memory leak
