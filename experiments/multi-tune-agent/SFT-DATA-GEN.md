# SFT Data Generation Pipeline

End-to-end guide for generating training data for Qwen3-Coder kernel optimization SFT. The pipeline spans two repos: **multi-tune-agent** (collection + assembly + export) and **GEAK-agent-coder** (final tokenization + sampling).

---

## Prerequisites

- 8x AMD MI308X GPUs with ROCm 7.0
- Docker container with vLLM 0.15.0+rocm700
- GEAK repo at `/home/danyzhan/GEAK`
- AITER repo at `/home/danyzhan/Lumen/third_party/aiter`
- A serving model (e.g., Qwen/Qwen3-Coder-30B-A3B-Instruct) on vLLM

```bash
pip install -e /home/danyzhan/Lumen-RL/experiments/multi-tune-agent
pip install -e /home/danyzhan/Lumen/experiments/GEAK-agent-coder
```

---

## Phase 0: Build Lineage Splits

Assign source lineages to train/dev/held_out splits **before any collection**. The unit of assignment is `source_lineage_id`, not individual tasks, to prevent leakage.

```bash
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent

python scripts/build_phase1_splits.py \
    --catalog cases/phase1-case-candidates-gfx942-expanded.yaml \
    --output-dir /home/danyzhan/phase1_control/splits/v4 \
    --prior-split-dir /home/danyzhan/phase1_control/splits/v3 \
    --salt geak-phase1-gfx942-v4 \
    --dev-fraction 0.15 \
    --held-out-fraction 0.25
```

**Output:** `train_groups.jsonl`, `dev_groups.jsonl`, `held_out_groups.jsonl`, `generation-requests.yaml`, `manifest.json`

**Key properties:**
- Stable hashing: `sha256(salt + "\0" + lineage_id)` maps to fractional ranges
- Append-safe: adding new lineages never changes existing assignments
- Held-out lineages are never collected (enforced at runtime)

---

## Phase 1: Collect SFT Data During Agent Runs

### 1.1 Configure Collection

In the multi-tune config (e.g., `configs/mi300x.yaml`), enable SFT:

```yaml
sft_enabled: true
sft_dataset_root: /home/danyzhan/geak_sft_dataset/phase1
sft_task_type: direction_conditioned   # or: cold_start, profile_guided, error_recovery, regression_balance
```

### 1.2 Serve the Model

```bash
HIP_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --port 8000 --tensor-parallel-size 1 --max-model-len 32000 \
    --gpu-memory-utilization 0.90 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --enforce-eager --trust-remote-code
```

### 1.3 Run Collection

**Interactive (single case):**

```bash
multi-tune --config configs/mi300x.yaml interactive
```

**Batch (full catalog):**

```bash
multi-tune --config configs/mi300x.yaml run --stream
```

**Production controller (quota-balanced across task types and lanes):**

```bash
python scripts/run_phase1_production_controller.py \
    --config configs/mi300x.yaml \
    --wave-config configs/phase1-top10-wave-2000.yaml \
    --split-manifest /home/danyzhan/phase1_control/splits/v4/manifest.json
```

**Parallel generation (7 GPUs for kernel eval, GPU 0 for model):**

```bash
python scripts/run_parallel_generation.py \
    --config configs/mi300x.yaml \
    --manifest cases/phase1-generation-requests.yaml \
    --output-catalog cases/phase1-cases.yaml
```

### 1.4 What Gets Collected

Per run, the `SFTCollector` writes to `{sft_dataset_root}/`:

| File | Content |
|------|---------|
| `runs/{case}_{ts}/sft_manifest.json` | Eligibility status, candidate counts |
| `runs/{case}_{ts}/environment.json` | ROCm version, GPU SKU, git SHAs, container digest |
| `runs/{case}_{ts}/round_N/frozen_input.json` | Contract, parent source, baseline, profile |
| `runs/{case}_{ts}/round_N/plan.json` | Tech lead's optimization directions (direction_conditioned only) |
| `runs/{case}_{ts}/round_N/candidates.jsonl` | Each candidate with diff patch, verification receipts |
| `blobs/sha256/{2ch}/{hash}` | Content-addressed source files, patches, configs |

**Eligibility criteria for a positive SFT sample:**
- Patch applies cleanly to parent source
- Independent verification passed (compile + correctness + performance)
- Parent source is frozen and available as blob
- Candidate was accepted by the multi-tune flow
- Role is "engineer"

### 1.5 Five SFT Task Types

| Type | Input | What the model learns |
|------|-------|----------------------|
| `cold_start` | Contract only (no parent, no profile) | Generate kernel from scratch |
| `profile_guided` | Contract + parent source + profiling data | Optimize based on profiler output |
| `direction_conditioned` | Contract + parent + tech lead direction | Follow specific optimization strategy |
| `error_recovery` | Contract + parent + error feedback | Fix compile/correctness errors |
| `regression_balance` | Contract + parent + regression constraints | Improve without breaking existing cases |

### 1.6 Wave Quotas

Production waves enforce balanced distribution:

```yaml
# configs/phase1-top10-wave-2000.yaml
total: 2000
lane_quotas:
  triton_gfx942: 1000
  hip_gfx942: 1000
mode_quotas:
  cold_start: 307
  profile_guided: 293
  direction_conditioned: 907
  error_recovery: 293
  regression_balance: 200
```

---

## Phase 2: Assemble Dataset from Raw Runs

### 2.1 Create Input Manifest

Create a JSON manifest listing the run directories to include:

```json
{
    "schema_version": "geak_sft_input_manifest_v1",
    "runs": [
        {"path": "/home/danyzhan/geak_sft_dataset/phase1/runs/case1_20260901_120000"},
        {"path": "/home/danyzhan/geak_sft_dataset/phase1/runs/case2_20260901_130000"}
    ]
}
```

### 2.2 Build Dataset

```bash
cd /home/danyzhan/Lumen-RL/experiments/multi-tune-agent

python scripts/build_sft_dataset.py \
    --input-manifest /home/danyzhan/geak_sft_dataset/phase1/input_manifest.json \
    --output-root /home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1
```

**What this does:**
1. Loads each run's `sft_manifest.json` and `candidates.jsonl`
2. Validates every candidate: schema version, patch replay (applies unified diff to parent, verifies exact child match), protected path check, verification receipts, contract provenance, environment provenance, architecture consistency
3. Deduplicates by (patch hash, normalized text hash), keeping highest-priority version
4. Writes `processed/train.jsonl`, `processed/dev.jsonl`, `processed/held_out.jsonl`, `processed/rejected.jsonl`
5. Produces `dataset_manifest.json` with SHA256 checksums

### 2.3 Validate and Audit

```bash
# Re-validate every sample (patch replay, source hashes, quality gates)
python scripts/validate_sft_dataset.py \
    --manifest /home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/dataset_manifest.json

# Check for split leakage across 4 dimensions
python scripts/audit_sft_leakage.py \
    --manifest /home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/dataset_manifest.json

# Measure coverage by operator, lane, task type, etc.
python scripts/report_sft_coverage.py \
    --manifest /home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/dataset_manifest.json
```

---

## Phase 3: Export to Qwen Training Format

```bash
python scripts/export_qwen_sft.py \
    --source-manifest /home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/dataset_manifest.json \
    --output-root /home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/export \
    --tokenizer-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --tokenizer-revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
    --expected-chat-template-sha256 <sha256> \
    --max-length 32768 \
    --split train \
    --overflow-policy quarantine
```

**What this does:**
1. Builds 3-message ChatML (system + user + assistant) per sample
2. System prompt: "You optimize AMD GPU kernels from a frozen, validated GEAK input..."
3. User message: structured prompt with `FROZEN_INPUT=<JSON>`
4. Assistant message: the verbatim unified diff patch
5. Tokenizes and computes content-only loss mask (only assistant patch tokens get loss=1)
6. Writes `train.qwen.jsonl` with `input_ids`, `attention_mask`, `loss_mask`

---

## Phase 4: Final Data Build (GEAK-agent-coder)

This phase combines kernel SFT data with general-coding replay samples for retention.

### 4.1 Configure Data Sources

Edit `configs/data/qwen3_30b_a3b_phase1.yaml`:

```yaml
schema_version: geak_data_config_v2

tokenizer:
  path: ../../models/Qwen3-30B-A3B-tokenizer
  sha256: "4dbc3bb2..."

tokenization:
  model_family: Qwen/Qwen3-30B-A3B
  max_length: 32768
  include_assistant_eot_in_loss: true
  overflow_policy: quarantine

sampling:
  steps: 1000
  length_penalty_power: 0.5
  seed: 0
  replay_mix:
    enabled: true
    min_assistant_loss_token_share: 0.15
    max_assistant_loss_token_share: 0.20

sources:
  - name: kernel_train
    type: local_jsonl
    path: /home/danyzhan/geak_sft_dataset/phase1-production-wave-2000-v1/processed/train.jsonl
    admission:
      type: generic
      allowed_splits: [train]
      require_checksum: true
      expected_sha256: "<sha256>"

  - name: kernel_dev
    type: local_jsonl
    path: /home/danyzhan/geak_sft_dataset/phase1-dev-wave-200-v1/processed/dev.jsonl

  - name: replay
    type: local_jsonl
    path: /home/danyzhan/phase1_control/general-coding-replay-v1/accepted.jsonl
    admission:
      type: geak_replay_release
      package_root: /home/danyzhan/phase1_control/general-coding-replay-v1
      accepted_rows: 500

output:
  directory: ../../data/build/qwen3_30b_a3b_phase1
```

### 4.2 Build

```bash
cd /home/danyzhan/Lumen
geak-agent-coder data-build --config experiments/GEAK-agent-coder/configs/data/qwen3_30b_a3b_phase1.yaml
```

**What this does:**
1. **Admission:** Validates SHA256 checksums, splits, required provenance fields
2. **Mapping:** Translates source fields to canonical schema via declarative DSL
3. **Tokenization:** Builds Qwen ChatML messages, computes loss masks
4. **Sampling:** Draws 1000 training samples with length-penalized stratified sampling; enforces 15-20% general-coding replay token share
5. **Output:** `train.tokenized.jsonl` (1000 rows), `dev.tokenized.jsonl` (200 rows), `manifest.json`

### 4.3 Verify

The manifest file contains:
- SHA256 of all artifacts
- Sampling indices for exact reproduction
- Token statistics
- Source provenance chain

---

## Quick Reference: Full Pipeline Commands

```bash
# 0. Splits
python scripts/build_phase1_splits.py --catalog candidates.yaml --output-dir splits/ --salt v4

# 1. Collect (with model serving on GPU 0)
multi-tune --config configs/mi300x.yaml run --stream

# 2. Assemble
python scripts/build_sft_dataset.py --input-manifest manifest.json
python scripts/validate_sft_dataset.py --manifest dataset_manifest.json
python scripts/audit_sft_leakage.py --manifest dataset_manifest.json

# 3. Export
python scripts/export_qwen_sft.py --source-manifest dataset_manifest.json --output-root export/ \
    --tokenizer-model Qwen/Qwen3-Coder-30B-A3B-Instruct --split train

# 4. Final build
geak-agent-coder data-build --config configs/data/qwen3_30b_a3b_phase1.yaml

# 5. Train
torchrun --nproc-per-node=8 -m geak_agent_coder.sft.train --config configs/sft/qwen3_coder_12k_production.yaml
```
