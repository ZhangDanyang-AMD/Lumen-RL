---
name: lumenrl-training
description: >-
  Debugging and reviewing LumenRL RL training runs. Use when an RL run
  diverges, reward curves plateau or collapse, MoE routing becomes
  unstable, FP8 precision causes train-inference mismatch, or KL
  divergence spikes unexpectedly.
---

# LumenRL Training Guide

## Core Principle

When an RL run behaves unexpectedly, compare against a known-good BF16 baseline before changing any hyperparameters. Use `.cursor/tmp-rl-bugs.md` in the Lumen-RL repo root as working memory. Do not solve stability problems by retuning parameters that already match the baseline.

## Use When

- Reward curve plateaus, collapses, or shows unexpected behavior
- KL divergence spikes (especially with FP8 or MoE)
- MoE router distributions diverge between train and inference
- FP8 rollout produces different behavior than BF16 rollout
- Training loss shows NaN or Inf
- Weight sync appears to corrupt model outputs
- Expert load balance degrades during training

Treat any fresh return to the same debugging problem as a new debug session.

## Default Workflow

1. Read `.cursor/tmp-rl-bugs.md` at the start of every new debug session
2. Stop the run once behavior is clearly wrong; save logs and the last good checkpoint
3. Compare current config against the baseline recipe (YAML diff, not memory)
4. Check reward / KL / entropy curves for the first anomalous step
5. For FP8 issues: compare BF16 vs FP8 rollout log-probs; check if rollout correction (TIS/MIS) is enabled
6. For MoE issues: check if R3 is enabled; compare router distribution entropy between train and inference
7. For weight sync issues: compare model outputs before and after weight transfer on the same input
8. After each meaningful test, update `.cursor/tmp-rl-bugs.md` with findings

## Hard Rules

- Stop the run when reward collapses. Debug from the last good checkpoint.
- Start every new debug session by reading `.cursor/tmp-rl-bugs.md`
- Never disable R3 "to see if it helps" without first recording the routing distributions
- Never change clip ratio, KL coefficient, or learning rate if they already match the baseline
- FP8 precision issues are not hyperparameter problems; isolate the quantization path first
- Compare effective runtime values, not just config files
- Do not continue a clearly bad run "for more signal"

## FP8-Specific Debug

| Symptom | Check |
|---------|-------|
| Reward diverges from BF16 baseline | Compare per-token log-prob distributions (FP8 vs BF16) |
| Loss spikes after weight sync | Verify QKV scale recalibration runs every RL step |
| Gradual quality drift | Check if `rollout_correction.enabled: true` and try `num_first_layers_in_bf16: 1` |
| NaN in rollout | Check FP8 weight quantizer for overflow; verify blockwise scaling ranges |

Escalation order for FP8:
1. Compare BF16 vs FP8 rollout output on same input, same weights
2. If different: check weight quantizer round-trip error per layer
3. If quantizer is fine: check KV-cache scale recalibration
4. If KV-cache is fine: check attention output (FP8 attention vs BF16)
5. Enable rollout correction (TIS) and verify it reduces the gap

## R3-Specific Debug

| Symptom | Check |
|---------|-------|
| MoE training collapse (reward -> 0) | Verify R3 is enabled; check `moe.r3.enabled: true` |
| KL divergence spikes with MoE | Compare router logits between ATOM rollout and Lumen training |
| Expert load imbalance grows | Check if R3 replay preserves expert selection distribution |
| R3 enabled but still unstable | Verify recorded logits are non-NaN, correct shape, correct layer count |

Escalation order for R3:
1. Verify router logits are recorded correctly (shape, range, no NaN)
2. Diff recorded logits vs replayed logits (must be bit-identical)
3. Check expert load balance before and after R3 (should be similar)
4. If load balance differs: check if EP layout differs between ATOM and Megatron
5. If still unstable: check if replay mode should be `hard_assignment` instead of `distribution`

## Weight Sync Debug

| Symptom | Check |
|---------|-------|
| Rollout ignores training updates | Verify `weight_sync.transfer()` is called after `train_step()` |
| Model outputs differ after sync | Compare state_dict checksums before send and after receive |
| OOM during sync | Check if FP8 on-the-fly conversion is enabled (avoids double-copy) |
| MoE experts scrambled after sync | Check EP layout resharding between Megatron and ATOM |

## Rationalizations to Reject

| Excuse | Reality |
|--------|---------|
| "Just tweak the clip ratio" | If it matches the baseline, the bug is elsewhere |
| "Disable R3, it's probably the problem" | R3 fixes instability; disabling it hides the root cause |
| "The FP8 gap is acceptable" | Quantify it. If `>5%` reward difference at convergence, investigate |
| "Keep running for more signal" | Sunk cost. Stop the bad run and debug from a snapshot |
| "The configs are basically the same" | "Basically" is not enough. Diff effective values |

## Red Flags

- Starting a debug session without reading `.cursor/tmp-rl-bugs.md`
- Changing aligned hyperparameters to chase a symptom
- Running with R3 disabled on MoE models without documenting why
- Claiming FP8 parity from memory instead of a written comparison
- Ignoring weight sync step when debugging output quality
- Comparing runs with different seeds, datasets, or tokenizers

## References

- `.cursor/tmp-rl-bugs.md` in the Lumen-RL repo root — open bug candidates and evidence
- [reference.md](reference.md) — pre-training diff checklist, FP8 debug checklist, R3 debug checklist, weight sync debug checklist, reward debug checklist
- `configs/grpo_dense_bf16.yaml` — BF16 baseline recipe (the primary reference for parity testing)

## Pairing

Pair with `lumenrl-coding` when code changes are needed.
Pair with `lumenrl-debug` when the bug originates in Lumen, ATOM, or AITER rather than LumenRL itself.
