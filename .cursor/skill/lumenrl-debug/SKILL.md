---
name: lumenrl-debug
description: >-
  Debugging upstream library bugs in LumenRL. Use when a bug originates in
  Lumen, ATOM, or AITER code rather than LumenRL itself. Guides where to
  apply fixes and how to propagate them back to the correct third_party
  submodule or upstream repo.
---

# LumenRL Upstream Debug Guide

## Core Principle

LumenRL depends on three upstream libraries. When a bug is found during RL
training or inference, first identify which library owns the buggy code, then
fix it in the correct location:

| Bug origin | Fix location | How to propagate |
|------------|-------------|------------------|
| **Lumen** (training ops, FP8 quantize, models) | `third_party/Lumen/` | Edit in the submodule, commit there, push to `dev/RL` branch |
| **ATOM** (inference engine, rollout) | `third_party/ATOM/` | Edit in the submodule, commit there, push to the ATOM repo |
| **AITER** (GPU kernels: attention, GEMM, MoE, norm) | AITER submodule (`third_party/Lumen/third_party/aiter/` or top-level) | Edit in the AITER repo, commit, push to `lumen/triton_kernels` branch |
| **MORI** (communication, collectives, expert dispatch) | MORI submodule | Edit in the MORI repo, commit, push to `sdma-new` branch |
| **LumenRL itself** (controller, workers, algorithms) | `lumenrl/` | Normal commit in the Lumen-RL repo |

**Never** patch upstream bugs directly in `lumenrl/` with workarounds. Always fix at the source.

## Identifying Bug Origin

Use this decision tree:

1. **Stack trace contains `lumen.quantize`, `lumen.models`, `lumen.ops`** → Lumen bug → fix in `third_party/Lumen/`
2. **Stack trace contains `atom.`, `atom.model_engine`, `atom.model_ops`** → ATOM bug → fix in `third_party/ATOM/`
3. **Stack trace contains `aiter.`, `aiter_fused_`, or ASM/CK kernel names** → AITER bug → fix in AITER submodule
4. **Stack trace contains `mori.`, `mori.ccl`, `mori.ep`** → MORI bug → fix in MORI submodule
5. **Stack trace only contains `lumenrl/`** → LumenRL bug → fix in `lumenrl/`

For ambiguous cases (e.g., wrong config passed from LumenRL to Lumen), fix the caller in `lumenrl/`.

## Workflow

### Step 1: Reproduce and log

```bash
# Document in .cursor/tmp-rl-bugs.md
echo "## [$(date)] Bug: <title>" >> .cursor/tmp-rl-bugs.md
echo "- Symptom: ..." >> .cursor/tmp-rl-bugs.md
echo "- Stack trace: ..." >> .cursor/tmp-rl-bugs.md
echo "- Origin: Lumen | ATOM | AITER | MORI | LumenRL" >> .cursor/tmp-rl-bugs.md
```

### Step 2: Fix in the correct location

**For Lumen bugs:**
```bash
cd third_party/Lumen
# make the fix
git add -A && git commit -m "fix: <description>"
git push origin dev/RL
cd ../..
git add third_party/Lumen
git commit -m "bump Lumen: fix <description>"
```

**For ATOM bugs:**
```bash
cd third_party/ATOM
# make the fix
git add -A && git commit -m "fix: <description>"
git push origin <branch>
cd ../..
git add third_party/ATOM
git commit -m "bump ATOM: fix <description>"
```

**For AITER bugs:**
```bash
# AITER lives at: https://github.com/ZhangDanyang-AMD/aiter.git (lumen/triton_kernels branch)
cd <aiter-submodule-path>
# make the fix
git add -A && git commit -m "fix: <description>"
git push origin lumen/triton_kernels
cd <back-to-lumenrl>
git add <aiter-submodule-path>
git commit -m "bump AITER: fix <description>"
```

**For MORI bugs:**
```bash
# MORI lives at: https://github.com/ZhangDanyang-AMD/mori.git (sdma-new branch)
cd <mori-submodule-path>
# make the fix
git add -A && git commit -m "fix: <description>"
git push origin sdma-new
cd <back-to-lumenrl>
git add <mori-submodule-path>
git commit -m "bump MORI: fix <description>"
```

### Step 3: Verify

After fixing, re-run the failing test or training step to confirm the fix.
Update `.cursor/tmp-rl-bugs.md` with the resolution.

## Hard Rules

- Never monkey-patch upstream code from `lumenrl/` (e.g., `lumen.quantize.some_fn = patched_fn`)
- Never copy upstream files into `lumenrl/` to work around a bug
- Always commit the fix in the upstream submodule first, then bump the submodule ref in Lumen-RL
- If a fix requires coordinated changes across multiple repos (e.g., Lumen API change + LumenRL caller update), commit the upstream change first, then the LumenRL adaptation
- Document every upstream bug and fix in `.cursor/tmp-rl-bugs.md`

## Common Upstream Bug Patterns

| Pattern | Likely origin |
|---------|--------------|
| FP8 quantization produces NaN/Inf | Lumen (`lumen/quantize/`) |
| KV-cache scale mismatch after weight sync | ATOM (`atom/model_engine/`) |
| Triton kernel launch failure or wrong result | AITER |
| NCCL/RCCL hang or collective mismatch | MORI |
| ROCm HIP API error (`hipMem*`, `HSA_STATUS_*`) | AITER or ROCm driver (file upstream issue) |
| Model output diverges between train and inference | Check both Lumen and ATOM; likely weight format mismatch |

## Pairing

Pair with `lumenrl-coding` for code changes in `lumenrl/`.
Pair with `lumenrl-training` for RL training stability issues.
