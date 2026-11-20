# Whole-network optimization of SimWAM NAVSIM FlowGRPO on 4× MI350X

> Performance, accuracy and reproduction report. 中文版：[README_PERF.md](README_PERF.md). Every switch is described in full in [README.md](README.md).

This report sums up the whole-network optimization of the FlowGRPO training in `examples/Embodied_SimWAM_NAVSIM` on 4 AMD Instinct MI350X (gfx950):
how much faster it is than the bit-identical baseline, where the time went, what each optimization contributed as measured, whether accuracy holds,
and how to reproduce it or go back to the bit-identical path. SimWAM, DeepSpeed and Lumen are not modified; every change is a runtime patch in this
directory's `plugin.py` behind an environment switch.

## Overview (4× MI350X)

| Version | Mean cycle | Mean rollout | PDMS | NC / DAC / EP / TTC / C / DDC |
|---|---:|---:|---:|---|
| torch bf16 benchmark | 1383.5 ms | 398.2 ms | 91.13 | 98.36 / 98.64 / 85.75 / 95.42 / 99.98 / 98.17 |
| final version | 371.2 ms | 204.4 ms | 91.11 | 98.36 / 98.65 / 85.80 / 95.28 / 99.99 / 98.25 |
| difference | −73% (3.73×) | −49% (1.95×) | −0.02 | 0.00 / +0.01 / +0.05 / −0.14 / +0.01 / +0.08 |

- Times: 4 GPUs, rank 0, mean of the 11 cycles with `step_start` 16–56 (steps 17–60, no eval). The benchmark is the bit-identical DeepSpeed ZeRO-1
  path `f27ab82`: 1383.5 ms is the mean of the 22 cycles of two 60-step runs (1422.8 / 1344.3 ms each). The final version's 371.2 ms is steps 17–60 of
  FIN20 itself (`daf60b17`, `max_steps=20000`). Both means are pulled up by a few cycles that wait for the PDM scores (medians of the same window
  1317.0 vs 303.3 ms, see 1.3).
- PDMS: full navtest (12146 tokens), checkpoint after 20000 training steps. The benchmark's PDMS is that of BX20, the bit-identical run that
  reproduces it bit for bit; its sub-scores are derived from the differences (±0.01).
- Over the whole 20000 steps (eval, visualisation and PDM waits included): final version mean cycle 522.9 ms, total 43.6 min; BX20 1054.1 ms, 87.8 min (section 1.1).
- The mean PDMS does not differ (paired per token −0.02, p = 0.75), but the gate flips lean towards worse, mainly in TTC (section 3.3).

---

## 1. Summary

### 1.1 Result

> **A 20000-step training run takes 43.6 instead of 87.8 minutes; the mean cycle goes from 1054.1 to 522.9 ms (2.02×, −50%).
> Full-navtest PDMS is 91.11 against 91.13 for the bit-identical baseline, a paired difference of −0.02 (95% CI −0.15 to +0.11, p = 0.75):
> no difference in the mean. The rollout's and the training's log-probs are bit-identical in all 5000 cycles × 4 ranks.**
> Known difference: the gate flips lean towards worse, mainly in TTC (section 3.3).

| Level | Result | Evidence |
|---|---|---|
| Whole network | **2.02× faster** | 20000 steps: mean cycle 1054.1 → 522.9 ms; total 87.8 → 43.6 min (same GPUs, seed and configuration) |
| GPU phases | **2.78× faster** | sum of the per-phase medians 816.1 → 293.2 ms |
| Accuracy | **same mean, gate flips lean to TTC** | full-navtest PDMS 91.11 vs 91.13 (paired −0.02, p = 0.75); 217 tokens flip a gate, 79 better / 124 worse (p = 0.002), TTC 46 / 63 |
| Train / rollout consistency | **bit-identical** | \|Δlogp\| = 0 (baseline: mean 1.16e-3, max 0.032, 4.6% outside the PPO clip) |
| Memory | **lower** | peak allocated per GPU 35.62 → 34.77 GiB, device memory in use 53.36 → 42.90 GiB |

| | bit-identical baseline BX20 | optimized FIN20 | change |
|---|---:|---:|---:|
| Code | #51 `f3f3ca5`, `LUMEN_ENABLE=hf_attn_patch` | `hz/grpo-fast` `daf60b17`, all defaults | |
| Mean cycle (all 5000 cycles, incl. eval and visualisation) | 1054.1 ms | **522.9 ms** | **−50.4% (2.02×)** |
| First to last cycle (20000 steps) | 87.8 min | **43.6 min** | **−50.3%** |
| Median cycle (all cycles, rank 0) | 814.1 ms | 398.9 ms | −51.0% (2.04×) |
| Non-eval cycles, 10th / 90th percentile (rank 0) | 804.1 / 912.8 ms | 293.4 / 664.9 ms | |
| Sum of GPU-phase medians | 816.1 ms | 293.2 ms | −64.1% (2.78×) |
| Full-navtest PDMS (12146 tokens) | 91.13 | 91.11 | −0.02 (p = 0.75) |
| \|Δlogp\| (rollout vs training) | mean 1.16e-3, max 0.032 | 0 | |

![cycle time](https://raw.githubusercontent.com/huizzhan/Lumen-RL/dac70f5d6fa9cc037709479c7212460dd87b3500/assets/grpo-fin20-vs-bx20-cycle-time.png)

### 1.2 Why the GPU phases are 2.78× faster but the network only 2.02×

A cycle is 1 rollout (VAE, video prefill, 10 denoising steps, old / reference log-probs, PDM scoring) plus 4 optimizer steps.
The GPU phases drop from 816.1 to 293.2 ms (medians), but a cycle also has time outside the GPU:

- **The tail of the PDM scoring.** Rewards come from PDM closed-loop simulation on the CPU (8 worker processes per rank); a few tokens take
  hundreds of milliseconds to load their metric cache or to simulate, which no GPU optimization changes. FIN20's non-eval cycles have a median of
  387.9 ms and a 90th percentile of 664.9 ms; the gap is mostly this wait.
- **The built-in eval and visualisation every 100 steps**, in both runs, counted in the mean cycle.
- The main figures are therefore the **mean cycle and total time over 20000 steps** (which include both); the median and the GPU phases are supplementary.

![phases](https://raw.githubusercontent.com/huizzhan/Lumen-RL/dac70f5d6fa9cc037709479c7212460dd87b3500/assets/grpo-fin20-vs-bx20-phases.png)

| Phase (median, ms, rank 0, from step 16, without eval / visualisation cycles) | BX20 | FIN20 | Main sources |
|---|---:|---:|---|
| VAE encode | 21.5 | 20.5 | FlyDSL convolutions (one encode −22%, not visible end to end) |
| condition (video prefill) | 73.2 | 31.7 | prefill CUDA graph, dropped masks, CK RMSNorm |
| denoising (10 steps) | 173.8 | 61.6 | XATTN_KV, LORA_CAT, fused kernels, KV slots |
| rollout log-probs (policy / ref) | 53.7 / 41.4 | 12.9 / 9.4 | batched no-grad calls, batched GEMMs, KV slots |
| training forward ×4 | 189.8 | 63.6 | batched GEMMs, fused kernels, KV slots in the training graphs |
| training backward ×4 | 248.8 | 78.8 | Triton RMSNorm backward, aiter attention backward, gradients written into the flat buffer |
| optimizer ×4 | 13.9 | 14.7 | |

Rollout sub-phases under the overview table's method (steps 17–60): denoising 2.77×, condition build 2.40×, policy log-probs 4.13×, reference log-probs
4.33×, VAE 1.05×. That window covers only the first 60 steps, where the PDM tail and the share of the eval differ from the whole 20000-step run, so the
main whole-network figures remain the 20000-step mean cycle and total time.

### 1.3 Against the DeepSpeed baseline (supplementary)

The original example trains with DeepSpeed ZeRO-1. #51's third round measured the bit-identical DeepSpeed path (`f27ab82`) on the same 4 GPUs
(2026-11-18, GPUs 4–7, `LUMEN_ENABLE=hf_attn_patch`, two runs of 60 steps). Over the same window as FIN20 (rank 0, `step_start` 16–56, i.e. steps 17–60, no eval):

| Steps 17–60 | DeepSpeed ZeRO-1 (`f27ab82`, 22 cycles of two runs) | FIN20 (`daf60b17`, 11 cycles) | ratio |
|---|---:|---:|---:|
| Mean cycle | 1383.5 ms (1422.8 / 1344.3 per run) | 371.2 ms | **3.73×** |
| Median cycle | 1317.0 ms (1328.0 / 1305.3 per run) | 303.3 ms | 4.34× |

Limits:
- Both means are pulled up by a few cycles that wait for the PDM scores, which is why the mean ratio (3.73×) is below the median ratio (4.34×).
- The DeepSpeed runs are separate 60-step runs while FIN20's window is the start of a 20000-step run; they were not interleaved in one group, and
  there is no 20000-step mean or total time for DeepSpeed.
- This section is supplementary; the main comparison is the like-for-like one against BX20 over 20000 steps in 1.1. The original example `d0ada15`
  (DeepSpeed, `hf_attn_patch,lumen_linear`) was measured once at the time, 6076 ms (its second run was interrupted by another process).

### 1.4 Scope

- One seed (42) and one 20000-step run per side; 4000-step subset PDMS of single-seed runs varies by about ±0.15 (section 3.2).
- FIN20 ran `daf60b17`. `62acd4c5` adds two switches on top, on by default (`SIMWAM_GRPO_EXPAND_VIEW`, `SIMWAM_GRPO_KV_SLOT_STEP_SHARED`); both are
  **bit-identical** to `daf60b17` (2 GPUs, three interleaved runs each: identical losses in all 9 runs; all defaults for 24 steps with every graph
  replay checked against eager: no mismatch). On 2 GPUs they save another 1.8–6.0 ms of GPU phases per cycle and 1.45 GB of peak allocated memory per
  GPU; they were not run for 20000 steps.
- The timings come from a measurement hook (`torch.cuda.synchronize()` at phase boundaries, about 1% overhead) that is not part of this repository;
  both the baseline and the optimized run carry it.

---

## 2. Test design

### 2.1 Environment

| Item | Value |
|---|---|
| GPU | 4× AMD Instinct MI350X (gfx950), cv350 GPUs 4–7, `NPROC=4`, one process per GPU |
| Container | `zhangdanyangamd/lumen-rl:embodied-MI350X` (torch `2.12.0+rocm7.14.1`, the image's aiter) |
| Dependencies | SimWAM `1433ee7` (unmodified); Lumen `293629f` (`dev/world-model-rl`); aiter checkout `huizzhan/aiter:hz/grpo-opt` `c9be3a5` (`AITER_SRC`, with conv3d and this example's GEMM tuning table) |
| Task | `navsim_grpo_action_pdm_384x672_flowgrpo_lora`, scene filter `configs/navtrain_pdm_score_below0p9_complete.yaml` (2784 navtrain scenes), warm start `SimWAM.pt` |
| Key hyperparameters | `max_steps=20000`, `batch_size=1` per rank, `group_size=8`, `num_inner_epochs=4`, lr 5e-5 (100 warmup steps), LoRA r=16 / alpha=32, seed 42, built-in eval and visualisation every 100 steps |

### 2.2 The two runs

| Run | Code | `LUMEN_ENABLE` | Note |
|---|---|---|---|
| BX20 | #51 `f3f3ca5` | `hf_attn_patch` | bit-identical baseline: identical to the original example `d0ada15` over 8 steps on 4 GPUs; its first 4000 steps identical to another run, C51b |
| FIN20 | `hz/grpo-fast` `daf60b17` | script default `hf_attn_patch,aiter_conv,flydsl_conv` | all switch defaults, without the new GEMM tuning table |

Both ran 20000 steps on the same GPUs (4–7), container, seed and configuration, with no other process on the GPUs (sampled every second).

### 2.3 Measurement

- **Mean cycle**: mean of the cycle intervals over all 5000 cycles, including eval, visualisation and PDM waits (recomputed on rank 0 as first cycle
  start to last cycle end ÷ 5000: 523.3 / 1054.5 ms, within 0.1%); **total time**: first cycle start to last cycle end (container start and model
  loading excluded).
- **Median, percentiles and phases**: per-cycle times of rank 0; phases are medians over the cycles from step 16 without eval / visualisation.
- **Accuracy**: the step-20000 LoRA merged into `SimWAM.pt` as `SimWAMGRPO.save_checkpoint` does, scored with NAVSIM v1 PDMS on the full navtest
  (12146 tokens); paired per token (bootstrap 95% CI, paired t-test, exact sign test on gate flips).
- **Train / rollout consistency**: at the first inner update of every cycle (weights still those that produced the rollout), the training forward's
  policy log-probs against the rollout's `old_logp` (24 per rank per cycle).
- **Per-optimization gains** were measured when each was added (2 or 4 GPUs, 40 or 60 steps, interleaved on the same GPUs) and do not add up; see section 4.

---

## 3. Results

### 3.1 Speed

See the table and figures in 1.1. In addition: the median cycles of FIN20's 4 ranks are within 1 ms of each other, as for BX20; neither run yielded or was interrupted.

### 3.2 Accuracy: full-navtest PDMS (step 20000)

| Metric | FIN20 (optimized) | BX20 (bit-identical baseline) | FIN20 − BX20 | `SimWAM.pt` (before RL, bf16 benchmark) |
|---|---:|---:|---:|---:|
| **PDMS** | **91.11** | **91.13** | **−0.02** (95% CI −0.15 to +0.11, p = 0.75) | 90.27 |
| NC (no at-fault collision) | 98.36 | 98.36 | 0.00 | 98.65 |
| DAC (drivable area) | 98.65 | 98.64 | +0.01 | 98.04 |
| EP (ego progress) | 85.80 | 85.75 | +0.05 | 84.02 |
| TTC (time to collision) | 95.28 | 95.42 | −0.14 | 95.78 |
| C (comfort) | 99.99 | 99.98 | +0.01 | 99.99 |
| DDC (driving direction, weight 0 in v1) | 98.25 | 98.17 | +0.08 | 98.51 |

- No difference in the mean; both are about 0.85 above `SimWAM.pt` before RL.
- 4000-step check (FIN4, `1869ff4a`, the first half of navtest, 6074 tokens): 91.15, paired +0.07 against the bit-identical C51b's 91.07 (CI −0.08 to +0.23).
  Single-seed 4000-step runs of the same recipe across code versions span 91.01–91.26, the run-to-run spread of RL training itself.

### 3.3 Known difference: gate flips lean towards worse, mainly TTC

217 tokens have at least one gate that differs between FIN20 and BX20: FIN20 is better on 79 and worse on 124 (exact sign test p = 0.002), mostly
through TTC (FIN20 better / worse 46 / 63). The mean PDMS does not differ (−0.02) because those losses are offset by small gains (EP and others) on the remaining tokens.

What is established: this is a real difference between the FIN20 and BX20 training trajectories, not scoring noise. What is not: whether it is a
systematic effect of one of the value-changing switches or single-seed variation of the RL trajectory. A 4000-step ablation saw the same pattern
before: a `bc8cd91e` run with low TTC, which disappeared when either LORA_CAT or FUSED_MODULATE was turned off, which points to that one trajectory.
Separating the two needs repeated seeds, which were not run.

### 3.4 Train / rollout consistency

| Run | mean \|Δlogp\| | max \|Δlogp\| | outside the PPO clip (0.005) |
|---|---:|---:|---:|
| BX20 | 1.16e-3 | 0.0324 | 4.60% |
| FIN20 | **0** | **0** | **0%** |

The baseline's gap comes from the original example: rollout attention runs Lumen's Triton kernel (`hf_attn_patch`), the training forward falls back to
torch SDPA. The optimized path runs the rollout and the training on the same attention and GEMM kernels (dropped masks, batched no-grad calls, batched
GEMMs and so on), so the importance ratio is exactly 1 before the first update.

### 3.5 Memory (peak per GPU)

| Metric | BX20 | FIN20 | `62acd4c5` on top (2 GPUs, measured) |
|---|---:|---:|---:|
| `max_memory_allocated` (rank 0 / others) | 35.62 / 35.45 GiB | 34.77 / 34.46 GiB | −1.45 GB |
| `max_memory_reserved` | 44.39 / 44.34 GiB | 39.61 / 39.54 GiB | −2.2 GB |
| Device memory in use (`mem_get_info`, max) | 53.36 GiB | 42.90 GiB | not measured the same way |

Both peaks are reached during the built-in eval. An MI350X has 288 GB; both runs use under 20% of it.

---

## 4. Optimizations

In the order they were added. "Bit-identical" means identical to the code before the change; "train/rollout consistent" means the rollout's and the
training's log-probs are identical (|Δlogp| = 0). Gains are as measured when each change was added, under the conditions stated per section; they do not add up.

### 4.1 #51's bit-identical part (`d0ada15` → `f3f3ca5`)

4 GPUs, 60 steps, rank-0 median: original example 6076 ms (one run) → `f3f3ca5` 812.5 ms (two-run mean). All **bit-identical**; this is BX20's code.

| Switch (default) | What it does | Measured |
|---|---|---|
| `SIMWAM_GRPO_REF_CACHE` (1) | reference log-probs once per rollout | first round together (BX, 4 GPUs): 6076 → 1389.5 ms |
| `SIMWAM_GRPO_CUDA_GRAPH` (1) | no-grad `action_velocity` (denoising, old / reference log-probs) replayed from CUDA graphs | as above |
| `SIMWAM_GRPO_BATCHED_LOGP` (1) | the grad-enabled log-probs of the 3 stochastic steps in one call, GEMMs still per step | as above |
| `SIMWAM_GRPO_LEAN_ATTN` (1) | Lumen attention's no-grad calls capturable and launched once | as above |
| `SIMWAM_GRPO_PDM_WORKERS` (8) | PDM rewards scored in a process pool | as above |
| `SIMWAM_GRPO_OVERLAP_REWARD`, `SIMWAM_GRPO_FUSED_LORA` | PDM scoring overlapped with the rollout's log-probs; LoRA output with one `torch.add` | second round together (BX2, 4 GPUs): −9.0% |
| `SIMWAM_GRPO_MANUAL_DP` (1), `SIMWAM_GRPO_TRAIN_GRAPH` (1) | data parallel without DeepSpeed (fp32-master AdamW, bit-identical to ZeRO-1) plus CUDA graphs for the training forward and backward | 4 GPUs (mean of the two runs' medians): DeepSpeed 1316.5 → 812.5 ms (−38.3%) |

### 4.2 Attention and GEMM paths (`c040e02`; 4 GPUs, 60 steps, one switch added per row, two interleaved runs)

| Switch (default) | What it does | Measured | Numerics |
|---|---|---|---|
| `SIMWAM_GRPO_DROP_MASKS` (1) | all-True masks not passed to SDPA; rollout and training both take torch's flash path (aiter CK underneath) | 810 → 748 ms (−7.7%) | not bit-identical, train/rollout consistent |
| `SIMWAM_GRPO_BATCHED_GEMM` (1) | each GEMM of the batched log-prob call runs once over all steps | 748 → 625.5 ms (−16.4%) | not bit-identical; alone, train/rollout differ |
| `SIMWAM_GRPO_BATCHED_LOGP_NOGRAD` (1) | the rollout's old / reference log-probs batched too, on the training's kernels | 625.5 → 606 ms (−3.1%) | not bit-identical; with the previous row train/rollout consistent |
| `SIMWAM_GRPO_AITER_RMSNORM` (1) | RMSNorm on aiter's CK `rms_norm` | 606 → 563.5 ms (−7.0%) | not bit-identical, train/rollout consistent |

### 4.3 Fused kernels and aiter ops (2 GPUs, 40 steps, interleaved, by GPU phase or median cycle)

| Switch (default) | What it does | Measured | Numerics |
|---|---|---|---|
| `SIMWAM_GRPO_TRITON_RMSNORM_BWD` (1) | the aiter RMSNorm's backward in one Triton kernel | cycle −12.2% | not bit-identical, train/rollout consistent |
| `SIMWAM_GRPO_TRITON_ROPE` (1) | RoPE forward / backward as an fp32 Triton kernel instead of float64 complex arithmetic | −4.3% | not bit-identical, train/rollout consistent |
| `SIMWAM_GRPO_PREFILL_GRAPH` (1) | video prefill replayed from a CUDA graph | −4.2% | not bit-identical, train/rollout consistent |
| `SIMWAM_GRPO_AITER_ATTN_BWD` (1) | grad-enabled attention backward on aiter (deterministic mode), forward unchanged | −8.5% | not bit-identical, train/rollout consistent |
| `flydsl_conv` / `aiter_conv` in `LUMEN_ENABLE` | VAE convolutions on aiter FlyDSL; DiT patch embedding on aiter Triton conv3d | one VAE encode −22%, end to end unchanged | not bit-identical, train/rollout consistent |
| `SIMWAM_GRPO_AITER_GEMM` (1) | tuned-table hits of `nn.Linear` forward on aiter's tuned GEMM | GPU phases −2.2% (denoising −5.5%) | not bit-identical, train/rollout consistent |
| `SIMWAM_GRPO_XATTN_KV` (1) | cross-attention K/V of the text context once per condition | GPU phases −8.9% (denoising −16%), +0.76 GB per GPU | not bit-identical, train/rollout consistent |
| `SIMWAM_GRPO_FUSED_LN_MODULATE` (1) | LayerNorm + modulate as one Triton kernel | GPU phases −6.5% | not bit-identical, train/rollout consistent |
| `SIMWAM_GRPO_LORA_CAT` (1) | LoRA A stored below the base weight: one GEMM gives the base output and `x @ A.T`, one Triton kernel adds `mid @ B.T` | GPU phases −5.9% (denoising −13.5%) | not bit-identical, train/rollout consistent |
| `SIMWAM_GRPO_AITER_LAYERNORM` (1) | affine LayerNorm on aiter's Triton `layer_norm` | kernel about 25% faster, no measurable change per cycle | not bit-identical, train/rollout consistent |

### 4.4 Removing repeated work and copies (bit-identical)

| Switch (default) | What it does | Measured |
|---|---|---|
| `SIMWAM_GRPO_SHARED_KV` (1) | the condition repeated W times in batched calls kept in one set of shared buffers, read in place by the graphs | 2 GPUs: peak memory −8.8 GB in training, −6.6 GB in the eval |
| `SIMWAM_GRPO_EVAL_EMPTY_CACHE` (1) | `empty_cache` after each eval | device memory after an eval −6.4 GB |
| `SIMWAM_GRPO_KV_SLOT` (1) | in the no-grad graphs the video / action K/V concatenation writes only the action rows | denoising −5.8% |
| `SIMWAM_GRPO_KV_SLOT_TRAIN` (1) | the same in the training graphs | training forward −7.4%, memory −2.1 GB |
| `SIMWAM_GRPO_XATTN_GROUPED` (nograd) | no-grad batched calls attend per step group without repeating the context K/V | rollout log-probs −2.5 ms |
| `SIMWAM_GRPO_EXACT_GATE` (1) | `x + gate * residual` as one Triton kernel that rounds like torch | denoising −4% |
| `SIMWAM_GRPO_LORA_NOGRAD_SKIP` (1) | no-grad calls keep no LoRA intermediates for a backward | GPU phases −3.6% |
| `SIMWAM_GRPO_TRAIN_GRAD_FLAT` (1) | the training graphs' backward writes gradients straight into the optimizer's flat buffer | backward −2.8 to −3.8 ms, optimizer about −3 ms |
| `SIMWAM_GRPO_KV_SLOT_SHARED_NOGRAD` (1) | the rollout's batched log-probs also write only the action rows (in the training's slot buffers) | rollout log-probs −4.0 to −5.2 ms |
| `SIMWAM_GRPO_EXPAND_VIEW` (1, `62acd4c5`) | a one-sample condition's video K/V as an expand view instead of 8 copies | memory −716 MB |
| `SIMWAM_GRPO_KV_SLOT_STEP_SHARED` (1, `62acd4c5`) | the per-step denoising graphs read the first 8 rows of the shared buffers in place instead of copying them for every new condition | bytes copied into the graphs 7.25 GB → 163 MB per 40 steps; GPU phases −1.8 to −6.0 ms; memory −1.45 GB |

### 4.5 Host side

| Switch (default) | What it does | Measured | Numerics |
|---|---|---|---|
| `SIMWAM_GRPO_PDM_PREFETCH` (1) | the next batch's PDM metric caches loaded while the current batch trains | 2 GPUs: cycle −4.6%; 4 GPUs, 1000 steps, against the `rollout` mode: mean cycle 425 vs 509 ms | bit-identical |
| `SIMWAM_GRPO_VIS_BACKGROUND` (1) | rank 0 draws its figures in a background thread | 4 GPUs: about −26 ms per cycle on average (the other ranks no longer wait for rank 0) | bit-identical, same figures byte for byte |

### 4.6 Tried and not kept

| Attempt | Result |
|---|---|
| Grad-enabled attention on aiter's Triton FMHA (`SIMWAM_GRPO_AITER_FMHA_GRAD`) | 4.2% faster but train and rollout differ again; off |
| `SIMWAM_GRPO_FUSED_MODULATE` (addcmul) | 2–3 ms slower backward in a 4-GPU 4000-step run, no faster cycle; off |
| LoRA second GEMM via `addmm`, NUMA pinning | 3.1% and 2.3% slower; off |
| `SIMWAM_GRPO_PREFILL_NOCOPY` | half the prefills fell back to eager, only 0.8 GB saved; off |
| aiter tuning table for LORA_CAT's new GEMM shapes | single kernels 8–23% faster, cycle about −1% (within noise); not included this time |

---

## 5. Other improvements

Besides the cycle time, the changes brought the following. Each figure names the run it comes from; "2-GPU A/B" means 40-step runs interleaved on
GPUs 5+6, "4-GPU 4000 steps" a stage-check run on GPUs 4–7.

### 5.1 Memory

| Switch or change | What it does | Measured (per GPU) | Source |
|---|---|---|---|
| `SIMWAM_GRPO_SHARED_KV` (1) | the video K/V and context repeated W times in batched calls kept in one set of shared buffers instead of one copy per condition dict and per graph | peak allocated −8.8 GB in training, −6.6 GB in the eval | 2 GPUs, round 5 R1 |
| `SIMWAM_GRPO_EVAL_EMPTY_CACHE` (1) | free the cache after each eval | device memory after an eval −6.4 GB, peaks unchanged | 2 GPUs, round 5 R3 |
| eval K/V buffers (`37c84ff9`) | XATTN_KV no longer keeps a resident set of buffers for the eval's larger condition; the eval computes them inside its graphs | eval peak allocated 39997 → 35318 MB | 2 GPUs, E23 vs E24 |
| no KV_SLOT in the eval (`bc8cd91e`) | at the eval's shapes the slot buffers were cloned and then allocated larger | avoids +5.9 GB of eval reserved peak | 2 GPUs, C25 vs K0 / F28 |
| `SIMWAM_GRPO_KV_SLOT_TRAIN` (1) | the training graphs' concatenation writes only the action rows; the shared buffers are the slots' leading rows | training peak allocated −2.1 GB | 2-GPU A/B, r19 |
| `SIMWAM_GRPO_XATTN_KV` (1) | shared precomputed K/V buffers | +0.76 GB (cost) | 2-GPU A/B |
| `SIMWAM_GRPO_EXPAND_VIEW` (1) | video K/V as an expand view instead of 8 copies | training peak allocated −716 MB | 2-GPU A/B, r25 |
| `SIMWAM_GRPO_KV_SLOT_STEP_SHARED` (1) | the per-step denoising graphs no longer own slot buffers | with the previous row: training peak allocated −1.45 GB, reserved −2.2 GB; eval peak allocated −1.4 GB | 2-GPU A/B, r25; XV40 vs V37 |
| PDM worker metric-cache bound (`37c84ff9`) | each worker keeps the caches of its latest 8 tokens | host memory of the 32 workers: 53 → 121 GiB over 4000 steps and still growing → flat at 41.8 GiB | 4-GPU 4000 steps, CFF vs CFG |
| overall (20000 steps) | all of the above | allocated 35.62 → 34.77 GiB, reserved 44.39 → 39.61 GiB, device memory in use 53.36 → 42.90 GiB | 4-GPU 20000 steps, BX20 vs FIN20 |

### 5.2 Train / rollout log-prob consistency: from 1.16e-3 to 0

In the original example the rollout and the training compute log-probs with different kernels (rollout attention on Lumen Triton, the training forward
on torch SDPA; the training's GEMMs also have other shapes than the rollout's), so the importance ratio is not 1 at the first inner update.

| Configuration (4 GPUs, 60 steps, one switch added per row) | mean / max \|Δlogp\| | outside the PPO clip |
|---|---:|---:|
| #51 `f3f3ca5` (bit-identical) | 1.10e-3 / 0.0133 | 4.0% |
| + `SIMWAM_GRPO_DROP_MASKS` | 0 / 0 | 0% |
| + `SIMWAM_GRPO_BATCHED_GEMM` (alone) | 1.72e-3 / 0.0186 | 8.0% |
| + `SIMWAM_GRPO_BATCHED_LOGP_NOGRAD` | 0 / 0 | 0% |
| 20000 steps: BX20 vs FIN20 | 1.16e-3 / 0.0324 → 0 / 0 | 4.60% → 0% |

- The key is putting the rollout on the training's path: with the all-True masks dropped, rollout and training both take torch's flash forward; the
  no-grad log-probs are batched too and use the training's GEMMs.
- Every later value-changing switch (aiter GEMM, LORA_CAT, fused kernels, ...) keeps rollout and training on one path; every 4-GPU 4000-step run checked |Δlogp| = 0.
- Grad-enabled attention on aiter's Triton kernel (`SIMWAM_GRPO_AITER_FMHA_GRAD=1`) is 4.2% faster but brings the gap back to 1.02e-3, so it is off.
- With consistency the clip fraction is about 1 point lower in every 500-step block (4-GPU 4000 steps, CFD vs C51b).

### 5.3 Host side and the tail of the PDM waits

The PDM rewards come from closed-loop simulation on the CPU; a few tokens take hundreds of milliseconds to load their metric cache (lzma + unpickle) or
to simulate, and the whole cycle waits for them.

| Switch (default) | What it does | Measured | Source |
|---|---|---|---|
| `SIMWAM_GRPO_PDM_WORKERS` (8) | 8 scoring processes per rank | PDM scoring about 133 → 35 ms | 2 GPUs, first round |
| `SIMWAM_GRPO_OVERLAP_REWARD` (1) | scoring runs during the rollout's log-probs | cycle −6.3% | 2 GPUs, second round |
| `SIMWAM_GRPO_PDM_PREFETCH` (1) | the next batch's metric caches loaded while the current batch trains | mean cycle −23%, 90th percentile −44% | 4-GPU 4000 steps, CFG vs CFF |
| same, against the `rollout` mode | the `rollout` mode queues the scoring behind the loads | mean cycle 425 vs 509 ms, median 326 vs 507 ms; slow cycles (over 200 ms waiting outside the GPU) 18–23% vs 46–50%; scoring queue p90 27–117 vs 342–350 ms | 4-GPU 1000 steps × 2 rounds, pf4 |
| `SIMWAM_GRPO_VIS_BACKGROUND` (1) | rank 0 draws its figures in a background thread | the other ranks no longer wait about 0.6 s per figure: about −26 ms per cycle on average on 4 GPUs | 4-GPU 4000 steps, per-rank analysis of CFG |

In the 20000-step FIN20 run the non-eval cycles have a median of 387.9 ms and a 90th percentile of 664.9 ms; that tail is still mostly PDM waits (section 1.2).

### 5.4 CUDA graphs and replay checks

| Switch (default) | Calls covered | Check | Latest check |
|---|---|---|---|
| `SIMWAM_GRPO_CUDA_GRAPH` (1) | no-grad `action_velocity`: rollout denoising, old / reference log-probs | `SIMWAM_GRPO_CUDA_GRAPH_VERIFY=1`: every replay compared bit for bit with an eager run | `62acd4c5`, all defaults, 24 steps with an eval: 92 per rank, no mismatch (XV41) |
| `SIMWAM_GRPO_TRAIN_GRAPH` (1) | the grad-enabled `action_velocity` forward and backward (LoRA gradients leave the graph as outputs) | `SIMWAM_GRPO_TRAIN_GRAPH_VERIFY=1`: output and all 480 LoRA gradients against eager | same run: 24 per rank, error 0 |
| `SIMWAM_GRPO_PREFILL_GRAPH` (1) | the video prefill (frozen video expert) | covered by the end-to-end comparisons above | 2 GPUs, round 3: −4.2% |
| `SIMWAM_GRPO_LEAN_ATTN` (1) | Lumen attention's no-grad calls made capturable | `SIMWAM_GRPO_LEAN_ATTN_VERIFY=1` | #51: 4380 calls identical to `hf_attn_patch` |

- The training graphs can only be captured without DeepSpeed (under DeepSpeed ZeRO-1 the capture segfaults; root cause not found); see 5.6.
- `TRAIN_GRAPH_VERIFY` draws its probe from a private `torch.Generator` (`e527632`), so turning the check on no longer changes the rollout's random numbers.
- Known issue: `SIMWAM_GRPO_BATCHED_LOGP_VERIFY=1` crashes with the defaults; do not turn it on with them.

### 5.5 Diagnostic statistics

The measurement hook writes these into each cycle's pstats (the hook is not in this repository; the statistics live in `plugin.py`'s `_STATS`, so any tool that reads `_STATS` can use them):

| Statistic | Content | Use |
|---|---|---|
| `cuda_graph` | replay, eager and mismatch counts, loads per new condition and `load_mb` (bytes copied into the graphs' static inputs), slot concatenation counts | `load_mb` measures KV_SLOT_STEP_SHARED directly (7.25 GB → 163 MB per 40 steps) |
| `train_graph` | graph replays and captures, checks and largest error, training slot concatenations | checking the training graphs |
| `pdm` | per rollout: collect wait, scoring queue delay, prefetch load and barrier times, broken barriers | found why FIN4 waited about 250 ms more per cycle (the PDM_PREFETCH mode) |
| `SIMWAM_GRPO_CAPTURE_OPLOG=1` | while each CUDA graph is captured, the copy-like ops (copy, clone, cat, add, mul, ...) with shape, count, bytes and calling lines, 25 largest per graph kind | confirmed no video K/V copy inside the graphs; found the training forward's context K/V repeat, about 2.2 GB per forward (a candidate, not done) |
| `expand_view`, `aiter_gemm`, `lora_cat`, `exact_gate`, ... | hit and fallback counts per switch | checking that a switch is active |

### 5.6 Training without DeepSpeed

| Item | Content | Measured | Source |
|---|---|---|---|
| `SIMWAM_GRPO_MANUAL_DP` (1) | the LoRA gradients flattened into one buffer and all-reduced, fp32-master AdamW; the averaged gradient rounded to bf16 and the gradient norm accumulated as ZeRO-1 does (`4af7c72`), bit-identical to DeepSpeed ZeRO-1 | 2 GPUs: without DeepSpeed alone 881 → 749 ms (−15%), with the training graphs 656 ms (−25.5%) | 2 GPUs, A / C / C+B interleaved |
| with the training graphs | the above plus `SIMWAM_GRPO_TRAIN_GRAPH` | 4 GPUs (mean of the two runs' medians): DeepSpeed 1316.5 → 812.5 ms (−38.3%), backward about 590 → 249 ms | 4 GPUs, third round |
| why ZeRO-1 is not needed | 15.7 M trainable parameters, so sharding saves about 94 MB per GPU; ZeRO-1's per-parameter norm and overflow checks are about 16.6k small kernels per step | about 60 ms per cycle in the profile | second-round profile |

Note: checkpoints are not interchangeable between the two modes; to resume an older DeepSpeed checkpoint set `SIMWAM_GRPO_MANUAL_DP=0`.

### 5.7 aiter ops

Already on aiter:

| Op | aiter implementation | Switch |
|---|---|---|
| attention forward (rollout and training) | torch flash, aiter CK `fmha_fwd_hd128` underneath | `SIMWAM_GRPO_DROP_MASKS` |
| attention backward | aiter CK `FmhaBwdDQDKDV` (deterministic mode) | `SIMWAM_GRPO_AITER_ATTN_BWD` |
| RMSNorm forward | aiter CK `rms_norm` | `SIMWAM_GRPO_AITER_RMSNORM` |
| affine LayerNorm forward and backward | aiter Triton `layer_norm` | `SIMWAM_GRPO_AITER_LAYERNORM` |
| `nn.Linear` forward (43 tuned shapes: 18 hipBLASLt, 14 Triton, 11 torch, no split-K) | aiter `tuned_gemm.gemm_a16w16` | `SIMWAM_GRPO_AITER_GEMM` |
| VAE convolutions | aiter FlyDSL implicit GEMM | `flydsl_conv` in `LUMEN_ENABLE` |
| DiT patch embedding (Conv3d) | aiter Triton conv3d (`AITER_SRC`) | `aiter_conv` in `LUMEN_ENABLE` |

Not on aiter (times from a one-cycle 2-GPU profile of `4415df7c`, P31; later changes removed part of them):

| Op | Current implementation | Per cycle | aiter has one | Why not replaced |
|---|---|---:|---|---|
| backward dX / dW GEMMs | torch → hipBLASLt | 40.3 ms | the tuned GEMM covers the forward layout only | not tried; would change the backward numerics and need re-validation |
| LORA_CAT's new GEMM shapes | torch → hipBLASLt | not split out | yes, 5 shapes tuned | cycle about −1%, within noise; not included this time |
| RoPE | own fp32 Triton | 6.1 ms | yes | not tried; already one kernel |
| RMSNorm backward | own Triton | 2.8 ms | aiter's Triton rmsnorm has a backward | not tried; small expected gain |
| LayerNorm + modulate (the gate was still torch in P31) | own Triton | 6.1 ms | no matching fusion | no matching op |
| element-wise add / mul | torch | 37.3 ms (mul 22.1, add 15.2) | yes | memory-bound, a one-for-one swap is not faster; fusion is what helps |
| cat / copy | torch / HIP runtime | 33.7 ms (cat 14.6, copy 8.4, memcpy 10.7) | no | later KV slots, shared buffers, flat gradients etc. removed part of it (measured per switch in section 4) |
| AdamW | torch foreach | 1.8 ms | no | no matching op |
| GELU (tanh) | torch | 3.3 ms | yes (JIT-compiled on first use) | memory-bound, not tried |
| all-reduce | RCCL | 4.7 ms | yes (for small messages) | not tried |

---

## 6. Runbook

### 6.1 Environment

The image, Lumen, aiter, SimWAM, weights and NAVSIM data are set up as in [README.md](README.md) ("What you need", "Environment"). These results also need:

- `AITER_SRC` pointing at a checkout of `huizzhan/aiter` branch `hz/grpo-opt` (`c9be3a5`): conv3d and this example's GEMM tuning table are loaded from
  there. Without it a warning is logged and conv3d and AITER_GEMM do nothing.
- `flydsl_conv` needs Lumen PR #32's `examples/wan2.2/flydsl_conv.py`; with the image's aiter run `$LUMEN_DIR/examples/perf-veomni/overlay_aiter.py --apply`
  in the container first.

```bash
bash examples/Embodied_SimWAM_NAVSIM/launch_container.sh
docker exec -it simwam-rl bash examples/Embodied_SimWAM_NAVSIM/setup_env.sh
```

### 6.2 Optimized run (FIN20's configuration, all defaults)

Inside the container, from the Lumen-RL root:

```bash
NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
LUMEN_DIR=<Lumen> SIMWAM_ROOT=<SimWAM> AITER_SRC=<aiter hz/grpo-opt checkout> \
  bash examples/Embodied_SimWAM_NAVSIM/run_embodied_grpo.sh \
  data.train.scene_filter=<Lumen-RL>/examples/Embodied_SimWAM_NAVSIM/configs/navtrain_pdm_score_below0p9_complete.yaml \
  max_steps=20000
```

Leave `LUMEN_ENABLE` unset to get the script default `hf_attn_patch,aiter_conv,flydsl_conv`. The log should show
`aiter_conv bound 63 Conv3d and 10 Conv2d` and no conv3d fallback warning.

### 6.3 Going back to the bit-identical path

**Most reliable**: use #51's head before this merge, `f3f3ca5` (BX20's code), with `LUMEN_ENABLE=hf_attn_patch`, otherwise as in 6.2.
That combination is bit-identical to the original example `d0ada15` over 8 steps on 4 GPUs.

**On the current code**: set `LUMEN_ENABLE=hf_attn_patch` and turn off every value-changing switch (section "Faster, not bit-identical" of README.md):

```bash
export LUMEN_ENABLE=hf_attn_patch
export SIMWAM_GRPO_DROP_MASKS=0 SIMWAM_GRPO_BATCHED_GEMM=0 SIMWAM_GRPO_BATCHED_LOGP_NOGRAD=0 \
       SIMWAM_GRPO_AITER_RMSNORM=0 SIMWAM_GRPO_TRITON_RMSNORM_BWD=0 SIMWAM_GRPO_AITER_ATTN_BWD=0 \
       SIMWAM_GRPO_TRITON_ROPE=0 SIMWAM_GRPO_PREFILL_GRAPH=0 SIMWAM_GRPO_XATTN_KV=0 \
       SIMWAM_GRPO_FUSED_LN_MODULATE=0 SIMWAM_GRPO_AITER_GEMM=0 SIMWAM_GRPO_AITER_LAYERNORM=0 \
       SIMWAM_GRPO_LORA_CAT=0 SIMWAM_GRPO_FUSED_MODULATE=0
```

The bit-identical switches (sections 4.4 and 4.5, including `62acd4c5`'s `SIMWAM_GRPO_EXPAND_VIEW` and `SIMWAM_GRPO_KV_SLOT_STEP_SHARED`) can stay on.
Note: this combination has not been re-checked bit for bit against `d0ada15` on the current head; use `f3f3ca5` when a strictly bit-identical baseline
is needed. To get `daf60b17`'s (FIN20's) behaviour back only: `SIMWAM_GRPO_EXPAND_VIEW=0 SIMWAM_GRPO_KV_SLOT_STEP_SHARED=0`.

### 6.4 Acceptance

- The run completes the target steps with exit code 0; the log shows 63 Conv3d bound and no fallback warning.
- A short run with `SIMWAM_GRPO_CUDA_GRAPH_VERIFY=1 SIMWAM_GRPO_TRAIN_GRAPH_VERIFY=1` (e.g. 24 steps, `eval_every=20`): every graph replay identical to eager.
- Compare performance with repeated runs interleaved on the same GPUs; compare switches that change the training trajectory by GPU phase
  (in 40 steps the median cycle is dominated by the tail of the PDM scoring).

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `aiter conv3d failed to load ...; Conv3d stays on torch` | `AITER_SRC` unset or a checkout without conv3d | point it at `huizzhan/aiter:hz/grpo-opt` |
| `SIMWAM_GRPO_AITER_GEMM: AITER_CONFIG_GEMM_BF16 is unset` | no tuning table for this example in `AITER_SRC` | as above, or set `AITER_CONFIG_GEMM_BF16` explicitly |
| VAE still on torch, FlyDSL warning in the log | the image's aiter lacks the FlyDSL conv kernel | run `overlay_aiter.py --apply` in the container first |
| Occasional cycles a few hundred ms longer | the tail of the PDM scoring (metric-cache load or simulation of some tokens) | expected; compare mean cycles or GPU phases |
| `SIMWAM_GRPO_BATCHED_LOGP_VERIFY=1` crashes | known issue, unrelated to the default switches | do not turn it on with the defaults |

---

## 8. Files

| File | Role |
|---|---|
| `plugin.py` | all runtime patches and switches |
| `fused_kernels.py` | hand-written Triton kernels (RMSNorm backward, RoPE, LayerNorm + modulate, gate, LoRA) |
| `train_grpo.py` | installs the patches and starts SimWAM's GRPO training |
| `run_embodied_grpo.sh` | training entry point; sets `LUMEN_ENABLE`, `AITER_SRC` and the GEMM tuning table |
| `README.md` | each switch in detail with its own measurement |
