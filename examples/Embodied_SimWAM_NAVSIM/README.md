# Embodied SimWAM on NAVSIM

FlowGRPO for [SimWAM](https://github.com/H-EmbodVis/SimWAM) on NAVSIM, four AMD Instinct MI350X. The video expert, action expert, and Wan2.2 VAE stay on their own code. This directory only turns on Lumen kernels.

SimWAM under `examples/SimWAM` is an unmodified checkout. The hook lives in `train_grpo.py`: after `configure_grpo` has copied the frozen reference expert and wrapped LoRA, `plugin.enable_lumen` applies the flags in `LUMEN_ENABLE`.

Quantization stays off (`LumenConfig(scaling="none")`). One patch failing to load does not turn the others off; that op stays on torch.

## What you need

| Piece | Where |
|---|---|
| GPU | MI350X. Default cards are `4,5,6,7`. |
| Base image | `zhangdanyangamd/lumen-rl:embodied-MI350X` (torch 2.12 + ROCm 7.14, already in the image). |
| Lumen | Host checkout at `$LUMEN_DIR`, branch `dev/world-model-rl`. Put on `PYTHONPATH`. Not installed as a wheel. |
| aiter | Host checkout at `$AITER_SRC` with `aiter/ops/triton/conv/conv3d.py`, for example branch `hz/grpo-opt` of `huizzhan/aiter` (upstream ROCm/aiter `882bd6207`). Conv3d is loaded from here because the image copy has no `conv3d`. No default: when it is unset, `aiter_conv` logs a warning and Conv3d (the DiT patch embedding) stays on torch. |
| SimWAM code | `examples/SimWAM` (upstream `main`). |
| Weights | `Wan-AI/Wan2.2-TI2V-5B` (includes `Wan2.2_VAE.pth`), Wan2.1 tokenizer `Wan-AI/Wan2.1-T2V-1.3B`, ActionDiT `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim_3outdim.pt`, IL warm start `weights/SimWAM.pt` (`SIMWAM_IL_CHECKPOINT`). |
| NAVSIM | nuPlan maps, navtrain logs, sensor blobs, metric cache. Paths are the `NAVSIM_*` / `NUPLAN_*` variables in `run_embodied_grpo.sh`. |

`examples/SimWAM` is its own git clone. Do not commit it into Lumen-RL.

## Environment

Two equivalent ways to get the Python packages. Neither one reinstalls torch or triton.

### Use the published image

```bash
bash examples/Embodied_SimWAM_NAVSIM/launch_container.sh
docker exec -it simwam-rl bash examples/Embodied_SimWAM_NAVSIM/setup_env.sh
```

`launch_container.sh` mounts the host home directory, passes GPUs `4,5,6,7`, and replaces the image's dangling `/root/.cache/miopen` symlink. `setup_env.sh` editable-installs SimWAM (`--no-deps`) and the trainer packages.

### Build the Dockerfile

The Dockerfile starts from the same image and installs those trainer packages. Lumen, aiter, SimWAM, weights, and data are still mounted at run time.

```bash
docker build \
  -f examples/Embodied_SimWAM_NAVSIM/Dockerfile \
  -t simwam-rl:mi350x \
  examples/Embodied_SimWAM_NAVSIM

IMAGE=simwam-rl:mi350x bash examples/Embodied_SimWAM_NAVSIM/launch_container.sh
docker exec -it simwam-rl bash examples/Embodied_SimWAM_NAVSIM/setup_env.sh
```

`setup_env.sh` is still required after the build: it installs the mounted SimWAM tree and checks that Lumen imports.

Override the mount root or the cards if this machine differs:

```bash
ROOT=/home/danyzhan CUDA_VISIBLE_DEVICES=4,5,6,7 \
  bash examples/Embodied_SimWAM_NAVSIM/launch_container.sh
```

## Patches

`LUMEN_ENABLE` is a comma-separated list. Default in `run_embodied_grpo.sh` and the Dockerfile:

```bash
export LUMEN_ENABLE=hf_attn_patch,aiter_conv
```

`lumen_linear` is not in the training default. The AITER tuned bf16 GEMM has no entry for the SimWAM shapes and falls back to the torch GEMM, so training is bit-identical to `hf_attn_patch,lumen_linear` (forward, LoRA gradients and weights, checked step by step) and only the dispatch cost is gone: a rollout cycle is 14% faster.

`hf_attn_patch` stays in the default so that results are bit-identical to the original example. SimWAM passes a bool mask on every attention call, so Lumen takes only the `no_grad` ones (rollout, old and reference log-probs); `install_lean_lumen_attention` in `plugin.py` runs them as one Triton launch with a resident zero bias and makes them capturable by the CUDA graphs (see Training-loop patches). Dropping it (`LUMEN_ENABLE=aiter_conv`) sends those calls to torch SDPA, which is faster but changes the log-probs (max abs difference 0.009 on one rollout). `aiter_conv` itself also changes results: with it the VAE convolutions run on AITER instead of torch, and a run differs from the original from the first step. For a run bit-identical to the original, use `LUMEN_ENABLE=hf_attn_patch`.

Each name is one `LumenConfig.enable` call. Empty, `0`, `false`, `none`, and `off` turn patching off.

| Flag | What it changes | Fallback |
|---|---|---|
| `hf_attn_patch` | `F.scaled_dot_product_attention` goes through Lumen attention. A bool keep-mask (`True` = attend) is converted to an additive bias, expanded so size-1 axes have stride 0, and a per-batch mask is launched one batch at a time (the Triton kernel ignores the batch stride). | Extra kwargs, a kernel error, or a masked call that needs a gradient: the original SDPA. Lumen's backward does not apply the bias, and those gradients overflow to inf in bf16. |
| `lumen_linear` | `nn.Linear` forward uses the tuned bf16 GEMM (`dispatch_gemm`, `scaling="none"`). Backward is a torch matmul autograd wrapper. LoRA `A`/`B` are parameters, not `nn.Linear`, so the same GEMM is installed on `LoRALinear.forward`. | If the flag fails to load, LoRA `A`/`B` stay on torch matmul. |
| `aiter_conv` | `nn.Conv2d` and `nn.Conv3d` forward use AITER. `CausalConv3d.forward` (the causal pad) is kept; only `_conv_forward` is rebound. This covers the Wan2.2 VAE (`WanVideoVAE38`, 48 latent channels). | Unsupported dtype or groups, or a kernel error, use the original conv. Conv3d is imported from `$AITER_SRC` when the installed aiter has no `conv3d`. |
| `flydsl_conv` | Only the VAE's convolutions (62 `CausalConv3d`, 10 `Conv2d`), on aiter's FlyDSL implicit-GEMM kernel, with Lumen PR #32's `examples/wan2.2/flydsl_conv.py` (`SIMWAM_GRPO_FLYDSL_DIR`, default `$LUMEN_DIR/examples/wan2.2`). A one-frame, uncached `CausalConv3d` call -- every call of the rollout's encode -- runs as the exact 2-D convolution with the last temporal tap. Applied after the `LumenConfig` flags, so `aiter_conv,flydsl_conv` gives the VAE FlyDSL and every other conv AITER Triton. The first `SIMWAM_GRPO_FLYDSL_LOG` encodes (default 2) log their FlyDSL / torch call counts. | A call that is not BF16, not on the GPU or needs a gradient uses torch. An aiter without ROCm/aiter#5370 (the image's) needs `$LUMEN_DIR/examples/perf-veomni/overlay_aiter.py --apply` in the container first; the kernel is then loaded through `examples/perf-veomni/veomni_patches/aiter_conv.py` (`SIMWAM_GRPO_FLYDSL_LOADER_DIR`). Without either, the VAE keeps its previous path. |

Examples:

```bash
# Attention only. Linear and conv stay on torch.
LUMEN_ENABLE=hf_attn_patch bash examples/Embodied_SimWAM_NAVSIM/run_embodied_grpo.sh

# GEMM and conv, no attention patch.
LUMEN_ENABLE=lumen_linear,aiter_conv bash examples/Embodied_SimWAM_NAVSIM/run_embodied_grpo.sh

# Stock torch for every op.
LUMEN_ENABLE=0 bash examples/Embodied_SimWAM_NAVSIM/run_embodied_grpo.sh
```

The training log line `Lumen enabled on SimWAM: [...]` is the list that actually loaded, followed by a module count (`nn.Linear`, `nn.Conv2d`, `nn.Conv3d`) and parameter size per expert.

`bench_kernels.py` builds the real video expert, a LoRA-wrapped action expert, and `WanVideoVAE38`, then checks a video block against torch, an action backward, and a VAE encode. It does not need NAVSIM data or checkpoints.

```bash
docker exec -it simwam-rl bash -lc \
  'cd /home/danyzhan/Lumen-RL && LUMEN_ENABLE=hf_attn_patch,lumen_linear,aiter_conv \
   python3 examples/Embodied_SimWAM_NAVSIM/bench_kernels.py'
```

Set `LUMEN_ENABLE` to a subset to check one patch at a time.

## Training-loop patches

`train_grpo.py` also installs these runtime patches from `plugin.py` (SimWAM is not modified). With `LUMEN_ENABLE=hf_attn_patch` (the default without `aiter_conv`, so the VAE convolutions stay on torch as in the original) and the four switches under "Faster, not bit-identical" set to `0`, a run is bit-identical to the original example with `LUMEN_ENABLE=hf_attn_patch,lumen_linear` and the LoRA seed below: the LoRA init, every rollout tensor and reward, every log-prob, every LoRA weight after every optimizer step, and loss, KL and grad norm. With their defaults (on) it is not.

| Patch | Switch (default) | What it does |
|---|---|---|
| `install_lora_init_seed` | `SIMWAM_GRPO_LORA_SEED` (`42`, `off` disables) | Seeds all RNGs (`set_global_seed`, plus rank) right before `configure_grpo`. On the NAVSIM path SimWAM seeds only in the trainer, after LoRA A is drawn, so without it two runs start from different adapters. |
| `install_ref_logprob_cache` | `SIMWAM_GRPO_REF_CACHE` (`1`) | Reference log-probs are computed once per rollout instead of once per inner epoch. |
| `install_lean_lumen_attention` | `SIMWAM_GRPO_LEAN_ATTN` (`1`; `loop`; `0`) | Only with `hf_attn_patch`. Lumen's Triton dispatcher copies unit scales from the host, which fails under CUDA graph capture, and `hf_attn_patch` then silently falls back to torch SDPA. This calls the same Triton forward with resident scales. `1`: inside `action_velocity` the masks are all True (checked once per condition), so the same zero bias is kept resident and broadcast over the batch: one launch instead of one per sample. `loop`: `hf_attn_patch`'s ops one for one. `SIMWAM_GRPO_LEAN_ATTN_VERIFY=1` compares every uncaptured call with `hf_attn_patch`. |
| `install_action_velocity_graphs` | `SIMWAM_GRPO_CUDA_GRAPH` (`1`) | `no_grad` `action_velocity` (rollout denoising, old and reference log-probs) replays one CUDA graph per input shape. A masked call that would go to `hf_attn_patch` outside the lean path is never captured. `SIMWAM_GRPO_CUDA_GRAPH_VERIFY=1` compares every replay with eager. |
| `install_batched_logprobs` | `SIMWAM_GRPO_BATCHED_LOGP` (`1`) | The grad-enabled log-probs of the 3 stochastic steps run in one `action_velocity` call, but every GEMM (`nn.Linear`, LoRA A/B, Lumen bf16 GEMM) still runs once per step on its slice, with the original shapes and strides. A single GEMM over all steps picks other kernels, and its LoRA weight gradient would be one reduction instead of 3 GEMMs summed in bf16. `SIMWAM_GRPO_BATCHED_CHUNK=rmsnorm,attn` also splits those ops; `SIMWAM_GRPO_BATCHED_LOGP_VERIFY=1` re-runs the per-step path and compares outputs and LoRA gradients (with this script's defaults it currently fails in the first inner update: the policy loss gets non-finite means). |
| `install_parallel_pdm_reward` | `SIMWAM_GRPO_PDM_WORKERS` (`8`, `0` disables) | PDM rewards are scored in a pool of spawned workers. |
| `install_manual_data_parallel` | `SIMWAM_GRPO_MANUAL_DP` (`1`; `0` uses DeepSpeed ZeRO-1) | Multi-GPU training without DeepSpeed: the LoRA gradients are all-reduced and an fp32-master AdamW steps them, bit-identical to DeepSpeed ZeRO-1. Needed for the training CUDA graphs below. |
| `install_train_graphs` | `SIMWAM_GRPO_TRAIN_GRAPH` (`1`) | Only with manual data parallel. The grad-enabled `action_velocity` forward and backward replay from CUDA graphs captured per input shape; the LoRA gradients leave the graph and are reduced and stepped outside it. `SIMWAM_GRPO_TRAIN_GRAPH_VERIFY=1` compares every replay (output and all LoRA gradients) with eager. |
| `install_batched_logprobs` | `SIMWAM_GRPO_SHARED_KV` (`1`; `0` turns it off) | The condition repeated over the batched steps (video K/V of every layer, context) is kept in one set of buffers per shape, written with the same concatenation, instead of one copy per condition plus a static copy in each CUDA graph that reads it (policy and reference no-grad graphs, training graph); the graphs read the buffers in place, and a buffer is refilled when another condition uses it. Same results bit for bit. On 2×MI350X (eval every 20 steps) the peak allocated memory per rank drops by 8.8 GB in training and 6.6 GB in the eval. |
| `install_prefill_graph` | `SIMWAM_GRPO_PREFILL_NOCOPY` (`0`) | The prefill graph returns its own output tensors instead of a copy; a call whose previous outputs are still referenced runs eagerly. Off: within a cycle the earlier rollouts' conditions are still alive, so 5 of 12 prefills ran eagerly and the eval peak dropped by only 0.8 GB. |
| `install_eval_empty_cache` | `SIMWAM_GRPO_EVAL_EMPTY_CACHE` (`1`; `0` turns it off) | `torch.cuda.empty_cache()` after each deterministic eval, which otherwise leaves its cached blocks reserved for the rest of the run: the device memory in use after an eval drops by about 6.4 GB per rank (2×MI350X); peaks are unchanged. |

### Faster, not bit-identical

| Patch | Switch (default) | What it does |
|---|---|---|
| `install_drop_true_masks` | `SIMWAM_GRPO_DROP_MASKS` (`1`) | Calls SDPA without the attention masks when they are all True, which they are here: the GRPO condition is one frame (so `first_frame_causal` masks nothing), the action rows see every token, and the NAVSIM dataset sets the text mask to ones. Checked once per condition and per video prefill; otherwise the masked path is kept. Torch then takes its flash path (AITER's CK forward) for rollout and training alike, instead of Lumen Triton for no-grad calls and aotriton with a bias for grad-enabled ones. |
| `install_batched_logprobs` | `SIMWAM_GRPO_BATCHED_GEMM` (`1`) | In the batched log-prob call each GEMM runs once over all steps instead of once per step. |
| `install_batched_logprobs` | `SIMWAM_GRPO_BATCHED_LOGP_NOGRAD` (`1`) | The no-grad log-prob calls (the rollout's old and reference log-probs) are batched the same way, so the rollout scores a chain with the same kernels as the first inner update. Keep it on with `SIMWAM_GRPO_BATCHED_GEMM`: alone, the batched GEMM makes the training log-probs differ from the rollout's. |
| `install_aiter_rmsnorm` | `SIMWAM_GRPO_AITER_RMSNORM` (`1`) | SimWAM's `RMSNorm` on AITER's CK `rms_norm` (one fused launch); grad-enabled calls use an autograd Function whose backward computes the input gradient from the RMSNorm formula in fp32. |
| `install_drop_true_masks` | `SIMWAM_GRPO_AITER_FMHA_GRAD` (`0`; `1`; `all`) | `1`: grad-enabled maskless calls on AITER's Triton `flash_attn_func`; `all`: the no-grad ones too. Off: `1` is faster but makes the training log-probs differ from the rollout's again, `all` is no faster than off. |
| `install_drop_true_masks` | `SIMWAM_GRPO_AITER_ATTN_BWD` (`1`) | Grad-enabled maskless calls keep torch's flash forward (the rollout's kernel, bit-identical output) and run the backward with AITER's flash-attention backward, in its deterministic mode (`SIMWAM_GRPO_AITER_ATTN_BWD_DETERMINISTIC`, `1`). |
| `install_aiter_rmsnorm` | `SIMWAM_GRPO_TRITON_RMSNORM_BWD` (`1`) | The input gradient of the AITER RMSNorm in one Triton kernel (`fused_kernels.py`) instead of a chain of fp32 element-wise and reduction kernels. |
| `install_triton_rope` | `SIMWAM_GRPO_TRITON_ROPE` (`1`) | `rope_apply` as one Triton kernel in fp32, forward and backward, instead of float64 complex arithmetic (casts, complex multiply, casts back). |
| `install_prefill_graph` | `SIMWAM_GRPO_PREFILL_GRAPH` (`1`) | The video prefill of `build_action_condition` (frozen expert, fixed shapes) replays a CUDA graph instead of about 30 layers of eager launches. |
| `LUMEN_ENABLE` | `flydsl_conv` (in the default) | Every VAE convolution on AITER's FlyDSL implicit-GEMM kernel through Lumen PR #32's `examples/wan2.2/flydsl_conv.py` (`SIMWAM_GRPO_FLYDSL_DIR`); the image's AITER needs `$LUMEN_DIR/examples/perf-veomni/overlay_aiter.py --apply` for the kernel. Without it the VAE keeps its previous path and a warning is logged. The first encode JIT-compiles the kernels (about 20 s). |
| `LUMEN_ENABLE` | `aiter_conv` (in the default) | With `flydsl_conv` also on, the only convolution left to `aiter_conv` is the DiT patch embedding (`video_expert.patch_embedding`, one `Conv3d` per video prefill), which runs on AITER's Triton conv3d from `$AITER_SRC`: the VAE stays on FlyDSL. Same cycle time as without it (405.0 vs 405.25 ms on 2×MI350X, two interleaved runs each), log-probs still bit-identical between rollout and training, same peak memory. `train_grpo.py` first runs `install_aiter_triton_imports` (`SIMWAM_GRPO_AITER_TRITON_IMPORTS`, `1`): it imports the installed aiter's tuned GEMM before Lumen puts the checkout in front of the installed Triton packages, and adds the checkout to the installed `_triton_kernels.common` package, which Lumen's loader leaves out (without that, conv3d stays on torch once the tuned GEMM is on). The tuned GEMM and Lumen's attention keep the installed aiter's Triton modules. `enable_lumen` logs how many Conv3d `aiter_conv` bound and warns loudly when it bound none. |
| `install_cross_attn_kv` | `SIMWAM_GRPO_XATTN_KV` (`0`) | The action expert's cross-attention K/V of the text context: computed on one copy of the context in a batched log-prob call and repeated (the rollout's no-grad log-probs and the training take this same path), and once per condition for rollout denoising (shared per-layer buffers, about 0.75 GB). Under evaluation. |
| `install_action_velocity_graphs` | `SIMWAM_GRPO_KV_SLOT` (`0`) | Inside the no-grad action graphs, the per-layer `cat` of the cached video K/V and the action K/V writes only the action rows behind the video rows of a static buffer (same values and layout, bit-identical to the `cat`). No measurable gain (413.75 vs 405.25 ms, the first run noisy), off. |
| `install_aiter_gemm` | `SIMWAM_GRPO_AITER_GEMM` (`1`; `0` turns it off) | `nn.Linear` forward on AITER's tuned bf16 GEMM (`gemm_a16w16`) for the shapes in `AITER_CONFIG_GEMM_BF16` whose best backend is not torch; the backward stays on torch. `run_embodied_grpo.sh` sets `AITER_CONFIG_GEMM_BF16` to `$AITER_SRC/aiter/configs/model_configs/simwam_grpo_bf16_tuned_gemm.csv` when that file exists (branch `hz/grpo-opt` of `huizzhan/aiter`: 43 shapes, hipBLASLt / Triton / torch, no split-K, tuned against the image's hipBLASLt); with no table set the patch logs a warning and does nothing. Rollout and training forward run the same kernels, so the log-probs stay bit-identical, and two runs give identical losses; peak memory +46 MB per rank. On 2×MI350X the GPU phases (VAE, condition, denoising, log-probs, update) are 2.2% faster, denoising 5.5%. Measuring note: a switch that changes the trajectories also changes how long the rollout waits for the PDM scores, and in 40 steps those waits dominate the cycle median, so compare such switches by GPU phase. |
| `install_fused_modulate` | `SIMWAM_GRPO_FUSED_MODULATE` (`0`) | `modulate` / `GateModule` with `torch.addcmul`; no measurable gain, off. |
| `install_fused_lora_add` | `SIMWAM_GRPO_LORA_ADDMM` (`0`) | The second LoRA GEMM accumulates into the base output (`addmm`); about 3% slower, off. |
| `install_numa_affinity` | `SIMWAM_GRPO_NUMA_AFFINITY` (`0`) | Pins each rank's threads to its GPU's NUMA node; no gain measured on GPU 5+6, off. |

Measured on 4×MI350X (`NPROC=4`, 60 steps, median of the 11 cycles from step 16 on rank 0, two interleaved runs each, `LUMEN_ENABLE=hf_attn_patch`), each row adding one switch: all four off 810 ms; masks dropped 748 ms; one GEMM over the steps 625.5 ms; no-grad calls batched 606 ms; AITER RMSNorm 563.5 ms (-30% in total). With the four on, the rollout's log-probs equal the first inner update's bit for bit (importance ratio exactly 1 before the first optimizer step); with all four off they differ by about 1e-3 on average, and 4% of the ratios start outside the default PPO clip of 0.005.

On 2×MI350X (40 steps, median of the cycles from step 16, `LUMEN_ENABLE=hf_attn_patch`), starting from those four: 537.5 ms; Triton RMSNorm backward 472 ms; Triton RoPE 451.5 ms; then, as the mean of two interleaved runs, prefill graph -4.2% and AITER attention backward -8.5%. All defaults of this script (with `flydsl_conv`) against the four alone, two interleaved runs each: 402.5 ms vs 521.5 ms (-22.8%). The rollout's log-probs still equal the training ones bit for bit, two runs of the same configuration give identical losses, and peak memory per rank grows by about 0.13 GB (allocated; reserved +0.02 GB).

## Training

Inside the container, from the Lumen-RL root:

```bash
bash examples/Embodied_SimWAM_NAVSIM/run_embodied_grpo.sh
```

`configs/embodied_simwam_navsim_grpo.yaml` is read as Hydra overrides (`task`, `num_workers`, `max_steps`, `save_every`, `grpo.eval.num_batches`). Extra overrides go after the script:

```bash
NPROC=4 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  bash examples/Embodied_SimWAM_NAVSIM/run_embodied_grpo.sh \
  max_steps=100
```

The run uses the NAVSIM task `navsim_grpo_action_pdm_384x672_flowgrpo_lora`: front camera 384×672, 9 frames, action LoRA only. The script exits before launch if maps, logs, sensor blobs, or the metric cache directory is missing.

## Worked recipe

This is the run that finished on 4×MI350X and was scored on the full NAVSIM v1 `navtest` (12146 tokens).

Two host checkouts have to be the ones this example was written against:

- Lumen `dev/world-model-rl`, including the masked-backward fallback in `lumen/ops/attention/hf_patch.py`. Without it every optimizer step logs `gnorm=inf` and the weights do not move.
- aiter with `conv3d`, at `AITER_SRC` (no default; see the table at the top).

`launch_container.sh` mounts `/mnt/k3` and creates `/work/tmp`. The image sets `TMPDIR=/work/tmp` but does not mount it; clang then fails to JIT HIP kernels and the process segfaults.

The published hard filter (`navtrain_pdm_score_below0p9`, 17389 tokens) lists frames whose camera jpg is not in the OpenScene tarballs. The scene loader reads every CAM_F0 frame in the 14-frame window, so the first missing file aborts startup. `configs/navtrain_pdm_score_below0p9_complete.yaml` keeps the 2784 tokens that have all 14 frames and a metric cache.

```bash
bash examples/Embodied_SimWAM_NAVSIM/launch_container.sh
docker exec -it simwam-rl bash examples/Embodied_SimWAM_NAVSIM/setup_env.sh

docker exec -it simwam-rl bash -lc \
  'cd /home/danyzhan/Lumen-RL && \
   RUN_ID=formal2 \
   bash examples/Embodied_SimWAM_NAVSIM/run_embodied_grpo.sh \
   data.train.scene_filter=/home/danyzhan/Lumen-RL/examples/Embodied_SimWAM_NAVSIM/configs/navtrain_pdm_score_below0p9_complete.yaml'
```

`configs/embodied_simwam_navsim_grpo.yaml` already sets `model.redirect_common_files=false` (the DiffSynth Wan conversion is gated; use the local `Wan-AI/Wan2.2-TI2V-5B` VAE and T5), `max_steps=20000`, `save_every=1000`, and `num_workers=4`. Defaults elsewhere: 4 GPUs, bf16, DeepSpeed ZeRO-1, LoRA r=16 on the action expert q/k/v/o, group size 8, 10 denoising steps of which 3 are stochastic. Checkpoints land in `examples/SimWAM/runs/grpo/.../<RUN_ID>/checkpoints/weights/`. The `weights/` files are LoRA-merged.

`runs/` is large (about 15 GB of optimizer state and 12 GB of weights per 1000 steps). Put it on a data disk. The root filesystem filled up at step ~13000 of this run; moving older checkpoints off the root disk while training continued was enough to finish.

### navtest

Official PDMS is the full `navtest` split, not the training-loop reward. The in-loop eval is 8 batches of the hard training subset.

Download OpenScene `openscene_metadata_test.tgz` and `openscene_sensor_test_camera_{0..31}.tgz` (camera only; the loader uses CAM_F0). Build the metric cache with `navsim/scripts/evaluation/run_metric_caching.sh` and `train_test_split=navtest`. A thread pool stays on one core; `worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=24` finishes the 12146 caches.

```bash
docker exec -it simwam-rl bash -lc \
  'CKPT=/path/to/step_020000.pt \
   bash examples/Embodied_SimWAM_NAVSIM/run_navtest_eval.sh'
```

The script applies `patches/eval_navsim_lumen.patch` to the local SimWAM checkout if it is not already applied. That drops `model.checkpoint_path` before `create_simwam_grpo` (the merged weights are loaded from `ckpt=`) and turns on `LUMEN_ENABLE`. Inference is action-only, current frame, 384×672, 10 steps, videos off. Score the `pred_actions/*.npy` with `pdm_score` at the reported weights (progress 5, TTC 5, comfort 2), not the training reward weights (10, 5, 2).

`step_020000.pt` on all 12146 navtest tokens:

|  | NC | DAC | EP | TTC | C | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| this run, 20000 steps, 2784 train tokens | 98.40 | 98.53 | 86.01 | 95.20 | 100.00 | 91.10 |
| SimWAM README, RL, 170000 steps | 98.5 | 99.1 | 86.5 | 96.1 | 99.8 | 91.9 |

## Layout

| File | Role |
|---|---|
| `Dockerfile` | Trainer packages on top of `lumen-rl:embodied-MI350X`. |
| `launch_container.sh` | Start `simwam-rl` with GPUs, Lumen, aiter, and a real MIOpen cache. |
| `setup_env.sh` | Editable-install SimWAM and check the Lumen import. |
| `plugin.py` | `enable_lumen` and the LoRA GEMM / `configure_grpo` hooks. |
| `train_grpo.py` | Hydra entry. Installs the hook, then calls SimWAM `run_grpo_training`. |
| `run_embodied_grpo.sh` | NAVSIM env, `LUMEN_ENABLE`, torchrun. |
| `bench_kernels.py` | Kernel check without data. |
| `configs/embodied_simwam_navsim_grpo.yaml` | Hydra overrides for the 4-GPU run. |
| `configs/navtrain_pdm_score_below0p9_complete.yaml` | Hard-subset tokens whose 14 CAM_F0 frames and metric cache exist. |
| `patches/eval_navsim_lumen.patch` | Load a merged checkpoint and enable Lumen inside SimWAM's navtest eval. |
| `run_navtest_eval.sh` | 4-GPU navtest inference. |
