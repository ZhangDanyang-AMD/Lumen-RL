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
| aiter | Host checkout at `$AITER_SRC` (default `/home/danyzhan/aiter`). Conv3d is loaded from here when the image copy has no `conv3d`. |
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
export LUMEN_ENABLE=hf_attn_patch,lumen_linear,aiter_conv
```

Each name is one `LumenConfig.enable` call. Empty, `0`, `false`, `none`, and `off` turn patching off.

| Flag | What it changes | Fallback |
|---|---|---|
| `hf_attn_patch` | `F.scaled_dot_product_attention` goes through Lumen attention. A bool keep-mask (`True` = attend) is converted to an additive bias, expanded so size-1 axes have stride 0, and a per-batch mask is launched one batch at a time (the Triton kernel ignores the batch stride). | Extra kwargs, a kernel error, or a masked call that needs a gradient: the original SDPA. Lumen's backward does not apply the bias, and those gradients overflow to inf in bf16. |
| `lumen_linear` | `nn.Linear` forward uses the tuned bf16 GEMM (`dispatch_gemm`, `scaling="none"`). Backward is a torch matmul autograd wrapper. LoRA `A`/`B` are parameters, not `nn.Linear`, so the same GEMM is installed on `LoRALinear.forward`. | If the flag fails to load, LoRA `A`/`B` stay on torch matmul. |
| `aiter_conv` | `nn.Conv2d` and `nn.Conv3d` forward use AITER. `CausalConv3d.forward` (the causal pad) is kept; only `_conv_forward` is rebound. This covers the Wan2.2 VAE (`WanVideoVAE38`, 48 latent channels). | Unsupported dtype or groups, or a kernel error, use the original conv. Conv3d is imported from `$AITER_SRC` when the installed aiter has no `conv3d`. |

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
- aiter with `conv3d`. `AITER_SRC` defaults to `/home/danyzhan/aiter`.

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
