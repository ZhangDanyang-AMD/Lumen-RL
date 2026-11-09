# LIBERO-Spatial world-model GRPO runbook

Reproduce the two committed experiments on this machine: OpenVLA-OFT trained with
GRPO inside a frozen Wan world model, and the same policy inside a frozen
OpenSora world model. Inference and training both stay on plain Hugging Face /
PyTorch. `embodied.lumen` is empty, so Lumen kernels are not part of this result.

Code pin: [Lumen-RL `dev/world-model-rl`](https://github.com/ZhangDanyang-AMD/Lumen-RL/tree/dev/world-model-rl)
at `6e8ba3e` (training code is `9f44cd6`). The checkout mounted into the
container is `/src/Lumen-RL`.

Recorded result on 8x MI350X (gfx950), training time only:

| Run | GPUs | Steps | Wall time | Greedy success, 128 episodes, seed 1234 |
|---|---|---|---|---|
| Wan + OpenVLA-OFT | 6 | 80 | about 3.0 h | SFT 61.7% → step 40 78.9% → step 80 89.1% |
| OpenSora + OpenVLA-OFT | 8 | 40 | about 3.0 h | SFT 80.5% → step 10/20/30/40 = 81.2 / 85.2 / 86.7 / 86.7% |

A 2-GPU OpenSora run was also tried (`/work/runs/os_grpo`). It is not this
result: too few groups, KL climbed, success fell. Do not use it.

## 1. Container

The running container is `wm-rl`, image `lumen-deepspeed:rocm714-aiter-main`,
network host, 256 GB shm:

```bash
docker exec -it wm-rl bash
```

Mounts: `/home/amd/danyzhan` → `/src`, `/mnt/qwen-perf/wm-rl` → `/work`.
Python is `/opt/venv` (torch 2.12.0+rocm7.14.1, transformers 5.21.0, timm 1.0.31,
`diffsynth` already installed).

If the container is gone, start an equivalent one before the commands below:

```bash
docker run -d --name wm-rl --network host --shm-size=256g \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v /home/amd/danyzhan:/src -v /mnt/qwen-perf/wm-rl:/work \
  --entrypoint sleep lumen-deepspeed:rocm714-aiter-main infinity
```

`--device` flags are only needed when creating a new container. The existing
`wm-rl` already sees the GPUs.

Video export needs one extra package (the training run does not):

```bash
docker exec wm-rl /opt/venv/bin/pip install imageio-ffmpeg
```

## 2. Weights and init data

Already on disk. Configs point at the container paths.

| Role | Path | What must be there |
|---|---|---|
| Policy, both runs | `/work/models/Openvla-oft-SFT-libero-spatial-traj1` (~15 GB) | `config.json`, `dataset_statistics.json`, `modeling_prismatic.py`, `model-0000{1..4}-of-00004.safetensors` |
| Wan world model | `/work/models/RLinf-Wan-LIBERO-Spatial` (~14 GB) | `model-00001.safetensors`, `Wan2.2_VAE.pth`, `resnet_rm.pth`, `dataset/*.npy` (692 files) |
| OpenSora world model | `/work/models/RLinf-OpenSora-LIBERO-Spatial` (~3.2 GB) | `config.json`, `model.safetensors.index.json`, `model-00001.safetensors`, `dataset_statistics.json`, `vae/` (diffusers SDXL VAE), `resnet_rm.pth`, `dataset/*.npy` (400 files) |

These are the published weight snapshots. The trainer reads init states from
`<model_dir>/dataset` and does not download anything at runtime.

## 3. Preflight

```bash
docker exec wm-rl bash -lc '
  cd /src/Lumen-RL && git fetch origin && git checkout dev/world-model-rl && git rev-parse HEAD
  test -f /work/models/Openvla-oft-SFT-libero-spatial-traj1/model.safetensors.index.json
  test -f /work/models/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth
  test -f /work/models/RLinf-OpenSora-LIBERO-Spatial/vae/config.json
  python -c "import diffsynth, timm, einops, safetensors, cv2"
  df -h / /mnt/qwen-perf | awk "NR==1 || /\\//"
  rocm-smi --showmeminfo vram | head
'
```

`HEAD` must be `6e8ba3e` (or any descendant that still contains `9f44cd6`).
`/` has only about 1–2 GB free. Checkpoints are ~14 GB each, so `log_dir` and
`save_dir` must stay under `/work`. Do not leave them at the yaml default
`results/...`, which lands on `/`.

One job owns the cards it is given. Do not start Wan and OpenSora together.

`hipblasLtMatmul ... HIPBLAS_STATUS_INTERNAL_ERROR ... recover by calling cublas`
in the log is the existing ROCm fallback. Training continues.

## 4. Wan + OpenVLA-OFT

The yaml default is 100 steps and `save_interval: 0`. The recorded run is the
command below: 6 GPUs, 80 steps, a checkpoint every 20 steps. Six cards is part
of the result (24 groups/step = 6 ranks × `groups_per_rank: 4`). Eight cards
would be a different run.

```bash
docker exec -d wm-rl bash -lc '
  cd /src/Lumen-RL
  CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 \
  NPROC=6 MASTER_PORT=29500 \
  bash examples/Embodied_Wan_OpenVLA_OFT_LIBERO/run_embodied_grpo.sh \
    num_training_steps=80 \
    embodied.save_interval=20 \
    embodied.log_dir=/work/runs/wan_grpo_repro \
    > /work/runs/wan_grpo_repro.log 2>&1
'
```

Use a new `log_dir` so the existing `/work/runs/wan_grpo` checkpoints are left
alone. Any six free devices work; set `CUDA_VISIBLE_DEVICES` to those six and
keep `NPROC=6`.

Done when the log prints `LumenRL finished` and this file exists:

```text
/work/runs/wan_grpo_repro/checkpoints/step_80/policy.safetensors
```

Per-step metrics append to `embodied_metrics.jsonl` in that directory
(`rollout/success_rate`, `train/approx_kl`, `timing/step_s`). The first steps
on the recorded run sat near 0.4–0.6 success; the last ten averaged about 0.78.
Wall time was about 3.0 h, about 134 s/step after warmup.

## 5. OpenSora + OpenVLA-OFT

Same policy, 8 GPUs, 40 steps, a checkpoint every 10 steps. This run needs all
eight cards (32 groups/step). The 2-GPU setting diverges.

```bash
docker exec -d wm-rl bash -lc '
  cd /src/Lumen-RL
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NPROC=8 MASTER_PORT=29640 \
  bash examples/Embodied_OpenSora_OpenVLA_OFT_LIBERO/run_embodied_grpo.sh \
    num_training_steps=40 \
    embodied.save_interval=10 \
    embodied.log_dir=/work/runs/os_grpo8_repro \
    > /work/runs/os_grpo8_repro.log 2>&1
'
```

`MASTER_PORT=29640` is only to stay off 29500. Any free port is fine.

Done when the log prints `LumenRL finished` and this file exists:

```text
/work/runs/os_grpo8_repro/checkpoints/step_40/policy.safetensors
```

Recorded training success stayed about 0.75–0.95, KL about 0.01. Wall time was
about 3.0 h, about 274 s/step.

## 6. Greedy eval and the comparison video

Single process, one GPU. `--policy base` is the SFT checkpoint in the yaml.
Every policy sees the same init states and the same world-model seed.

Table numbers (128 episodes, no video):

```bash
docker exec wm-rl bash -lc '
  cd /src/Lumen-RL
  CUDA_VISIBLE_DEVICES=0 python -m lumenrl.trainer.embodied_eval \
    --config examples/Embodied_Wan_OpenVLA_OFT_LIBERO/configs/embodied_wan_openvla_oft_grpo.yaml \
    --policy base \
    --policy /work/runs/wan_grpo_repro/checkpoints/step_40/policy.safetensors \
    --policy /work/runs/wan_grpo_repro/checkpoints/step_80/policy.safetensors \
    --label SFT --label "GRPO step 40" --label "GRPO step 80" \
    --num-episodes 128 --batch 64 --greedy --seed 1234 \
    --out /work/runs/wan_grpo_repro/eval_128.json
'
```

OpenSora is the same command with the OpenSora config and
`step_{10,20,30,40}` checkpoints.

Comparison video (64 episodes, 4 rows, 20 fps). Rows are ordered so episodes
the last policy newly solves come first. Green border is success, red is fail.

```bash
docker exec wm-rl bash -lc '
  cd /src/Lumen-RL
  CUDA_VISIBLE_DEVICES=0 python -m lumenrl.trainer.embodied_eval \
    --config examples/Embodied_Wan_OpenVLA_OFT_LIBERO/configs/embodied_wan_openvla_oft_grpo.yaml \
    --policy base \
    --policy /work/runs/wan_grpo_repro/checkpoints/step_80/policy.safetensors \
    --label SFT --label "GRPO step 80" \
    --num-episodes 64 --batch 64 --greedy --seed 1234 \
    --video /work/demo/wan_sft_vs_grpo_repro.mp4 --video-episodes 4 --fps 20 \
    --out /work/runs/wan_grpo_repro/eval_64.json
'
```

Swap the config and the step-40 checkpoint for the OpenSora video. The 64-episode
subset on the recorded checkpoints was Wan 60.9% → 89.1% and OpenSora 84.4% → 89.1%.
That is a different sample from the 128-episode table.

## 7. What is intentionally off

- `embodied.lumen` stays `{}`. Turning on `actor` / `rollout` / `world_model`
  kernel flags is a different experiment and will not match the table.
- `third_party/Lumen` does not need `git submodule update` for this run.
  `apply_lumen` returns immediately when the flag dict is empty.
- Training temperature is 1.6 (`embodied.rollout.temperature`). The table and
  the videos are `--greedy`.
- Train seed is 42. Eval seed is 1234. Both are required for the same init
  order and the same reported rates.
