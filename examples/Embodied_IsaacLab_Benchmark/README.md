# Isaac Lab 2.3 benchmarks on MI350X

The four [Isaac Lab 2.3 performance-benchmark](https://docs.nvidia.com/isaaclab/2.3/overview/performance-benchmarks) cases, run on AMD Instinct MI350X with one, four and eight GPUs. Physics is hipPhysics. The Cartpole RGB camera is rendered either by hipPhysics or by hipRT_render. Training goes through Lumen-RL's `VecEnvPPOTrainer` (`python -m lumenrl.trainer.main`), not Isaac Lab's RL-Games or RSL-RL.

Each FPS cell has three numbers, `num_envs * control steps / wall time`:

1. environment step, random actions in `[-1, 1]`
2. environment step and inference, one policy forward plus a sampled action
3. environment step, inference, and train, `horizon` collected steps plus one PPO update, amortized

VRAM is the device-wide memory in use on each GPU at the end of the run (includes the CUDA context and the hipPhysics / HIPRT allocations).

## Results

Multi-GPU runs are weak scaling: every GPU runs its own environments (the per-GPU count is the same as the single-GPU row), and the FPS is the sum over GPUs.

### Cartpole-Direct, 4096 envs per GPU

| Platform | GPUs | FPS | VRAM per GPU (GB) | MI350X vs L40, same GPU count |
|---|---:|---|---:|---|
| Isaac Lab, L40 | 1 | 620k / 490k / 260k | not published | |
| Isaac Lab, L40 (single node) | 4 | 2.70M / 2.10M / 950k | not published | |
| MI350X | 1 | 2.64M / 1.88M / 705k | 5.40 | 4.3× / 3.8× / 2.7× |
| MI350X | 4 | 10.2M / 6.37M / 2.30M | 7.77 | 3.8× / 3.0× / 2.4× |
| MI350X | 8 | 19.8M / 12.5M / 4.54M | 8.85 | |

### Cartpole-RGB-Camera, hipPhysics renderer, 1024 envs per GPU

| Platform | GPUs | FPS | VRAM per GPU (GB) | MI350X vs L40, same GPU count |
|---|---:|---|---:|---|
| Isaac Lab, L40 | 1 | 30k / 28k / 21k | not published | |
| Isaac Lab, L40 (single node) | 4 | 130k / 120k / 90k | not published | |
| MI350X | 1 | 663k / 375k / 88k | 7.81 | 22× / 13× / 4.2× |
| MI350X | 4 | 2.69M / 1.49M / 332k | 10.92 | 21× / 12× / 3.7× |
| MI350X | 8 | 5.12M / 2.93M / 661k | 11.97 | |

### Cartpole-RGB-Camera, hipRT_render, 1024 envs per GPU

| Platform | GPUs | FPS | VRAM per GPU (GB) | MI350X vs L40, same GPU count |
|---|---:|---|---:|---|
| Isaac Lab, L40 | 1 | 30k / 28k / 21k | not published | |
| Isaac Lab, L40 (single node) | 4 | 130k / 120k / 90k | not published | |
| MI350X | 1 | 132k / 114k / 58k | 8.21 | 4.4× / 4.1× / 2.7× |
| MI350X | 4 | 473k / 411k / 212k | 10.80 | 3.6× / 3.4× / 2.4× |
| MI350X | 8 | 824k / 736k / 395k | 11.85 | |

### Repose-Cube-Shadow, 8192 envs per GPU

| Platform | GPUs | FPS | VRAM per GPU (GB) | MI350X vs L40, same GPU count |
|---|---:|---|---:|---|
| Isaac Lab, L40 | 1 | 170k / 140k / 120k | not published | |
| Isaac Lab, L40 (single node) | 4 | 440k / 420k / 390k | not published | |
| MI350X | 1 | 130k / 130k / 125k | 27.19 | 0.76× / 0.93× / 1.04× |
| MI350X | 4 | 509k / 499k / 478k | 29.12 | 1.16× / 1.19× / 1.22× |
| MI350X | 8 | 1.01M / 974k / 923k | 31.92 | |

### Velocity-Rough-G1, 4096 envs per GPU

| Platform | GPUs | FPS | VRAM per GPU (GB) | MI350X vs L40, same GPU count |
|---|---:|---|---:|---|
| Isaac Lab, L40 | 1 | 72k / 64k / 62k | not published | |
| Isaac Lab, L40 (single node) | 4 | 290k / 270k / 250k | not published | |
| MI350X | 1 | 33k / 34k / 34k | 16.41 | 0.46× / 0.53× / 0.54× |
| MI350X | 4 | 131k / 135k / 131k | 19.46 | 0.45× / 0.50× / 0.52× |
| MI350X | 8 | 262k / 272k / 257k | 20.51 | |

### VRAM against the Isaac Lab memory table

The Isaac Lab 2.3 page has one Memory Consumption table. It does not name the GPU, and it is the only VRAM figure the page publishes, so the L40 rows above have no VRAM entry. For reference, the MI350X single-GPU VRAM beside those numbers:

| Case | Envs | Isaac Lab page VRAM (GB) | MI350X, 1 GPU (GB) |
|---|---:|---:|---:|
| Cartpole-Direct | 4096 | 3.3 | 5.40 |
| Cartpole-RGB-Camera (hipPhysics) | 1024 | 16.7 | 7.81 |
| Cartpole-RGB-Camera (hipRT) | 1024 | 16.7 | 8.21 |
| Velocity-Rough-G1 | 4096 | 6.1 | 16.41 |
| Repose-Cube-Shadow | 8192 | 6.4 | 27.19 |

### Single GPU against RTX 4090

The Isaac Lab page also publishes single-GPU RTX 4090 numbers (RL-Games for Cartpole and Shadow, RSL-RL for G1). Against those, the MI350X single-GPU rows are:

| Case | Envs | Isaac RTX 4090 | MI350X, 1 GPU | MI350X vs 4090 |
|---|---:|---|---|---|
| Cartpole-Direct | 4096 | 1.10M / 910k / 510k | 2.64M / 1.88M / 705k | 2.4× / 2.1× / 1.4× |
| Cartpole-RGB, hipPhysics renderer | 1024 | 50k / 45k / 32k | 663k / 375k / 88k | 13× / 8.3× / 2.8× |
| Cartpole-RGB, hipRT_render | 1024 | 50k / 45k / 32k | 132k / 114k / 58k | 2.6× / 2.5× / 1.8× |
| Repose-Cube-Shadow | 8192 | 200k / 190k / 170k | 130k / 130k / 125k | 0.65× / 0.68× / 0.74× |
| Velocity-Rough-G1 | 4096 | 94k / 88k / 82k | 33k / 34k / 34k | 0.35× / 0.39× / 0.41× |

Cartpole and the RGB camera rows run faster than Isaac on every column; Shadow and G1 run slower. Shadow and G1 are hipPhysics ports of the Isaac tasks (see the approximations in `g1_env.py` and `shadow_env.py`). The PPO value coefficient is Lumen-RL's 0.5, not RL-Games' `critic_coef` (4, or 2 for the camera task) or RSL-RL's 1.0. The other hyperparameters follow the Isaac Lab v2.3.0 agent configs named in `configs/`.

Eight GPUs reach 6.3–7.9× the single-GPU environment-step rate (Cartpole 7.5×, RGB hipPhysics 7.7×, RGB hipRT 6.3×, Shadow 7.7×, G1 7.9×). The train column scales less (6.4× for Cartpole, 7.5× for RGB hipPhysics, 6.8× for RGB hipRT, 7.7× for G1) because each PPO update all-reduces gradients across GPUs.

## Run

Inside the `wm-rl` container, which has the ROCm torch and the hipPhysics build. `HP_REPO` (default `/work/hipphysics/src`) must expose `python/hipphysics_lab`. `HP_BUILD` (default `/work/hipphysics/build-libero`) must contain `libhipgl.so`. Import torch before hipphysics.

```bash
# One GPU. Writes /work/runs/isaaclab_bench/cartpole_direct_4096.json
bash examples/Embodied_IsaacLab_Benchmark/run_bench_cartpole.sh

# Four or eight GPUs: one process per GPU with torchrun. Writes ..._x4.json / ..._x8.json
HIP_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash examples/Embodied_IsaacLab_Benchmark/run_bench_cartpole.sh

# Cartpole RGB, 1024 envs. renderer is hipphysics or hiprt.
CONFIG=examples/Embodied_IsaacLab_Benchmark/configs/cartpole_rgb.yaml \
  bash examples/Embodied_IsaacLab_Benchmark/run_bench_cartpole.sh \
  vec_env.env_kwargs.renderer=hiprt

# Shadow Hand in-hand repose, 8192 envs. G1 turns on the heightfield collider.
bash examples/Embodied_IsaacLab_Benchmark/run_bench_shadow.sh
bash examples/Embodied_IsaacLab_Benchmark/run_bench_g1.sh
```

`bench.py` builds the same `VecEnvPPOTrainer` as the training entry point and only times it. Extra `key=value` arguments override the YAML. `train` runs the full schedule through `lumenrl.trainer.main`, and `NPROC` works there too:

```bash
bash examples/Embodied_IsaacLab_Benchmark/run_bench_cartpole.sh train
```

| Task | Config | Environments | Network |
|---|---|---|---|
| `cartpole` | `configs/cartpole.yaml` | hipphysics_lab Cartpole, 4096 | shared MLP `[32, 32]` ELU |
| `cartpole_rgb` | `configs/cartpole_rgb.yaml` | `cartpole_rgb_env.py`, 1024, one 100×100 camera | conv 32/64/64 then MLP `[512]` |
| `shadow` | `configs/shadow.yaml` | `shadow_env.py`, 8192 | shared MLP `[512, 512, 256, 128]` ELU |
| `g1_rough` | `configs/g1_rough.yaml` | `g1_env.py`, 4096 | separate actor and critic `[512, 256, 128]` ELU |

`isaaclab_plugin.py` is the `vec_env.plugin`. It only builds the environment and the network. Rollout, GAE (`gae_vec_env`), the PPO loss, the KL-adaptive learning rate and the multi-GPU reductions live in `lumenrl.trainer.vec_env_ppo_trainer`. With several ranks, gradients are averaged after each mini-batch, observation and advantage statistics are pooled across ranks, and rank 0's initial weights are broadcast, so every GPU holds the same policy.

The Shadow and G1 MJCF files under `assets/` come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). Mesh files are not vendored; `shadow_model.py` and `g1_model.py` replace the mesh collision geoms. See `assets/menagerie_shadow_hand/PROVENANCE.json`.
