# Isaac Lab 2.3 benchmarks on MI350X

The four [Isaac Lab 2.3 performance-benchmark](https://docs.nvidia.com/isaaclab/2.3/overview/performance-benchmarks) cases, run on one AMD Instinct MI350X. Physics is hipPhysics. The Cartpole RGB camera is rendered either by hipPhysics or by [hipRT_render](https://github.com/ZhangDanyang-AMD/hipRT_render). Training goes through Lumen-RL's `VecEnvPPOTrainer` (`python -m lumenrl.trainer.main`), not Isaac Lab's RL-Games or RSL-RL.

Each cell is three effective FPS numbers, `num_envs * control steps / wall time`:

1. environment step, random actions in `[-1, 1]`
2. environment step and inference, one policy forward plus a sampled action
3. environment step, inference, and train, `horizon` collected steps plus one PPO update, amortized

The RTX 4090 column is the published Isaac Lab 2.3 single-GPU table (RL-Games for Cartpole and Shadow, RSL-RL for G1).

## Results

| Case | Envs | Isaac RTX 4090 | MI350X, this run | vs 4090 |
|---|---:|---|---|---|
| Cartpole-Direct | 4096 | 1.10M / 910k / 510k | 2.49M / 1.87M / 685k | 2.3× / 2.1× / 1.3× |
| Cartpole-RGB, hipPhysics renderer | 1024 | 50k / 45k / 32k | 651k / 380k / 86k | 13× / 8.4× / 2.7× |
| Cartpole-RGB, hipRT_render | 1024 | 50k / 45k / 32k | 132k / 115k / 57k | 2.6× / 2.6× / 1.8× |
| Repose-Cube-Shadow | 8192 | 200k / 190k / 170k | 129k / 130k / 125k | 0.65× / 0.68× / 0.74× |
| Velocity-Rough-G1 | 4096 | 94k / 88k / 82k | 33k / 34k / 33k | 0.35× / 0.39× / 0.41× |

Cartpole and the RGB camera task are faster than the 4090 row on every column. Shadow and G1 are slower. Those two are hipPhysics ports of the Isaac tasks (see the approximations in `g1_env.py` and `shadow_env.py`), and the PPO value coefficient is Lumen-RL's 0.5 rather than RL-Games' `critic_coef` (4, or 2 for the camera task) or RSL-RL's 1.0. The other hyperparameters follow the Isaac Lab v2.3.0 agent configs named in `configs/`.

## Run

Inside the `wm-rl` container, which has the ROCm torch and the hipPhysics build. `HP_REPO` (default `/work/hipphysics/src`) must expose `python/hipphysics_lab`. `HP_BUILD` (default `/work/hipphysics/build-libero`) must contain `libhipgl.so`. Import torch before hipphysics.

```bash
# Cartpole-Direct, 4096 envs. Writes /work/runs/isaaclab_bench/cartpole_direct_4096.json
bash examples/Embodied_IsaacLab_Benchmark/run_bench_cartpole.sh

# Cartpole RGB, 1024 envs. renderer is hipphysics or hiprt.
CONFIG=examples/Embodied_IsaacLab_Benchmark/configs/cartpole_rgb.yaml \
  bash examples/Embodied_IsaacLab_Benchmark/run_bench_cartpole.sh \
  vec_env.env_kwargs.renderer=hiprt

# Shadow Hand in-hand repose, 8192 envs. G1 turns on the heightfield collider.
bash examples/Embodied_IsaacLab_Benchmark/run_bench_shadow.sh
bash examples/Embodied_IsaacLab_Benchmark/run_bench_g1.sh
```

`bench.py` builds the same `VecEnvPPOTrainer` as the training entry point and only times it. Extra `key=value` arguments override the YAML. `train` runs the full schedule through `lumenrl.trainer.main`:

```bash
bash examples/Embodied_IsaacLab_Benchmark/run_bench_cartpole.sh train
```

| Task | Config | Environments | Network |
|---|---|---|---|
| `cartpole` | `configs/cartpole.yaml` | hipphysics_lab Cartpole, 4096 | shared MLP `[32, 32]` ELU |
| `cartpole_rgb` | `configs/cartpole_rgb.yaml` | `cartpole_rgb_env.py`, 1024, one 100×100 camera | conv 32/64/64 then MLP `[512]` |
| `shadow` | `configs/shadow.yaml` | `shadow_env.py`, 8192 | shared MLP `[512, 512, 256, 128]` ELU |
| `g1_rough` | `configs/g1_rough.yaml` | `g1_env.py`, 4096 | separate actor and critic `[512, 256, 128]` ELU |

`isaaclab_plugin.py` is the `vec_env.plugin`. It only builds the environment and the network. Rollout, GAE (`gae_vec_env`), the PPO loss, and the KL-adaptive learning rate live in `lumenrl.trainer.vec_env_ppo_trainer`.

The Shadow and G1 MJCF files under `assets/` come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). Mesh files are not vendored; `shadow_model.py` and `g1_model.py` replace the mesh collision geoms. See `assets/menagerie_shadow_hand/PROVENANCE.json`.
