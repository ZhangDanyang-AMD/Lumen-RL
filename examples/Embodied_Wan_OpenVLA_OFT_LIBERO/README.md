# Wan + OpenVLA-OFT GRPO: evaluation results

Policy: `Openvla-oft-SFT-libero-spatial-traj1`. World model: `RLinf-Wan-LIBERO-Spatial`.
Trained and evaluated on 8x MI350X unless noted. Lumen kernels are off.

RLinf's published Wan table is scored inside the Wan simulator, not in the
LIBERO physics simulator. The two are reported separately below.

## Real LIBERO

LIBERO-Spatial, 500 episodes, 10 tasks, step cap 512, seed 1234.
Sampled rows use temperature 1.6. The same init states are used for every row
in a given run. Object and Goal are 0% for both the SFT policy and the RL
checkpoint, because this SFT checkpoint was trained on Spatial only.

| run | reward | SFT | step 10 | step 20 | step 30 | step 40 | step 50 | step 80 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v1, 6 GPU, 192 episodes/step, lr 5e-6 | episode success | 64.2% | | | | | | 62.4% |
| v2, 1024 episodes/step, lr 2e-5 | episode success, sample | 64.2% | 66.6% | 67.6% | 68.2% | 58.6% | 64.4% | |
| v2 | episode success, greedy | | | | | 64.2% | 67.6% | |
| v3, same scale as v2 | relative reward, sample | 64.2% | 64.0% | 68.2% | 68.2% | 67.6% | 68.2% | |
| v3 | relative reward, greedy | | | | | 70.6% | 67.4% | |

A binomial standard error on 500 episodes is about 2 points, so a difference of
about 3 points is one standard error of the difference. v3 step 40 greedy,
70.6%, is the highest real-LIBERO number. v2 step 40 sample falls below the
SFT policy; v3 does not.

v1 config: `configs/embodied_wan_openvla_oft_grpo.yaml`.
v2 config: `configs/embodied_wan_openvla_oft_grpo_rlinf.yaml`.
v3 config: `configs/embodied_wan_openvla_oft_grpo_rlinf_v3.yaml`.
v3 uses RLinf's per-frame relative reward (`5 * r_t - r_{t-1}`) and drops a
group unless its mean return is in `[0.5, 4.5]`.

## Wan world model

Same decoding and seed as the real-LIBERO sample rows. 500 episodes drawn from
the world model's 692 init states, step cap 512. Only v3 was evaluated this way.
The success flag is the reward classifier, which is also the training objective.

| policy | success |
|---|---:|
| SFT | 68.2% |
| v3 step 10 | 81.4% |
| v3 step 20 | 90.0% |
| v3 step 30 | 91.8% |
| v3 step 40 | 92.4% |
| v3 step 50 | 92.2% |

An earlier greedy check of the v1 run, 128 episodes, seed 1234, was
61.7% (SFT) → 78.9% (step 40) → 89.1% (step 80). That protocol is not the
500-episode temperature-1.6 table above.

## RLinf's published Wan table

From the Wan world-model page. Both SFT and RL are sampled at temperature 1.6.
The page reports 1500 environments across Spatial, Object, and Goal. The SFT
column is the LoRA-base policy, not `Openvla-oft-SFT-libero-spatial-traj1`.

| model | Spatial | Object | Goal |
|---|---:|---:|---:|
| OpenVLA-OFT (LoRA-base) | 61.2% | 36.7% | 48.2% |
| OpenVLA-OFT (RLinf-GRPO, Wan simulator) | 77.5% | 77.9% | 60.1% |

Our world-model Spatial numbers are not that table: the starting policy is the
Spatial traj1 SFT (68.2% here, 61.2% in the doc), the trial count is 500 rather
than their 1500 across three suites, and Object and Goal were not trained.

## Artifacts

| what | path |
|---|---|
| v1 checkpoints and real-LIBERO eval | `/work/runs/wan_grpo/` |
| v2 checkpoints and real-LIBERO eval | `/work/runs/wan_grpo_v2/` |
| v3 checkpoints, real-LIBERO eval, world-model eval | `/work/runs/wan_grpo_v3/` |
