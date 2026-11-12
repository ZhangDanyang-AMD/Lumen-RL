# OpenSora + OpenVLA-OFT GRPO: evaluation results

Policy: `Openvla-oft-SFT-libero-spatial-traj1`. World model: `RLinf-OpenSora-LIBERO-Spatial`.
The recorded run is 8x MI350X, 40 steps, about 192 episodes per step, learning
rate 5e-6, episode-success reward. Config:
`configs/embodied_opensora_openvla_oft_grpo.yaml`. Lumen kernels are off.
Checkpoints: `/work/runs/os_grpo8/checkpoints/step_{10,20,30,40}`.

RLinf's published OpenSora table is scored inside the OpenSora simulator, not
in the LIBERO physics simulator. The two are reported separately below.

## Real LIBERO

LIBERO-Spatial, 500 episodes, 10 tasks, step cap 512, seed 1234.
SFT is greedy. The RL checkpoint is sampled at temperature 1.6. That is the
decoding pair used in RLinf's OpenSora write-up. Only step 40 was evaluated.
Object was not evaluated.

| policy | decoding | success |
|---|---|---:|
| SFT | greedy | 58.6% |
| step 40 | temperature 1.6 | 64.0% |

The gain is 5.4 points. A binomial standard error on 500 episodes is about
2.2 points, so the difference is about 1.7 standard errors. Eval output:
`/work/runs/os_grpo8/eval_libero/`.

## OpenSora world model

Greedy, 128 episodes, seed 1234, from the training runbook. This is not the
500-episode temperature-1.6 protocol used for the real-LIBERO table, and it
was not rerun for this README.

| policy | greedy success |
|---|---:|
| SFT | 80.5% |
| step 10 | 81.2% |
| step 20 | 85.2% |
| step 30 | 86.7% |
| step 40 | 86.7% |

A 2-GPU run (`/work/runs/os_grpo`) is not this result: too few groups, KL
climbed, and success fell.

## RLinf's published OpenSora table

From the OpenSora world-model page. 500 environments per suite (10 tasks x 50
trials). SFT is greedy. RL is sampled at temperature 1.6. The SFT column is
the LoRA-base policy, not `Openvla-oft-SFT-libero-spatial-traj1`.

| model | Spatial | Object |
|---|---:|---:|
| OpenVLA-OFT (LoRA-base) | 51.61% | 50.20% |
| OpenVLA-OFT (RLinf-GRPO, OpenSora simulator) | 64.5% | 75.5% |

Our 58.6% → 64.0% numbers are real LIBERO, so they do not sit on this table.
The world-model greedy check starts at 80.5%, well above the doc's 51.61%,
because the policy and the decoding both differ.
