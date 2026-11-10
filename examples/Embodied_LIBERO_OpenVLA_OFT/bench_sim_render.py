"""Split one LIBERO sim step into physics and CPU (OSMesa) rendering time.

Steps the same task twice from the same init state: once with camera
observations off (physics + controller only) and once with them on. The
difference is the cost of rendering the agent-view and wrist cameras.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \\
    python examples/Embodied_LIBERO_OpenVLA_OFT/bench_sim_render.py --steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def _time_steps(env, init_state, steps: int, rng: np.random.Generator) -> float:
    env.reset()
    env.set_init_state(init_state.copy())
    actions = rng.uniform(-0.3, 0.3, size=(steps, 7)).astype(np.float32)
    actions[:, -1] = -1.0
    t0 = time.perf_counter()
    for a in actions:
        env.step(a)
    return (time.perf_counter() - t0) / steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--size", type=int, action="append", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sizes = args.size or [128, 256]

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    init_state = np.asarray(torch.load(init_path, weights_only=False)[0])

    rows = []
    for size in sizes:
        for camera in (False, True):
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size,
                                     use_camera_obs=camera, horizon=args.steps + 50)
            env.seed(0)
            _time_steps(env, init_state, 20, np.random.default_rng(1))
            per_step = _time_steps(env, init_state, args.steps, np.random.default_rng(0))
            env.close()
            rows.append({"size": size, "camera_obs": camera, "ms_per_step": per_step * 1e3})
            print(f"size={size} camera_obs={camera}: {per_step * 1e3:.2f} ms/step", flush=True)

    for size in sizes:
        off = next(r["ms_per_step"] for r in rows if r["size"] == size and not r["camera_obs"])
        on = next(r["ms_per_step"] for r in rows if r["size"] == size and r["camera_obs"])
        print(f"size={size}: physics {off:.2f} ms, rendering {on - off:.2f} ms ({(on - off) / on:.0%} of a step)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
