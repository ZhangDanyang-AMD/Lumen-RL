"""Batched LIBERO environment on hipPhysics: physics, OSC control and cameras run on the GPU.

LIBERO (robosuite + MuJoCo) still builds every task scene; ``hipphysics_libero`` then runs
``worlds_per_lane`` worlds of one task on one hipPhysics engine and renders them with the
hip ray tracer. A lane is built the way hipPhysics' LIBERO eval builds one
(``eval/stack/eval_full.py``: one soft-reset reference env per lane, arm reset with the
episode's fixture poses, ``set_init_state``, fixture receipt check) and every lane round
steps all of its live worlds in one ``step_batch`` call. Finished or padded worlds are
neither simulated nor rendered.

Lanes are cached per task for the life of the env: rebuilding an engine does not give its
device memory back. Success comes from the hipPhysics success bundles
(``bundle_<suite>_t<task>.json``), so only tasks that have one can run (the four LIBERO
eval suites: spatial, object, goal, 10).

The interface is ``LiberoEnv``'s: ``reset(inits)``, ``step(actions [B, chunk, 7])``,
``observation()``, ``frames``, ``done``, ``success``, ``elapsed``, ``timing``.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import torch

from lumenrl.envs.libero_env import (
    CHUNK, WAIT_ACTION, LiberoInit, _libero, _agentview, suite_task_origins, suite_tasks,
)

logger = logging.getLogger(__name__)


def _import_hipphysics(hip_repo: str) -> None:
    path = os.path.join(hip_repo, "python")
    if path not in sys.path:
        sys.path.insert(0, path)
    os.environ.setdefault("HP_GPU_RESIDENT", "1")
    os.environ.setdefault("HP_RENDERER", "rt")
    if not os.environ.get("HP_RT_AA"):
        # MuJoCo EGL look (msaa4 + hw mips), the frames LIBERO's demonstrations were rendered with.
        os.environ.setdefault("HP_RENDER_PRESET", "classic_gpugl")


class _Lane:
    """``n`` worlds of one LIBERO task on one hipPhysics engine."""

    def __init__(self, suite: str, task_id: int, task, n: int, seed: int, hip_repo: str,
                 bundle_dir: str, gpu_id: int, resolution: int) -> None:
        _, get_libero_path, offscreen = _libero()
        from hipphysics_libero.gpu_resident_env import GpuResidentLiberoEnv, _RoundEngineView

        self.suite, self.task_id, self.n = suite, int(task_id), int(n)
        self._view = _RoundEngineView
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        with open(os.path.join(bundle_dir, f"bundle_{suite}_t{self.task_id}.json")) as f:
            meta = json.load(f)
        if not meta.get("goal_state"):
            raise RuntimeError(f"hipPhysics bundle for {suite} t{task_id} has no goal_state")
        ref = offscreen(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
        ref.seed(seed)
        # Soft resets keep the model; LIBERO's visualize toggles some site alphas, so every
        # episode restores the constructor's site colours first.
        ref.env.hard_reset = False
        self._rgba0 = np.asarray(ref.env.sim.model.site_rgba).copy()
        # The facade replaces every camera image with a hipPhysics frame.
        for ob in ref.env._observables.values():
            if ob.modality == "image":
                ob.set_enabled(False)
        self.ref = ref
        self.facades, shared = [], None
        for k in range(self.n):
            f = GpuResidentLiberoEnv(ref_env=ref, hip_repo=hip_repo, gpu_id=gpu_id, resolution=resolution,
                                     success_metadata=meta, env_slot=k, num_envs=self.n, shared=shared)
            shared = f.shared
            self.facades.append(f)
        self.env = self.facades[0]._env
        if not (self.env._use_gpu_osc and self.env._gpu_osc_configured
                and getattr(self.env, "_mj_ctrl_fn", None) is None
                and not getattr(self.env, "_use_mj_stepping", False)):
            raise RuntimeError(f"{suite} t{task_id}: hipPhysics lane is not on the GPU-OSC path")
        self.coord = self.facades[0]._visual
        self.checker = self.env._success_checker
        if self.checker is None:
            raise RuntimeError(f"{suite} t{task_id}: hipPhysics lane has no success checker")
        self._joined: set[int] = set()

    def start_episode(self, k: int, trial: int, init_state: np.ndarray) -> None:
        from hipphysics_libero.fixture_coherence import arm_reset

        f = self.facades[k]
        if k in self._joined:
            f._barrier.leave(k)
        self.ref.env.sim.model.site_rgba[:] = self._rgba0
        arm_reset(f, self.suite, self.task_id, int(trial))
        f.reset()
        f.set_init_state(init_state.copy())
        rc = f._fixture_restore_receipt
        want = "historical-fixture-pose-publication-v1" + ("-per-world" if self.n > 1 else "")
        if (rc is None or rc["schema"] != want or rc["original_episode_id"] != int(trial)
                or (self.n > 1 and rc["world"] != k)):
            raise RuntimeError(f"{self.suite} t{self.task_id} world {k}: fixture receipt {rc} != {want}")
        f._barrier.rejoin(k)
        self._joined.add(k)

    def step(self, actions: np.ndarray, live: list[int], render: list[int], check: bool):
        """One round for rows ``live``; returns ({row: agent-view frame}, {row: success})."""
        A = np.tile(WAIT_ACTION, (self.n, 1))
        A[live] = actions
        step_rows = live if len(live) < self.n else None
        # step_all always draws at least one row; a frame nobody reads costs one render.
        render_rows = render or live[:1]
        with self.coord.transaction():
            kw = {"render_rows": render_rows}
            if step_rows is not None:
                kw["step_rows"] = step_rows
            raw, _, _, _ = self.coord.step_batch(A.astype(np.float32), **kw)
            frames = {k: _agentview(self.facades[k]._merge(raw[k])) for k in render}
            success = {}
            if check:
                view = self._view(self.env.engine)
                success = {k: bool(self.checker.check(view, k)) for k in live}
        return frames, success

    def close(self) -> None:
        from hipphysics_libero.wave import release_wave

        release_wave(self.facades)
        self.facades = []


class HipPhysicsLiberoEnv:
    """``LiberoEnv`` on hipPhysics lanes. Slots of one task share lanes of ``worlds_per_lane``."""

    chunk = CHUNK

    def __init__(
        self,
        task_suite: str,
        max_episode_steps: int,
        camera_height: int = 256,
        camera_width: int = 256,
        num_steps_wait: int = 10,
        seed: int = 0,
        hip_repo: str = "",
        bundle_dir: str = "",
        gpu_id: int = 0,
        worlds_per_lane: int = 8,
        record_frames: bool = False,
    ) -> None:
        if camera_height != camera_width:
            raise ValueError("hipphysics_libero renders square cameras")
        if num_steps_wait < 1:
            raise ValueError("hipphysics_libero renders the first frame on the last settle step: num_steps_wait >= 1")
        self.hip_repo = hip_repo or os.environ.get("HP_REPO", "")
        if not self.hip_repo or not os.path.isdir(os.path.join(self.hip_repo, "python", "hipphysics_libero")):
            raise ValueError(f"embodied.env.hip_repo / HP_REPO must point at a hipPhysics tree, got {self.hip_repo!r}")
        self.bundle_dir = (bundle_dir or os.environ.get("GPU_RESIDENT_BUNDLE_DIR")
                           or os.path.join(self.hip_repo, "eval", "bundles"))
        _import_hipphysics(self.hip_repo)
        self.tasks = suite_tasks(task_suite)
        self.origins = suite_task_origins(task_suite)
        missing = [f"{s}/t{t}" for s, t in self.origins
                   if not os.path.isfile(os.path.join(self.bundle_dir, f"bundle_{s}_t{t}.json"))]
        if missing:
            raise ValueError(
                f"{task_suite}: {len(missing)} of {len(self.origins)} tasks have no hipPhysics success bundle "
                f"in {self.bundle_dir} (e.g. {missing[:3]}); use libero_spatial/object/goal/10 or the libero backend"
            )
        self.max_episode_steps = int(max_episode_steps)
        self.resolution = int(camera_height)
        self.num_steps_wait = int(num_steps_wait)
        self.seed = int(seed)
        self.gpu_id = int(gpu_id)
        self.worlds_per_lane = int(worlds_per_lane)
        self.device = torch.device("cpu")
        self.record_frames = record_frames
        self.history: list[list[torch.Tensor]] = []
        self.timing = {"reset_s": 0.0, "sim_s": 0.0, "sim_steps": 0}
        self._lanes: dict[int, list[_Lane]] = {}
        self._slots: list[tuple[_Lane, int]] = []
        self.num_envs = 0
        self.frames: torch.Tensor | None = None
        self.elapsed: torch.Tensor | None = None
        self.done: torch.Tensor | None = None
        self.success: torch.Tensor | None = None
        self.instructions: list[str] = []

    def _lane(self, task_id: int, j: int) -> _Lane:
        lanes = self._lanes.setdefault(task_id, [])
        while len(lanes) <= j:
            suite, local = self.origins[task_id]
            t0 = time.perf_counter()
            lanes.append(_Lane(suite, local, self.tasks[task_id], self.worlds_per_lane,
                               self.seed + 1000 * sum(len(v) for v in self._lanes.values()),
                               self.hip_repo, self.bundle_dir, self.gpu_id, self.resolution))
            logger.info("hipPhysics lane %s t%d #%d built in %.1fs (%d lanes cached)",
                        suite, local, len(lanes) - 1, time.perf_counter() - t0,
                        sum(len(v) for v in self._lanes.values()))
        return lanes[j]

    def _by_lane(self, slots) -> dict[int, tuple[_Lane, list[int], list[int]]]:
        """{id(lane): (lane, slot indices, lane rows)} for ``slots``."""
        out: dict[int, tuple[_Lane, list[int], list[int]]] = {}
        for i in slots:
            lane, row = self._slots[i]
            entry = out.setdefault(id(lane), (lane, [], []))
            entry[1].append(i)
            entry[2].append(row)
        return out

    def close(self) -> None:
        for lanes in self._lanes.values():
            for lane in lanes:
                lane.close()
        self._lanes, self._slots = {}, []

    def reset(self, inits: list[LiberoInit]) -> dict:
        t0 = time.perf_counter()
        n = len(inits)
        by_task: dict[int, list[int]] = {}
        for i, init in enumerate(inits):
            if init.trial < 0:
                raise ValueError(f"{init.source}: hipphysics_libero needs the init state's trial index")
            by_task.setdefault(int(init.task_id), []).append(i)
        self._slots = [None] * n  # type: ignore[list-item]
        L = self.worlds_per_lane
        for task_id, idx in by_task.items():
            for j in range(0, len(idx), L):
                lane = self._lane(task_id, j // L)
                for row, i in enumerate(idx[j:j + L]):
                    self._slots[i] = (lane, row)
                    lane.start_episode(row, inits[i].trial, inits[i].init_state)
        images: list[np.ndarray | None] = [None] * n
        wait = WAIT_ACTION[None]
        for lane, slots, rows in self._by_lane(range(n)).values():
            for s in range(self.num_steps_wait):
                last = s == self.num_steps_wait - 1
                frames, _ = lane.step(np.repeat(wait, len(rows), 0), rows, rows if last else [], check=False)
                if last:
                    for i, row in zip(slots, rows):
                        images[i] = frames[row]
        self.num_envs = n
        self.instructions = [init.instruction for init in inits]
        self.frames = torch.from_numpy(np.stack(images))
        self.elapsed = torch.zeros(n, dtype=torch.long)
        self.done = torch.zeros(n, dtype=torch.bool)
        self.success = torch.zeros(n, dtype=torch.bool)
        if self.record_frames:
            self.history = [[self.frames[i:i + 1].clone()] for i in range(n)]
        self.timing = {"reset_s": time.perf_counter() - t0, "sim_s": 0.0, "sim_steps": 0}
        return self.observation()

    def observation(self) -> dict:
        return {"images": self.frames, "instructions": self.instructions}

    def step(self, actions: torch.Tensor):
        assert self.frames is not None and self.done is not None and self.elapsed is not None
        acts = np.clip(actions.detach().to("cpu", torch.float64).numpy()[:, : self.chunk], -1.0, 1.0)
        reward = torch.zeros(self.num_envs, dtype=torch.float32)
        active = (~self.done).nonzero(as_tuple=False).flatten().tolist()
        steps = {i: min(self.chunk, self.max_episode_steps - int(self.elapsed[i])) for i in active}
        stopped: set[int] = {i for i in active if steps[i] <= 0}
        recorded: dict[int, list[np.ndarray]] = {i: [] for i in active}
        t0 = time.perf_counter()
        for lane, slots, rows in self._by_lane(active).values():
            for t in range(self.chunk):
                live = [(i, r) for i, r in zip(slots, rows) if i not in stopped and t < steps[i]]
                if not live:
                    break
                render = [r for i, r in live if self.record_frames or t == steps[i] - 1]
                frames, success = lane.step(acts[[i for i, _ in live], t], [r for _, r in live], render, check=True)
                self.timing["sim_steps"] += len(live)
                for i, r in live:
                    self.elapsed[i] += 1
                    if r in frames:
                        self.frames[i] = torch.from_numpy(frames[r])
                        if self.record_frames:
                            recorded[i].append(frames[r])
                    if success[r]:
                        self.success[i] = True
                        reward[i] = 1.0
                        stopped.add(i)
        self.timing["sim_s"] += time.perf_counter() - t0
        for i in active:
            if self.record_frames and recorded[i]:
                self.history[i].append(torch.from_numpy(np.stack(recorded[i])))
            if bool(self.success[i]) or steps[i] <= 0 or int(self.elapsed[i]) >= self.max_episode_steps:
                self.done[i] = True
        return self.observation(), reward, self.done.clone(), {}
