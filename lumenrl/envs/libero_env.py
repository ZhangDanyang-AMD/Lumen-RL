"""Batched LIBERO environment. Physics and rendering both run on CPU (OSMesa).

Each slot is one robosuite ``OffScreenRenderEnv``. A step applies one OpenVLA-OFT
action chunk: up to ``chunk`` sim steps, stopping a slot when the task succeeds
or the step budget runs out. The image returned to the policy is the agent-view
camera rotated 180 degrees, matching the OpenVLA / RLinf preprocessing.

With ``num_workers > 0`` every slot lives in its own process so the slots of a
batch step in parallel; otherwise they step one after another in this process.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

CHUNK = 8
WAIT_ACTION = np.zeros(7, dtype=np.float32)
WAIT_ACTION[-1] = -1.0  # LIBERO: -1 opens the gripper


def _use_osmesa() -> None:
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def _libero():
    _use_osmesa()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv


def suite_tasks(task_suite: str) -> list:
    """Tasks of a LIBERO suite. ``libero_130`` is every suite merged, as in RLinf:
    spatial, object, goal, 90, 10 in that order, de-duplicated by task name."""
    benchmark, _, _ = _libero()
    suites = benchmark.get_benchmark_dict()
    if task_suite == "libero_130":
        merged: dict[str, object] = {}
        for name in benchmark.libero_suites:
            for task_name, task in benchmark.task_maps[name].items():
                merged.setdefault(task_name, task)
        return list(merged.values())
    if task_suite not in suites:
        raise KeyError(f"unknown LIBERO suite {task_suite!r}; known: {sorted(suites) + ['libero_130']}")
    suite = suites[task_suite]()
    return [suite.get_task(i) for i in range(int(suite.n_tasks))]


def _agentview(obs: dict) -> np.ndarray:
    image = obs["agentview_image"]
    return np.ascontiguousarray(image[::-1, ::-1])


def _step_env(env, action: np.ndarray):
    out = env.step(np.asarray(action, dtype=np.float32))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
        done = bool(done)
    return obs, float(reward), done, info or {}


def _reset_env(env, init_state: np.ndarray, num_steps_wait: int) -> np.ndarray:
    env.reset()
    obs = env.set_init_state(init_state.copy())
    for _ in range(num_steps_wait):
        obs, _, _, _ = _step_env(env, WAIT_ACTION)
    return _agentview(obs)


def _run_chunk(env, actions: np.ndarray, budget: int) -> tuple[np.ndarray, bool, bool, float]:
    """Step ``actions`` until success, done or ``budget`` steps; returns (frames, success, done, sim seconds)."""
    frames = []
    success = done = False
    t0 = time.perf_counter()
    for action in actions[: max(budget, 0)]:
        obs, _, done, _ = _step_env(env, action)
        frames.append(_agentview(obs))
        if env.check_success():
            success = True
            break
        if done:
            break
    sim_s = time.perf_counter() - t0
    return np.stack(frames) if frames else np.zeros((0,), dtype=np.uint8), success, done, sim_s


class _Sim:
    """One LIBERO sim, rebuilt whenever the slot switches task."""

    def __init__(self, bddl: dict[int, str], camera_height: int, camera_width: int, horizon: int, seed: int) -> None:
        _, _, self._offscreen = _libero()
        self.bddl = bddl
        self.kwargs = dict(camera_heights=camera_height, camera_widths=camera_width, horizon=horizon)
        self.seed = seed
        self.task_id = -1
        self.env = None

    def ensure(self, task_id: int):
        if self.env is None or self.task_id != task_id:
            self.close()
            self.env = self._offscreen(bddl_file_name=self.bddl[task_id], **self.kwargs)
            self.env.seed(self.seed)
            self.task_id = task_id
        return self.env

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self.task_id = -1


def _worker(conn, bddl, camera_height, camera_width, horizon, num_steps_wait, seed) -> None:
    sim = _Sim(bddl, camera_height, camera_width, horizon, seed)
    while True:
        try:
            cmd, payload = conn.recv()
        except EOFError:
            sim.close()
            return
        if cmd == "reset":
            task_id, init_state = payload
            conn.send(_reset_env(sim.ensure(task_id), init_state, num_steps_wait))
        elif cmd == "chunk":
            actions, budget = payload
            conn.send(_run_chunk(sim.env, actions, budget))
        elif cmd == "close":
            sim.close()
            conn.send(None)
            return


@dataclass
class LiberoInit:
    task_id: int
    init_state: np.ndarray
    instruction: str
    source: str


class LiberoInitDataset:
    """One entry per published LIBERO initial state (task x trial)."""

    def __init__(self, task_suite: str) -> None:
        _, get_libero_path, _ = _libero()
        tasks = suite_tasks(task_suite)
        init_root = get_libero_path("init_states")
        self.states: list[LiberoInit] = []
        for task_id, task in enumerate(tasks):
            init_path = os.path.join(init_root, task.problem_folder, task.init_states_file)
            init_states = torch.load(init_path, weights_only=False)
            for trial, state in enumerate(init_states):
                self.states.append(LiberoInit(
                    task_id=task_id,
                    init_state=np.asarray(state, dtype=np.float64),
                    instruction=str(task.language),
                    source=f"{task_suite}/task{task_id}/trial{trial}",
                ))
        self.instructions = sorted({s.instruction for s in self.states})
        logger.info(
            "LIBERO %s: %d tasks, %d init states",
            task_suite, len(tasks), len(self.states),
        )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, i: int) -> LiberoInit:
        return self.states[i]


class LiberoEnv:
    """``num_envs`` LIBERO sims driven from this process. Rendering is OSMesa on CPU."""

    chunk = CHUNK

    def __init__(
        self,
        task_suite: str,
        max_episode_steps: int,
        camera_height: int = 128,
        camera_width: int = 128,
        num_steps_wait: int = 10,
        seed: int = 0,
        num_workers: int = 0,
        record_frames: bool = False,
    ) -> None:
        _, get_libero_path, _ = _libero()
        bddl_root = get_libero_path("bddl_files")
        self._bddl: dict[int, str] = {}
        for task_id, task in enumerate(suite_tasks(task_suite)):
            self._bddl[task_id] = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        self.max_episode_steps = int(max_episode_steps)
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.num_steps_wait = int(num_steps_wait)
        self.horizon = self.max_episode_steps + self.num_steps_wait + 10
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.device = torch.device("cpu")
        # Keeps every sim frame of every episode on the CPU (for videos).
        self.record_frames = record_frames
        self.history: list[list[torch.Tensor]] = []
        self.timing = {"reset_s": 0.0, "sim_s": 0.0, "sim_steps": 0}
        self._local: list[_Sim] = []
        self._procs: list = []
        self._conns: list = []
        self.num_envs = 0
        self.frames: torch.Tensor | None = None
        self.elapsed: torch.Tensor | None = None
        self.done: torch.Tensor | None = None
        self.success: torch.Tensor | None = None
        self.instructions: list[str] = []

    def _grow(self, n: int) -> None:
        if self.num_workers > 0:
            ctx = mp.get_context("spawn")
            while len(self._conns) < n:
                i = len(self._conns)
                parent, child = ctx.Pipe()
                proc = ctx.Process(
                    target=_worker,
                    args=(child, self._bddl, self.camera_height, self.camera_width,
                          self.horizon, self.num_steps_wait, self.seed + i),
                    daemon=True,
                )
                proc.start()
                child.close()
                self._procs.append(proc)
                self._conns.append(parent)
        else:
            while len(self._local) < n:
                self._local.append(_Sim(self._bddl, self.camera_height, self.camera_width,
                                        self.horizon, self.seed + len(self._local)))

    def close(self) -> None:
        for sim in self._local:
            sim.close()
        for conn in self._conns:
            conn.send(("close", None))
            conn.recv()
        for proc in self._procs:
            proc.join(timeout=10)
        self._local, self._procs, self._conns = [], [], []

    def reset(self, inits: list[LiberoInit]) -> dict:
        t0 = time.perf_counter()
        n = len(inits)
        self._grow(n)
        if self.num_workers > 0:
            for conn, init in zip(self._conns, inits):
                conn.send(("reset", (init.task_id, init.init_state)))
            images = [self._conns[i].recv() for i in range(n)]
        else:
            images = [
                _reset_env(self._local[i].ensure(init.task_id), init.init_state, self.num_steps_wait)
                for i, init in enumerate(inits)
            ]
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
        assert self.frames is not None and self.done is not None
        actions_np = actions.detach().to("cpu", torch.float32).numpy()[:, : self.chunk]
        reward = torch.zeros(self.num_envs, dtype=torch.float32)
        active = (~self.done).nonzero(as_tuple=False).flatten().tolist()
        budgets = {i: self.max_episode_steps - int(self.elapsed[i]) for i in active}
        if self.num_workers > 0:
            for i in active:
                self._conns[i].send(("chunk", (actions_np[i], budgets[i])))
            results = {i: self._conns[i].recv() for i in active}
        else:
            results = {i: _run_chunk(self._local[i].env, actions_np[i], budgets[i]) for i in active}
        for i in active:
            frames, success, done, sim_s = results[i]
            steps = int(len(frames))
            self.timing["sim_s"] += sim_s
            self.timing["sim_steps"] += steps
            if steps:
                self.frames[i] = torch.from_numpy(frames[-1])
                if self.record_frames:
                    self.history[i].append(torch.from_numpy(frames))
            self.elapsed[i] += steps
            if success:
                self.success[i] = True
                reward[i] = 1.0
            if success or done or steps == 0 or int(self.elapsed[i]) >= self.max_episode_steps:
                self.done[i] = True
        return self.observation(), reward, self.done.clone(), {}
