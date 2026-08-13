#!/usr/bin/env python3
"""Preflight checks for the ATOM teacher path.

Loading K3 takes about 20 minutes, so a wiring mistake that only surfaces at
engine start costs a long round trip. Everything here runs in a couple of
seconds and needs no GPU: it verifies that the ATOM APIs this integration builds
on still look the way the worker expects, and that the worker script itself is
consistent.

    python3 examples/Kimi_K3_SDDD_MI350_ATOM/selfcheck/preflight.py

Exits non-zero on the first problem, with a description of what drifted.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import pathlib
import re
import sys

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  OK    {label}")
    else:
        print(f"  FAIL  {label}" + (f" -- {detail}" if detail else ""))
        FAILURES.append(label)


def main() -> int:
    print("ATOM teacher preflight")

    # ---- ATOM's engine-side API ---------------------------------------------
    from atom.rollout.async_engine import AsyncLLMEngine

    params = list(
        inspect.signature(AsyncLLMEngine.configure_hidden_states).parameters
    )
    check(
        "configure_hidden_states(self, aux_layer_ids, mooncake_config)",
        params == ["self", "aux_layer_ids", "mooncake_config"],
        f"got {params}; a capture_mode argument would need re-adding to the worker",
    )
    check(
        "AsyncLLMEngine.generate_hidden_states exists",
        hasattr(AsyncLLMEngine, "generate_hidden_states"),
    )
    check(
        "AsyncLLMEngine has no combined generate_with_hidden_states",
        not hasattr(AsyncLLMEngine, "generate_with_hidden_states"),
        "if ATOM regained it, the two-sweep split could be collapsed",
    )
    for name in ("add_request", "step", "is_finished"):
        check(f"AsyncLLMEngine.{name} exists (decode loop)", hasattr(AsyncLLMEngine, name))

    # ---- The runner methods our subclass overrides ---------------------------
    # Importing the module needs a GPU (aiter shells out to rocminfo), so inspect
    # the source instead.
    spec = importlib.util.find_spec("atom.rollout.model_runner_ext")
    tree = ast.parse(pathlib.Path(spec.origin).read_text())
    atom_methods = {
        n.name
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
    }
    for name in ("configure_hidden_states", "_store_hidden_states", "run_model"):
        check(
            f"RLHFModelRunner.{name} still defined",
            name in atom_methods,
            "our override would silently stop taking effect",
        )

    runner_src = pathlib.Path(spec.origin).read_text()
    # The generate sweep parks capture by withholding the external id, so that
    # has to remain what ATOM keys writes on.
    check(
        "ATOM still keys hidden-state writes on external_request_ids",
        "external_request_ids" in runner_src,
        "the generate sweep parks capture by withholding that id; if ATOM "
        "switched to another key, A1 would start writing its prompt prefill",
    )
    # Documents why enable_chunked_prefill must stay off: ATOM writes whatever
    # was scheduled this step, with no idea it is looking at a partial prefill.
    check(
        "ATOM's store still ignores is_final_chunk (chunked prefill unsafe)",
        "is_final_chunk" not in runner_src,
        "ATOM learned about partial prefills; enable_chunked_prefill could be "
        "revisited, and this check retired",
    )

    # ---- Engine settings are really Config fields ----------------------------
    # ATOM drops unknown kwargs without a word, so a typo in the yaml would be a
    # silent no-op. Cross-check the shipped configs against the dataclass.
    from dataclasses import fields as dc_fields

    from atom.config import Config as AtomConfig

    known = {f.name for f in dc_fields(AtomConfig)}
    cfg_dir = pathlib.Path(__file__).resolve().parent.parent / "configs"
    try:
        import yaml

        for cfg_path in sorted(cfg_dir.glob("*.yaml")):
            doc = yaml.safe_load(cfg_path.read_text())
            algorithm = doc.get("algorithm", {})
            atom_cfg = algorithm.get("teacher", {}).get("atom") or {}

            # separate_last_hidden describes vLLM's split storage layout: aux
            # layers minus one in hidden_states, the last one (pre-norm) in
            # last_hidden_states, rejoined by the trainer. ATOM stores all aux
            # layers together and its last_hidden_states is the model output
            # after the final norm, so rejoining would hand the draft's fc one
            # layer too many and re-apply the norm.
            spec = algorithm.get("spec_distill", {}) or {}
            aux_ids = spec.get("aux_hidden_state_layer_ids") or []
            check(
                f"{cfg_path.name}: separate_last_hidden is false (ATOM layout)",
                spec.get("separate_last_hidden") is False,
                f"true would make the trainer concat last_hidden_states onto "
                f"the {len(aux_ids)} aux layers, so the draft's fc would see "
                f"{(len(aux_ids) + 1)} x hidden_size instead of "
                f"{len(aux_ids)} x hidden_size",
            )
            unknown = sorted(set(atom_cfg) - known)
            check(
                f"{cfg_path.name}: all atom.* keys are Config fields",
                not unknown,
                f"ATOM would ignore {unknown}",
            )
            # A prefill split across scheduler steps overwrites its own Mooncake
            # entry, leaving only the last chunk.
            budget = atom_cfg.get("max_num_batched_tokens")
            model_len = atom_cfg.get("max_model_len")
            if budget and model_len:
                check(
                    f"{cfg_path.name}: max_num_batched_tokens >= max_model_len",
                    budget >= model_len,
                    f"{budget} < {model_len} would chunk prefill and truncate "
                    f"hidden states",
                )
            # Both default to True in ATOM, and both silently truncate the
            # stored features. Absent is as bad as true, hence the explicit
            # `is False` rather than a falsy test.
            check(
                f"{cfg_path.name}: enable_chunked_prefill is false",
                atom_cfg.get("enable_chunked_prefill") is False,
                "ATOM defaults it on; a chunked prefill writes every chunk "
                "under the same key, so only the last one survives",
            )
            check(
                f"{cfg_path.name}: enable_prefix_caching is false",
                atom_cfg.get("enable_prefix_caching") is False,
                "ATOM defaults it on; a cached prefix is excluded from the "
                "scheduled tokens, so A2 would store features with the prompt "
                "section missing",
            )
    except ImportError:
        print("  SKIP  config scan (pyyaml missing)")

    # ---- The worker script --------------------------------------------------
    from lumenrl.engine.inference.atom_teacher_engine import (
        _TEACHER_WORKER_SCRIPT as worker,
    )
    from lumenrl.engine.inference.atom_teacher_engine import AtomTeacherEngine

    try:
        compile(worker, "<atom worker>", "exec")
        check("worker script compiles", True)
    except SyntaxError as exc:
        check("worker script compiles", False, str(exc))

    calls = re.findall(r"configure_hidden_states\(.*", worker)
    check(
        "worker calls configure_hidden_states without capture_mode",
        calls == ["configure_hidden_states(aux_layer_ids, mooncake_config)"],
        f"found {calls}",
    )
    check(
        "worker does not call the removed generate_with_hidden_states",
        "generate_with_hidden_states" not in worker,
    )
    # Match the actual path list, not any mention of it: the comment above that
    # line names third_party/ATOM precisely to explain why it is excluded.
    sys_path_entries = re.search(r"for _p in \[(.*?)\]", worker, re.S)
    check(
        "worker keeps the stale third_party/ATOM off sys.path",
        sys_path_entries is not None
        and "ATOM" not in sys_path_entries.group(1),
        f"sys.path seed is {sys_path_entries.group(1) if sys_path_entries else 'missing'}; "
        f"a checked-in ATOM would shadow the newer /app/ATOM in the image",
    )
    check(
        "worker selects LumenRL's runner",
        "runner_qualname" in worker and "LumenRLModelRunner" in worker,
    )
    check(
        "worker sizes the host buffer for the real aux-layer count",
        "num_aux_layers=len(aux_layer_ids)" in worker,
        "the helper defaults to 3, which is too small for a 5-layer contract",
    )
    for cmd in ("generate_tokens", "extract_hidden", "shutdown"):
        check(f'worker handles cmd "{cmd}"', f'cmd == "{cmd}"' in worker)

    # An unrecognised utility command is the worst failure shape ATOM has: the
    # engine core logs "Unknown utility command", never pushes the
    # UTILITY_RESPONSE that broadcast_utility_command_sync is blocked on, and the
    # worker hangs until the start timeout fires -- 20 minutes after the model
    # finished loading, with the cause buried in the worker log.
    from atom.model_engine.engine_utility import EngineUtilityHandler

    whitelist = set(EngineUtilityHandler._UTILITY_HANDLERS)
    broadcast = set(re.findall(r'broadcast_utility_command\w*\(\s*"([^"]+)"', worker))
    check(
        "every utility command the worker sends has an ATOM handler",
        broadcast <= whitelist,
        f"{sorted(broadcast - whitelist)} would hang forever; ATOM accepts "
        f"{sorted(whitelist)}",
    )
    check(
        "worker no longer sends the unhandled set_extract_mode",
        "set_extract_mode" not in worker,
        "capture is selected per request now, via the presence of a data id",
    )
    # The generate sweep must not hand ATOM an external id, or it would write its
    # prompt prefill to Mooncake.
    check(
        "generate sweep submits without request ids",
        "io_proc.preprocess(p, sp)" in worker
        and "add_request(prompts, sp, request_ids" not in worker,
        "passing request_ids in the generate sweep would arm capture and push "
        "gigabytes per batch that nobody reads",
    )

    for name in ("generate_tokens", "extract_hidden_states", "switch_mode", "mode"):
        check(f"AtomTeacherEngine.{name} exists", hasattr(AtomTeacherEngine, name))

    # ---- The torchspec shim ATOM imports ------------------------------------
    from torchspec.config.mooncake_config import MooncakeConfig
    from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

    check(
        "torchspec shim maps onto lumenrl.transfer",
        MooncakeConfig.__module__.startswith("lumenrl.")
        and any(
            base.__module__.startswith("lumenrl.")
            for base in EagleMooncakeStore.__mro__[1:2]
        ),
        f"{MooncakeConfig.__module__}, {EagleMooncakeStore.__mro__[1].__module__}",
    )
    # ATOM constructs the store and puts straight away, never calling setup().
    check(
        "shim store connects without an explicit setup() call",
        hasattr(EagleMooncakeStore, "_ensure_setup"),
    )

    print()
    if FAILURES:
        print(f"PREFLIGHT FAILED: {len(FAILURES)} problem(s): {FAILURES}")
        return 1
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
