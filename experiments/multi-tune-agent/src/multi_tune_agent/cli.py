"""Command-line interface for MultiTune."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import requests
import yaml
from geak_utils.templates import (
    architecture_for_target,
    get_gemm_template,
    normalize_gemm_format,
    validate_template_target,
)

from .config import MultiTuneConfig
from .flow import MultiTuneFlow
from .geak_tool import GEAKToolEnvironment
from .request_parser import recognize_gemm_request
from .resume import find_latest_checkpoint, mark_checkpoint_continued
from .runtime import OpenAIModelBackend
from .task_factory import (
    generate_gemm_task,
    parse_gemm_request,
    register_generated_case,
)


_CASE_TYPES = (
    "gemm",
    "fused_attention",
    "grouped_gemm",
    "scaled_quant_gemm",
    "quant_fp4_mxfp",
)
_PRINT_LOCK = threading.Lock()

_GEMM_FORMAT_CHOICES = (
    ("fp16", "FP16"),
    ("fp8", "FP8 A8W8"),
    ("mxfp4", "MXFP4 (native gfx950 only)"),
)


def _prompt_gemm_format(
    recognized_format: object,
    target_gpu: object,
    *,
    input_fn=None,
    output_fn=print,
) -> str:
    """Prompt until exactly one target-compatible GEMM format is selected."""

    input_fn = input if input_fn is None else input_fn
    default = normalize_gemm_format(recognized_format)
    architecture = architecture_for_target(target_gpu)
    while True:
        output_fn("Choose GEMM format:")
        for index, (value, label) in enumerate(_GEMM_FORMAT_CHOICES, 1):
            suffix = " [default]" if value == default else ""
            output_fn("  %d. %s%s" % (index, label, suffix))
        raw = input_fn("Format choice (Enter uses default): ").strip().lower()
        aliases = {
            "1": "fp16",
            "2": "fp8",
            "3": "mxfp4",
            "fp16": "fp16",
            "fp8": "fp8",
            "a8w8": "fp8",
            "mxfp4": "mxfp4",
            "mx-fp4": "mxfp4",
        }
        selected = default if not raw else aliases.get(raw)
        if selected is None:
            output_fn("Choose exactly one of 1, 2, or 3.")
            continue
        try:
            validate_template_target(selected, target_gpu)
        except ValueError as exc:
            output_fn("Unsupported selection: %s" % exc)
            continue
        # Architecture resolution above makes the selected combination explicit
        # and keeps unsupported choices in this menu.
        assert architecture in get_gemm_template(selected).supported_architectures
        return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multi-tune")
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe")
    run = subparsers.add_parser("run")
    run.add_argument("--case", action="append", dest="cases")
    run.add_argument("--request", help="natural-language objective for one case")
    run.add_argument(
        "--stream", action="store_true", help="print role/tool progress while running"
    )
    subparsers.add_parser(
        "interactive",
        help="open a prompt-driven shell for existing or custom GEAK tasks",
    )
    return parser


def _model_backend(config: MultiTuneConfig) -> OpenAIModelBackend:
    return OpenAIModelBackend(
        config.base_url,
        config.model,
        timeout=config.request_timeout,
    )


def _console_event(record: Mapping[str, Any]) -> None:
    event = str(record.get("event") or "")
    role = str(record.get("role") or "orchestrator")
    phase = str(record.get("phase") or "")
    payload = record.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    message = ""
    if event == "environment_create":
        message = "%s workspace ready" % role
    elif event == "role_response":
        message = "%s/%s completed (%.2fs)" % (
            role,
            phase,
            float(payload.get("elapsed_seconds") or 0.0),
        )
    elif event == "tool_result":
        result = payload.get("result")
        result = result if isinstance(result, Mapping) else {}
        message = "%s tool=%s ok=%s (%.2fs)" % (
            role,
            payload.get("action") or "unknown",
            bool(result.get("ok")),
            float(payload.get("elapsed_seconds") or 0.0),
        )
    elif event == "round_end":
        message = "round %s committed=%s current_speedup=%.4f" % (
            payload.get("round"),
            bool(payload.get("committed")),
            float(payload.get("current_speedup") or 0.0),
        )
    elif event == "run_end":
        message = "run finished status=%s total=%.2fs" % (
            payload.get("status"),
            float((payload.get("timing_seconds") or {}).get("total") or 0.0),
        )
    if message:
        with _PRINT_LOCK:
            print("[lumen-code] " + message, flush=True)


def _write_custom_catalog(
    config: MultiTuneConfig,
    *,
    case_id: str,
    case_type: str,
    kernel_path: Path,
    request: str,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id):
        raise ValueError(
            "case ID must start with a letter or digit and contain only "
            "letters, digits, '.', '_', and '-'"
        )
    catalog_dir = config.trajectory_root / "requests"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / ("%s-%s.yaml" % (case_id, uuid.uuid4().hex[:8]))
    payload = {
        "tasks": [
            {
                "id": case_id,
                "type": case_type,
                "kernel_path": str(kernel_path.expanduser().resolve()),
                "direction": request,
            }
        ]
    }
    catalog_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return catalog_path


def _default_case_id(kernel_path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", kernel_path.name).strip("-._")
    if not value:
        return "custom-kernel"
    if not value[0].isalnum():
        return "custom-" + value
    return value


def _has_geak_contract(kernel_path: Path) -> bool:
    return (
        (kernel_path / "config.yaml").is_file()
        or (kernel_path / "config.yml").is_file()
        or (kernel_path / "config.json").is_file()
        or (kernel_path / "COMMANDMENT.md").is_file()
        or (
            (kernel_path / "unittest.py").is_file()
            and (kernel_path / "meta.json").is_file()
        )
    )


def _print_summary(summary: Mapping[str, Any]) -> None:
    evaluation = summary.get("final_evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    performance = summary.get("final_kernel_performance")
    performance = performance if isinstance(performance, Mapping) else {}
    timing = summary.get("timing_seconds")
    timing = timing if isinstance(timing, Mapping) else {}
    print(
        json.dumps(
            {
                "case_id": summary.get("case_id"),
                "status": summary.get("status"),
                "baseline_geomean_ms": performance.get("baseline_geomean_ms"),
                "final_kernel_geomean_ms": performance.get(
                    "final_kernel_geomean_ms"
                ),
                "speedup": performance.get(
                    "speedup_geomean", evaluation.get("speedup_geomean")
                ),
                "final_kernel_per_case_ms": performance.get(
                    "final_kernel_per_case_ms"
                ),
                "measurement_valid": performance.get("measurement_valid"),
                "total_seconds": timing.get("total"),
                "run_dir": summary.get("run_dir"),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def _run_interactive(base_config: MultiTuneConfig) -> int:
    backend = _model_backend(base_config)
    print(
        "\nMulti-Tune interactive CLI\n"
        "GEMM tasks can be generated from a request containing M, N, and K. "
        "Other custom operators need an existing trustworthy GEAK harness.\n"
    )
    while True:
        checkpoint = find_latest_checkpoint(base_config.trajectory_root)
        continue_choice = (
            ", [c] continue last failed run" if checkpoint is not None else ""
        )
        try:
            choice = input(
                "Choose [1] existing case, [2] generate GEMM task, "
                "[3] existing GEAK task%s, [q] quit: " % continue_choice
            ).strip().lower()
        except EOFError:
            print()
            return 0
        if choice in {"q", "quit", "exit"}:
            return 0
        try:
            run_config = base_config
            resume_options: dict[str, Any] = {}
            if choice in {"c", "continue"}:
                if checkpoint is None:
                    print("No recoverable failed run was found.\n")
                    continue
                case_id = checkpoint.case_id
                request = checkpoint.user_request
                environment = GEAKToolEnvironment(base_config)
                if case_id not in environment.cases:
                    catalog_path = _write_custom_catalog(
                        base_config,
                        case_id=case_id,
                        case_type=checkpoint.case_type,
                        kernel_path=checkpoint.workspace,
                        request=request,
                    )
                    run_config = replace(base_config, cases_path=catalog_path)
                resume_options = {
                    "resume_workspace": checkpoint.workspace,
                    "baseline_override": checkpoint.baseline,
                    "resume_context": checkpoint.resume_context,
                }
                print(
                    "Recovering %s from session %s\n"
                    "  workspace: %s\n"
                    "  baseline: %.6f ms"
                    % (
                        case_id,
                        checkpoint.session_id,
                        checkpoint.workspace,
                        float(checkpoint.baseline.get("geomean_ms") or 0.0),
                    )
                )
            elif choice == "1":
                environment = GEAKToolEnvironment(base_config)
                case_ids = list(environment.cases)
                for index, case_id in enumerate(case_ids, 1):
                    case = environment.cases[case_id]
                    print("  %d. %s (%s)" % (index, case_id, case.case_type))
                selected = input("Case number or ID: ").strip()
                if selected.isdigit() and 1 <= int(selected) <= len(case_ids):
                    case_id = case_ids[int(selected) - 1]
                else:
                    case_id = selected
                if case_id not in environment.cases:
                    raise ValueError("unknown case: %s" % case_id)
                request = input(
                    "Optimization request (blank uses catalog objective): "
                ).strip()
            elif choice == "2":
                request = input(
                    "Describe target GPU and GEMM, including M, N, K: "
                ).strip()
                original_request = request
                while True:
                    try:
                        recognized = recognize_gemm_request(request, backend)
                    except ValueError as exc:
                        print("Could not recognize the task: %s" % exc)
                        correction = input(
                            "Enter a corrected request or M/N/K only (1/128/128), "
                            "or [b] back: "
                        ).strip()
                        if correction.lower() in {"b", "back", "q", "quit"}:
                            task = None
                            break
                        try:
                            corrected = parse_gemm_request(correction)
                        except ValueError:
                            request = correction
                        else:
                            request = "%s; M=%d, N=%d, K=%d" % (
                                original_request,
                                corrected["m"],
                                corrected["n"],
                                corrected["k"],
                            )
                        continue
                    print(
                        "Recognized by %s: GPU=%s, language=%s, format=%s, "
                        "M=%d, N=%d, K=%d"
                        % (
                            recognized.get("recognition", "unknown"),
                            recognized["target_gpu"],
                            recognized.get("language", "triton"),
                            recognized["dtype"],
                            recognized["m"],
                            recognized["n"],
                            recognized["k"],
                        )
                    )
                    selected_format = _prompt_gemm_format(
                        recognized["dtype"], recognized["target_gpu"]
                    )
                    recognized = dict(recognized)
                    recognized["dtype"] = selected_format
                    descriptor, architecture = validate_template_target(
                        selected_format, recognized["target_gpu"]
                    )
                    print(
                        "Selected: format=%s, backend=%s, architecture=%s"
                        % (selected_format, descriptor.backend, architecture)
                    )
                    raw_decision = input(
                        "Confirm format/backend/architecture and values? "
                        "[Y/e edit/b back, or type a correction]: "
                    ).strip()
                    decision = raw_decision.lower()
                    if decision in {"b", "back"}:
                        task = None
                        break
                    if decision in {"e", "edit", "n", "no"}:
                        request = input("Corrected request: ").strip()
                        continue
                    if decision not in {"", "y", "yes"}:
                        request = "%s; user correction: %s" % (
                            request,
                            raw_decision,
                        )
                        continue
                    language = str(recognized.get("language") or "triton")
                    if language != "triton":
                        raise ValueError(
                            "%s was recognized, but automatic %s GEMM task "
                            "generation is not implemented yet"
                            % (language, language)
                        )
                    task = generate_gemm_task(
                        base_config, request, parsed_spec=recognized
                    )
                    break
                if task is None:
                    print("Cancelled.\n")
                    continue
                register_generated_case(base_config.cases_path, task)
                case_id = task.case_id
                print("Generated task: %s" % task.task_dir)
                print("Registered case: %s" % base_config.cases_path)
                print(
                    "Task contract: format=%s, backend=%s, architecture=%s"
                    % (task.format, task.backend, task.architecture)
                )
            elif choice == "3":
                raw_path = input("GEAK task directory: ").strip()
                kernel_path = Path(raw_path).expanduser().resolve()
                if not kernel_path.is_dir():
                    raise ValueError("task directory does not exist: %s" % kernel_path)
                if not _has_geak_contract(kernel_path):
                    raise ValueError(
                        "task has no GEAK contract; add config.yaml, COMMANDMENT.md, "
                        "or unittest.py + meta.json"
                    )
                case_type = (
                    input(
                        "Case type [%s] (gemm): " % "/".join(_CASE_TYPES)
                    ).strip()
                    or "gemm"
                )
                if case_type not in _CASE_TYPES:
                    raise ValueError("unsupported case type: %s" % case_type)
                default_id = _default_case_id(kernel_path)
                case_id = input("Case ID (%s): " % default_id).strip() or default_id
                request = input("Optimization request: ").strip()
                if not request:
                    raise ValueError("a custom task requires an optimization request")
                catalog_path = _write_custom_catalog(
                    base_config,
                    case_id=case_id,
                    case_type=case_type,
                    kernel_path=kernel_path,
                    request=request,
                )
                run_config = replace(base_config, cases_path=catalog_path)
            else:
                print("Please enter 1, 2, 3, c, or q.\n")
                continue

            confirm = input(
                "Start %s with request %r? [Y/n]: "
                % (case_id, request or "(catalog objective)")
            ).strip().lower()
            if confirm not in {"", "y", "yes"}:
                print("Cancelled.\n")
                continue
            if checkpoint is not None and resume_options:
                mark_checkpoint_continued(checkpoint)
            summary = asyncio.run(
                MultiTuneFlow(
                    run_config, backend, event_sink=_console_event
                ).run_case(
                    case_id,
                    user_request=request or None,
                    **resume_options,
                )
            )
            _print_summary(summary)
            print()
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        except Exception as exc:
            print("Error: %s\n" % exc, file=sys.stderr)
            recovered = find_latest_checkpoint(base_config.trajectory_root)
            if recovered is not None:
                print(
                    "Checkpoint available for %s. Enter [c] to continue from "
                    "workspace %s.\n"
                    % (recovered.case_id, recovered.workspace)
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = MultiTuneConfig.from_yaml(args.config)
    if args.command == "probe":
        response = requests.get(config.base_url.rstrip("/") + "/models", timeout=30)
        response.raise_for_status()
        models = [item["id"] for item in response.json().get("data") or []]
        result = {
            "base_url": config.base_url,
            "configured_model_present": config.model in models,
            "models": models,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["configured_model_present"] else 1

    if args.command == "interactive":
        return _run_interactive(config)

    if args.request and (not args.cases or len(args.cases) != 1):
        parser.error("--request requires exactly one --case")
    backend = _model_backend(config)
    summaries = asyncio.run(
        MultiTuneFlow(
            config, backend, event_sink=_console_event if args.stream else None
        ).run_all(args.cases, user_request=args.request)
    )
    print(json.dumps(summaries, indent=2, sort_keys=True, default=str))
    return 0 if all(item["status"] == "success" for item in summaries) else 2


if __name__ == "__main__":
    sys.exit(main())

