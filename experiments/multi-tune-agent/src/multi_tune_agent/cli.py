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
    canonical_gemm_template_for_contract,
    get_gemm_template,
    normalize_gemm_format,
    validate_template_target,
)
from geak_utils.local_templates import (
    VerifiedTemplateRecord,
    find_verified_template,
    register_verified_template,
)

from .config import MultiTuneConfig
from .flow import MultiTuneFlow
from .geak_tool import GEAKToolEnvironment
from .request_parser import recognize_kernel_request
from .resume import find_latest_checkpoint, mark_checkpoint_continued
from .runtime import OpenAIModelBackend
from .task_factory import (
    GeneratedKernelTask,
    generate_gemm_task,
    materialize_verified_template_task,
    register_generated_case,
)
from .template_bootstrap import (
    BootstrapError,
    KernelContract,
    TemplateBootstrapper,
    promote_validated_template,
    run_template_gpu_gate,
)


_CASE_TYPES = (
    "gemm",
    "fused_attention",
    "grouped_gemm",
    "scaled_quant_gemm",
    "quant_fp4_mxfp",
    "aiter_generated",
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


def _prompt_gemm_supplement(
    request: str,
    reason: object,
    *,
    input_fn=None,
    output_fn=print,
) -> str | None:
    """Collect missing GEMM details without dropping the original request."""

    input_fn = input if input_fn is None else input_fn
    output_fn("More information is needed: %s" % reason)
    while True:
        supplement = input_fn(
            "Add only the missing details shown above ([b] back): "
        ).strip()
        if supplement.lower() in {"b", "back", "q", "quit"}:
            return None
        if supplement:
            if request.strip():
                return "%s; user supplement: %s" % (request.strip(), supplement)
            return supplement
        output_fn("Please enter the missing details, or [b] to go back.")


def _missing_kernel_fields(recognized: Mapping[str, Any]) -> list[str]:
    missing = []
    if not recognized.get("operator"):
        missing.append("operator")
    if not recognized.get("target_gpu"):
        missing.append("GPU")
    if not recognized.get("dimensions") and not recognized.get("shapes"):
        missing.append("shape/dimensions")
    format_name = recognized.get("format") or recognized.get("input_dtype")
    if not format_name:
        missing.append("format/input dtype")
    quantized = str(format_name or "").lower() in {
        "fp8",
        "float8",
        "int8",
        "mxfp8",
        "mxfp4",
        "fp4",
    }
    if quantized and str(format_name).lower() not in {"mxfp8", "mxfp4", "fp4"}:
        if not recognized.get("input_scale_granularity"):
            missing.append("input scale granularity")
        if not recognized.get("weight_scale_granularity"):
            missing.append("weight scale granularity")
    return missing


def _kernel_contract(recognized: Mapping[str, Any], request: str) -> KernelContract:
    target = str(recognized["target_gpu"])
    try:
        architecture = architecture_for_target(target)
    except ValueError:
        architecture = re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_")
    shapes = recognized.get("shapes") or []
    if not shapes:
        dimensions = recognized.get("dimensions")
        if isinstance(dimensions, Mapping) and dimensions:
            shapes = [list(dimensions.values())]
    format_name = recognized.get("format")
    input_dtype = recognized.get("input_dtype") or format_name
    weight_dtype = recognized.get("weight_dtype")
    if not weight_dtype and str(recognized.get("operator")) == "gemm":
        weight_dtype = input_dtype
    return KernelContract(
        operator=str(recognized["operator"]),
        request=request,
        target_gpu=target,
        architecture=architecture,
        language=str(recognized.get("language") or "triton"),
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        output_dtype=recognized.get("output_dtype"),
        input_format=format_name,
        weight_format=format_name if weight_dtype else None,
        input_scale_granularity=recognized.get("input_scale_granularity"),
        weight_scale_granularity=recognized.get("weight_scale_granularity"),
        block_size=recognized.get("block_size"),
        shapes=shapes,
    )


def _validate_native_quant_target(contract: KernelContract) -> None:
    native_cdna4_formats = {"mxfp4", "mxfp6", "mxfp8", "mxint8"}
    formats = {contract.input_format, contract.weight_format}
    requested = sorted(
        value for value in formats if value in native_cdna4_formats
    )
    if requested and contract.architecture != "gfx950":
        raise ValueError(
            "%s requires native CDNA4 block-scaled MFMA and is unsupported on "
            "%s (%s). Use MI350/MI355, or use FP8 A8W8 E4M3FNUZ on MI308."
            % ("/".join(requested).upper(), contract.target_gpu, contract.architecture)
        )


def _generated_record(
    contract: KernelContract,
    template_path: Path,
    request: str,
) -> VerifiedTemplateRecord:
    metadata = json.loads(
        (template_path / "metadata.json").read_text(encoding="utf-8")
    )
    if not isinstance(metadata, Mapping):
        raise BootstrapError("promoted metadata.json is not an object")
    return VerifiedTemplateRecord(
        contract_hash=contract.contract_hash,
        operator=contract.operator,
        template_path=template_path,
        architecture=contract.architecture,
        language=contract.language,
        backend=contract.language,
        provenance=metadata.get("provenance") or {
            "generation_method": "verified_bootstrap"
        },
        direction=request,
    )


def _generate_kernel_task_interactive(
    config: MultiTuneConfig,
    backend: OpenAIModelBackend,
    *,
    input_fn=None,
    output_fn=print,
) -> GeneratedKernelTask | Any | None:
    """Recognize, generate or reuse, and return one runnable kernel task."""

    input_fn = input if input_fn is None else input_fn
    request = input_fn(
        "Describe the operator, GPU, shape/dimensions, and format/input dtype: "
    ).strip()
    recognized_state: dict[str, Any] = {}
    while True:
        try:
            recognized = recognize_kernel_request(request, backend)
        except ValueError as exc:
            supplemented = _prompt_gemm_supplement(
                request, exc, input_fn=input_fn, output_fn=output_fn
            )
            if supplemented is None:
                return None
            request = supplemented
            continue
        for field in (
            "operator",
            "target_gpu",
            "format",
            "input_dtype",
            "weight_dtype",
            "output_dtype",
            "input_scale_granularity",
            "weight_scale_granularity",
            "block_size",
            "dimensions",
            "shapes",
            "language",
        ):
            if not recognized.get(field) and recognized_state.get(field):
                recognized[field] = recognized_state[field]
        recognized_state = dict(recognized)
        missing = _missing_kernel_fields(recognized)
        if missing:
            output_fn(
                "Recognized so far: operator=%s, GPU=%s, format=%s, "
                "shape=%s, language=%s"
                % (
                    recognized.get("operator") or "?",
                    recognized.get("target_gpu") or "?",
                    recognized.get("format")
                    or recognized.get("input_dtype")
                    or "?",
                    recognized.get("shapes")
                    or recognized.get("dimensions")
                    or "?",
                    recognized.get("language") or "?",
                )
            )
            supplemented = _prompt_gemm_supplement(
                request,
                ", ".join(missing),
                input_fn=input_fn,
                output_fn=output_fn,
            )
            if supplemented is None:
                return None
            request = supplemented
            continue
        break

    output_fn(
        "Recognized by %s: operator=%s, GPU=%s, format=%s, shapes=%s"
        % (
            recognized.get("recognition", "unknown"),
            recognized["operator"],
            recognized["target_gpu"],
            recognized.get("format") or recognized.get("input_dtype"),
            recognized.get("shapes") or recognized.get("dimensions"),
        )
    )
    descriptor = canonical_gemm_template_for_contract(recognized)
    canonical_architecture = None
    if descriptor is not None:
        try:
            descriptor, canonical_architecture = validate_template_target(
                descriptor.format, recognized["target_gpu"]
            )
        except ValueError:
            descriptor = None
    if descriptor is not None and all(
        recognized.get(name) is not None for name in ("m", "n", "k")
    ):
        decision = input_fn(
            "Confirm canonical format=%s, backend=%s, architecture=%s? "
            "[Y/b, or type a correction]: "
            % (descriptor.format, descriptor.backend, canonical_architecture)
        ).strip()
        if decision.lower() in {"b", "back"}:
            return None
        if decision.lower() not in {"", "y", "yes"}:
            request = "%s; user correction: %s" % (request, decision)
            return _generate_kernel_task_from_request(
                config, backend, request, input_fn=input_fn, output_fn=output_fn
            )
        spec = dict(recognized)
        spec["dtype"] = descriptor.format
        return generate_gemm_task(config, request, parsed_spec=spec)

    return _bootstrap_kernel_task(
        config, backend, request, recognized, input_fn=input_fn, output_fn=output_fn
    )


def _generate_kernel_task_from_request(
    config: MultiTuneConfig,
    backend: OpenAIModelBackend,
    request: str,
    *,
    input_fn,
    output_fn,
) -> GeneratedKernelTask | Any | None:
    """Re-recognize an edited request without asking for a new initial description."""

    while True:
        recognized = recognize_kernel_request(request, backend)
        missing = _missing_kernel_fields(recognized)
        if not missing:
            break
        supplemented = _prompt_gemm_supplement(
            request, ", ".join(missing), input_fn=input_fn, output_fn=output_fn
        )
        if supplemented is None:
            return None
        request = supplemented
    descriptor = canonical_gemm_template_for_contract(recognized)
    if descriptor is not None and all(
        recognized.get(name) is not None for name in ("m", "n", "k")
    ):
        try:
            descriptor, _ = validate_template_target(
                descriptor.format, recognized["target_gpu"]
            )
        except ValueError:
            descriptor = None
    if descriptor is not None:
        spec = dict(recognized)
        spec["dtype"] = descriptor.format
        return generate_gemm_task(config, request, parsed_spec=spec)
    return _bootstrap_kernel_task(
        config, backend, request, recognized, input_fn=input_fn, output_fn=output_fn
    )


def _bootstrap_kernel_task(
    config: MultiTuneConfig,
    backend: OpenAIModelBackend,
    request: str,
    recognized: Mapping[str, Any],
    *,
    input_fn,
    output_fn,
) -> GeneratedKernelTask | None:
    contract = _kernel_contract(recognized, request)
    _validate_native_quant_target(contract)
    registry_path = config.generated_template_root / "templates.yaml"
    try:
        existing = find_verified_template(
            registry_path,
            contract.contract_hash,
            verified_root=(
                config.generated_template_root
                if config.generated_template_root.exists()
                else None
            ),
        )
        if existing is not None:
            output_fn("[lumen-code] Reusing locally verified template")
            return materialize_verified_template_task(
                config, existing, request=request
            )
        if not config.bootstrap_enabled:
            output_fn(
                "[lumen-code] No exact verified template; bootstrap is disabled."
            )
            return None
        output_fn("[lumen-code] Generating isolated template draft")

        def bootstrap_event(event: Mapping[str, Any]) -> None:
            phase = str(event.get("phase") or "event")
            details = ", ".join(
                "%s=%s" % (key, value)
                for key, value in event.items()
                if key != "phase" and value is not None
            )
            output_fn(
                "[lumen-code][bootstrap] %s%s"
                % (phase, ": " + details if details else "")
            )

        draft = TemplateBootstrapper(
            backend=backend,
            draft_root=config.generated_template_root.parent / ".generated",
            aiter_root=config.aiter_root,
            minimum_aiter_score=config.bootstrap_min_aiter_score,
            event_sink=bootstrap_event,
        ).generate(contract)
        output_fn("[lumen-code] Running compile, correctness, and performance gate")
        gate = run_template_gpu_gate(
            draft,
            geak_root=config.geak_root,
            run_root=config.trajectory_root / "template-validation",
            gpu_ids=config.gpu_ids,
            command_timeout=config.command_timeout,
        )
        if not gate.trusted:
            output_fn("[lumen-code] Template gate failed; nothing was cataloged.")
            for error in gate.errors:
                output_fn("  " + error)
            if gate.validation_workspace is not None:
                output_fn("  diagnostics: %s" % gate.validation_workspace)
            return None
        if config.bootstrap_auto_promote:
            output_fn(
                "[lumen-code] GPU gate passed; automatically promoting and "
                "registering the template"
            )
        else:
            decision = input_fn(
                "GPU gate passed. Promote and permanently register this template? [Y/n]: "
            ).strip().lower()
            if decision not in {"", "y", "yes"}:
                output_fn("[lumen-code] Promotion cancelled; nothing was cataloged.")
                return None
        output_fn("[lumen-code] Promoting trusted template")
        promoted = promote_validated_template(
            draft, gate, config.generated_template_root
        )
        record = _generated_record(contract, promoted, request)
        record = register_verified_template(
            registry_path,
            record,
            verified_root=config.generated_template_root,
        )
        return materialize_verified_template_task(config, record, request=request)
    except BootstrapError as exc:
        output_fn("[lumen-code] Bootstrap failed: %s" % exc)
        if exc.validation_report is not None:
            for issue in exc.validation_report.errors:
                output_fn("  " + str(issue))
        if exc.draft_path is not None:
            output_fn("  draft diagnostics: %s" % exc.draft_path)
        return None


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
        "Kernel tasks can be generated from an operator contract or an exact "
        "canonical GEMM template.\n"
    )
    while True:
        checkpoint = find_latest_checkpoint(base_config.trajectory_root)
        continue_choice = (
            ", [c] continue last failed run" if checkpoint is not None else ""
        )
        try:
            choice = input(
                "Choose [1] existing case, [2] generate kernel task, "
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
                task = _generate_kernel_task_interactive(base_config, backend)
                if task is None:
                    print("Cancelled.\n")
                    continue
                register_generated_case(base_config.cases_path, task)
                case_id = task.case_id
                request = task.request
                print("Generated task: %s" % task.task_dir)
                print("Registered case: %s" % base_config.cases_path)
                if isinstance(task, GeneratedKernelTask):
                    print(
                        "Task contract: operator=%s, hash=%s, backend=%s, architecture=%s"
                        % (
                            task.operator,
                            task.contract_hash,
                            task.backend,
                            task.architecture,
                        )
                    )
                else:
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

