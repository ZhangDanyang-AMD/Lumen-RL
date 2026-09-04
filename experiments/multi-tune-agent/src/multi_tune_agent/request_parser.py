"""Model-assisted natural-language task recognition with strict validation."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Mapping

from geak_utils.templates import FORMAT_ALIASES, normalize_gemm_format

from .agents import extract_json
from .runtime import ModelBackend
from .task_factory import parse_gemm_request


def recognize_gemm_request(
    request: str, backend: ModelBackend
) -> dict[str, Any]:
    """Recognize a GEMM request with the model, then fall back deterministically."""

    try:
        turn = backend.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You extract GPU kernel task fields from noisy Chinese or "
                        "English text, including typos and pinyin. Return one JSON "
                        "object only. Never invent dimensions. Use null when absent. "
                        "If the user appends a correction, the later correction "
                        "overrides conflicting information earlier in the request. "
                        "Schema: {\"operator\":\"gemm|null\","
                        "\"target_gpu\":\"string|null\","
                        "\"format\":\"fp16|fp8|mxfp4|null\","
                        "\"dtype\":\"fp16|fp8|mxfp4|null\","
                        "\"m\":\"integer|null\",\"n\":\"integer|null\","
                        "\"k\":\"integer|null\","
                        "\"language\":\"triton|hip|flydsl|null\","
                        "\"confidence\":\"number from 0 to 1\"}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"request": request}, ensure_ascii=False, sort_keys=True
                    ),
                },
            ]
        )
        return _validate_model_result(request, extract_json(turn.text))
    except Exception as model_error:
        try:
            result = parse_gemm_request(request)
            result["language"] = _language_from_text(request) or "triton"
            result["recognition"] = "deterministic_fallback"
            result["model_error"] = str(model_error)
            return result
        except ValueError:
            raise ValueError(
                "model and deterministic parsers could not recognize the GEMM "
                "request: %s" % model_error
            ) from model_error


def recognize_kernel_request(
    request: str, backend: ModelBackend
) -> dict[str, Any]:
    """Recognize an arbitrary GPU operator without trusting unsupported fields."""
    if not request or not request.strip():
        raise ValueError("kernel request must not be empty")

    model_error: Exception | None = None
    try:
        turn = backend.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract a GPU operator request from noisy English or Chinese. "
                        "Return one JSON object only and use null/empty values when a "
                        "field is absent; never invent values. For every non-null "
                        "field, include the exact supporting substring from the user "
                        "request in evidence. Values may be normalized (for example "
                        "MI308 to gfx942), but evidence must preserve the raw text. Schema: "
                        '{"operator":"string|null","target_gpu":"string|null",'
                        '"format":"string|null","input_dtype":"string|null",'
                        '"weight_dtype":"string|null","output_dtype":"string|null",'
                        '"input_scale_granularity":"string|null",'
                        '"weight_scale_granularity":"string|null",'
                        '"block_size":"integer|null","dimensions":{"name":"integer"},'
                        '"shapes":[["integer"]],"language":"string|null",'
                        '"evidence":{"field":"exact substring from request"},'
                        '"confidence":"number from 0 to 1"}.'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"request": request}, ensure_ascii=False, sort_keys=True
                    ),
                },
            ]
        )
        return _validate_kernel_model_result(request, extract_json(turn.text))
    except Exception as exc:
        model_error = exc

    result = _deterministic_kernel_result(request)
    result["recognition"] = "deterministic_fallback"
    result["model_error"] = str(model_error)
    return result


def _validate_kernel_model_result(
    request: str, payload: dict[str, Any]
) -> dict[str, Any]:
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("recognizer returned no confidence") from exc
    if confidence < 0.5:
        raise ValueError("recognizer confidence is too low: %.3f" % confidence)

    evidence = _kernel_text_evidence(request)
    model_evidence = _validated_model_evidence(request, payload.get("evidence"))
    operator = _normalize_operator(payload.get("operator")) or evidence["operator"]
    dimensions = _validated_dimensions(request, payload.get("dimensions"))
    shapes = _validated_shapes(request, payload.get("shapes"))

    # GEMM's conventional scalar fields are accepted as a compatibility aid.
    if operator == "gemm":
        for name in ("m", "n", "k"):
            if name not in dimensions and payload.get(name) is not None:
                dimensions[name] = _validated_number(request, payload[name], name)
        dimensions.update(_gemm_dimensions_from_text(request))
        if all(name in dimensions for name in ("m", "n", "k")):
            shapes = [[dimensions["m"], dimensions["n"], dimensions["k"]]]

    block_size = None
    if payload.get("block_size") is not None:
        block_size = _validated_number(request, payload["block_size"], "block_size")

    # Model-normalized fields are accepted only when their exact evidence span
    # survives validation; deterministic extraction remains the fallback.
    target_gpu = _target_gpu_with_model_evidence(
        request,
        payload.get("target_gpu"),
        model_evidence.get("target_gpu") or model_evidence.get("gpu"),
    ) or evidence["target_gpu"]
    format_name = _model_dtype_with_evidence(
        payload.get("format"), model_evidence.get("format")
    ) or evidence["format"]
    input_dtype = _model_dtype_with_evidence(
        payload.get("input_dtype"), model_evidence.get("input_dtype")
    ) or evidence["input_dtype"]
    weight_dtype = _model_dtype_with_evidence(
        payload.get("weight_dtype"), model_evidence.get("weight_dtype")
    ) or evidence["weight_dtype"]
    output_dtype = _model_dtype_with_evidence(
        payload.get("output_dtype"), model_evidence.get("output_dtype")
    ) or evidence["output_dtype"]
    input_scale = _model_scale_with_evidence(
        payload.get("input_scale_granularity"),
        model_evidence.get("input_scale_granularity"),
    ) or evidence["input_scale_granularity"]
    weight_scale = _model_scale_with_evidence(
        payload.get("weight_scale_granularity"),
        model_evidence.get("weight_scale_granularity"),
    ) or evidence["weight_scale_granularity"]
    language = _model_language_with_evidence(
        payload.get("language"), model_evidence.get("language")
    ) or evidence["language"]
    explicit_fields = set(evidence["explicit_fields"])
    if operator:
        explicit_fields.add("operator")
    if dimensions:
        explicit_fields.add("dimensions")
    if shapes:
        explicit_fields.add("shapes")
    if block_size is not None:
        explicit_fields.add("block_size")
    if target_gpu is not None:
        explicit_fields.add("target_gpu")
    for field, value in (
        ("format", format_name),
        ("input_dtype", input_dtype),
        ("weight_dtype", weight_dtype),
        ("output_dtype", output_dtype),
        ("input_scale_granularity", input_scale),
        ("weight_scale_granularity", weight_scale),
        ("language", language),
    ):
        if value is not None and field in model_evidence:
            explicit_fields.add(field)

    result: dict[str, Any] = {
        "operator": operator,
        "target_gpu": target_gpu,
        "format": format_name,
        "input_dtype": input_dtype,
        "weight_dtype": weight_dtype,
        "output_dtype": output_dtype,
        "input_scale_granularity": input_scale,
        "weight_scale_granularity": weight_scale,
        "block_size": block_size,
        "dimensions": dimensions,
        "shapes": shapes,
        "language": language or "triton",
        "confidence": confidence,
        "request": request.strip(),
        "recognition": "model",
        "model_evidence": model_evidence,
    }
    return _finish_kernel_result(result, explicit_fields)


def _validated_model_evidence(request: str, value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    normalized_request = unicodedata.normalize("NFKC", request).casefold()
    validated: dict[str, str] = {}
    for field, raw_snippet in value.items():
        if not isinstance(field, str) or not isinstance(raw_snippet, str):
            continue
        snippet = unicodedata.normalize("NFKC", raw_snippet).strip()
        if snippet and snippet.casefold() in normalized_request:
            validated[field] = snippet
    return validated


def _normalized_freeform(value: Any) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[\s_-]+", "", str(value).strip().lower())
    return normalized or None


def _model_dtype_with_evidence(value: Any, snippet: str | None) -> str | None:
    if value is None or not snippet:
        return None
    proposed = _dtype_alias_from_text(str(value)) or _normalized_freeform(value)
    observed = _dtype_alias_from_text(snippet) or _normalized_freeform(snippet)
    return str(proposed) if proposed and proposed == observed else None


def _model_scale_with_evidence(value: Any, snippet: str | None) -> str | None:
    if value is None or not snippet:
        return None
    proposed = _normalize_scale(value)
    observed_values = _scale_mentions(snippet)
    observed = observed_values[-1] if observed_values else _normalize_scale(snippet)
    return proposed if proposed == observed else None


def _normalize_scale(value: Any) -> str | None:
    normalized = _normalized_freeform(value)
    if normalized is None:
        return None
    aliases = {
        "pertoken": "per_token",
        "perchannel": "per_channel",
        "perblock": "per_block",
        "pertensor": "per_tensor",
        "pergroup": "per_group",
        "token": "per_token",
        "channel": "per_channel",
        "block": "per_block",
        "tensor": "per_tensor",
        "group": "per_group",
    }
    return aliases.get(normalized)


def _model_language_with_evidence(value: Any, snippet: str | None) -> str | None:
    if value is None or not snippet:
        return None
    try:
        proposed = _normalize_language(value)
        observed = _language_from_text(snippet) or _normalize_language(snippet)
    except ValueError:
        return None
    return proposed if proposed == observed else None


def _deterministic_kernel_result(request: str) -> dict[str, Any]:
    evidence = _kernel_text_evidence(request)
    dimensions = _dimensions_from_text(request)
    operator = evidence["operator"]
    shapes: list[list[int]] = []
    if operator == "gemm":
        gemm_dimensions = _gemm_dimensions_from_text(request)
        dimensions.update(gemm_dimensions)
        if all(name in dimensions for name in ("m", "n", "k")):
            shapes = [[dimensions["m"], dimensions["n"], dimensions["k"]]]
    if not shapes:
        shapes = _shapes_from_text(request)
    explicit_fields = set(evidence["explicit_fields"])
    if operator:
        explicit_fields.add("operator")
    if dimensions:
        explicit_fields.add("dimensions")
    if shapes:
        explicit_fields.add("shapes")
    block_size = _block_size_from_text(request)
    if block_size is not None:
        explicit_fields.add("block_size")

    result: dict[str, Any] = {
        "operator": operator,
        "target_gpu": evidence["target_gpu"],
        "format": evidence["format"],
        "input_dtype": evidence["input_dtype"],
        "weight_dtype": evidence["weight_dtype"],
        "output_dtype": evidence["output_dtype"],
        "input_scale_granularity": evidence["input_scale_granularity"],
        "weight_scale_granularity": evidence["weight_scale_granularity"],
        "block_size": block_size,
        "dimensions": dimensions,
        "shapes": shapes,
        "language": evidence["language"] or "triton",
        "confidence": 0.0,
        "request": request.strip(),
    }
    return _finish_kernel_result(result, explicit_fields)


def _finish_kernel_result(
    result: dict[str, Any], explicit_fields: set[str]
) -> dict[str, Any]:
    implied_fields: list[str] = []
    if result["format"] in {"mxfp4", "mxfp8"}:
        for field in ("input_scale_granularity", "weight_scale_granularity"):
            if result[field] is None:
                result[field] = "per_block"
                implied_fields.append(field)
        if result["block_size"] is None:
            result["block_size"] = 32
            implied_fields.append("block_size")

    if result["operator"] == "gemm" and all(
        name in result["dimensions"] for name in ("m", "n", "k")
    ):
        result.update(
            {name: result["dimensions"][name] for name in ("m", "n", "k")}
        )
        result["shapes"] = [[result["m"], result["n"], result["k"]]]

    missing: list[str] = []
    for field in ("operator", "target_gpu"):
        if not result[field]:
            missing.append(field)
    if not result["dimensions"] and not result["shapes"]:
        missing.append("shape_or_dimensions")
    if _is_quantized(result):
        for field in ("input_scale_granularity", "weight_scale_granularity"):
            if not result[field]:
                missing.append(field)

    result["explicit_fields"] = sorted(explicit_fields)
    result["format_implied_fields"] = implied_fields
    result["implied_fields"] = list(implied_fields)
    result["missing_fields"] = missing
    return result


def _is_quantized(result: dict[str, Any]) -> bool:
    values = {
        result.get("format"),
        result.get("input_dtype"),
        result.get("weight_dtype"),
        result.get("output_dtype"),
    }
    return bool(values & {"fp8", "int8", "mxfp8", "mxfp4", "fp4"})


def _kernel_text_evidence(text: str) -> dict[str, Any]:
    normalized = unicodedata.normalize("NFKC", text)
    result: dict[str, Any] = {
        "operator": _operator_from_text(normalized),
        "target_gpu": _target_gpu_from_text(normalized),
        "format": _named_alias_from_text(normalized, "format"),
        "input_dtype": _named_alias_from_text(normalized, "input_dtype"),
        "weight_dtype": _named_alias_from_text(normalized, "weight_dtype"),
        "output_dtype": _named_alias_from_text(normalized, "output_dtype"),
        "input_scale_granularity": _named_scale_from_text(
            normalized, "input_scale_granularity"
        ),
        "weight_scale_granularity": _named_scale_from_text(
            normalized, "weight_scale_granularity"
        ),
        "language": _language_from_text(normalized),
    }
    # An unqualified format/dtype mention describes the overall format.
    if result["format"] is None:
        result["format"] = _dtype_alias_from_text(normalized)
    scales = _scale_mentions(normalized)
    if result["input_scale_granularity"] is None and scales:
        result["input_scale_granularity"] = scales[0]
    if result["weight_scale_granularity"] is None and len(scales) > 1:
        result["weight_scale_granularity"] = scales[1]

    result["explicit_fields"] = {
        key for key, value in result.items() if key != "explicit_fields" and value
    }
    return result


_DTYPE_ALIASES = {
    "fp16": ("fp16", "float16", "half"),
    "bf16": ("bf16", "bfloat16"),
    "fp8": ("fp8", "float8", "e4m3", "e5m2"),
    "int8": ("int8", "i8"),
    "mxfp8": ("mxfp8", "mx-fp8", "mx fp8"),
    "mxfp4": ("mxfp4", "mx-fp4", "mx fp4"),
    "fp4": ("fp4", "float4"),
}
_SCALE_ALIASES = {
    "per_token": ("per-token", "per token", "pertoken"),
    "per_channel": ("per-channel", "per channel", "perchannel"),
    "per_block": ("per-block", "per block", "perblock"),
    "per_tensor": ("per-tensor", "per tensor", "pertensor"),
    "per_group": ("per-group", "per group", "pergroup"),
}


def _alias_pattern(aliases: tuple[str, ...]) -> str:
    return "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))


def _dtype_alias_from_text(text: str) -> str | None:
    found: list[tuple[int, str]] = []
    for canonical, aliases in _DTYPE_ALIASES.items():
        for match in re.finditer(
            r"(?<![A-Za-z0-9])(?:%s)(?![A-Za-z0-9])" % _alias_pattern(aliases),
            text,
            re.I,
        ):
            found.append((match.start(), canonical))
    if not found:
        return None
    return max(found, key=lambda item: item[0])[1]


def _named_alias_from_text(text: str, field: str) -> str | None:
    labels = {
        "format": r"(?:format|格式)",
        "input_dtype": r"(?:input(?:\s+dtype)?|activation|输入)",
        "weight_dtype": r"(?:weight(?:\s+dtype)?|weights|权重)",
        "output_dtype": r"(?:output(?:\s+dtype)?|输出)",
    }[field]
    for canonical, aliases in _DTYPE_ALIASES.items():
        value = _alias_pattern(aliases)
        if re.search(
            rf"(?:{labels})\s*[:=]?\s*(?:{value})|(?:{value})\s+(?:{labels})",
            text,
            re.I,
        ):
            return canonical
    return None


def _scale_mentions(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for canonical, aliases in _SCALE_ALIASES.items():
        for match in re.finditer(_alias_pattern(aliases), text, re.I):
            found.append((match.start(), canonical))
    return [value for _, value in sorted(found)]


def _named_scale_from_text(text: str, field: str) -> str | None:
    label = (
        r"(?:input|activation|输入)(?:\s+scale)?"
        if field == "input_scale_granularity"
        else r"(?:weight|weights|权重)(?:\s+scale)?"
    )
    for canonical, aliases in _SCALE_ALIASES.items():
        value = _alias_pattern(aliases)
        if re.search(
            rf"(?:{label})\s*[:=]?\s*(?:{value})|(?:{value})\s+(?:{label})",
            text,
            re.I,
        ):
            return canonical
    return None


def _target_gpu_from_text(text: str) -> str | None:
    match = re.search(
        r"(?<![A-Za-z0-9])(MI(?:300|308|325|350|355)(?:X)?|GFX(?:942|950)|"
        r"[HABLV]\d{2,3}|RTX\s*\d{4})(?![A-Za-z0-9])",
        text,
        re.I,
    )
    if not match:
        return None
    value = match.group(1).upper()
    if value.startswith(("MI300", "MI308", "MI325")):
        return "gfx942"
    if value.startswith(("MI350", "MI355")):
        return "gfx950"
    if value.startswith("GFX"):
        return value.lower()
    return re.sub(r"\s+", "", value)


def _target_gpu_with_model_evidence(
    request: str, value: Any, evidence_snippet: str | None = None
) -> str | None:
    if value is None:
        return None
    proposed = str(value).strip()
    if not proposed:
        return None
    normalized_proposed = _target_gpu_from_text(proposed) or proposed.lower()
    if evidence_snippet:
        normalized_evidence = _target_gpu_from_text(evidence_snippet)
        if normalized_evidence == normalized_proposed:
            return normalized_evidence
    compact_pattern = r"[\s_-]*".join(
        re.escape(part) for part in re.split(r"[\s_-]+", proposed) if part
    )
    if not re.search(
        r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % compact_pattern,
        unicodedata.normalize("NFKC", request),
        re.I,
    ):
        return None
    return _target_gpu_from_text(proposed) or proposed.upper()


def _normalize_operator(value: Any) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[\s-]+", "_", str(value).strip().lower())
    return normalized or None


def _operator_from_text(text: str) -> str | None:
    operators = (
        (
            r"(?<![A-Za-z0-9])(?:gemm(?:[_\s-]*kernel)?|matmul|"
            r"matrix\s+multiplication)"
            r"(?![A-Za-z0-9])|矩阵乘法",
            "gemm",
        ),
        (
            r"(?<![A-Za-z0-9])(?:flash[_\s-]?)?attention(?![A-Za-z0-9])|注意力",
            "attention",
        ),
        (
            r"(?<![A-Za-z0-9])conv(?:olution)?(?:[123]d)?(?![A-Za-z0-9])|卷积",
            "convolution",
        ),
        (r"(?<![A-Za-z0-9])softmax(?![A-Za-z0-9])", "softmax"),
        (
            r"(?<![A-Za-z0-9])(?:layer|rms|batch)[_\s-]?"
            r"norm(?:alization)?(?![A-Za-z0-9])",
            "normalization",
        ),
        (
            r"(?<![A-Za-z0-9])(?:embedding|scatter|gather|reduction|reduce|"
            r"transpose)(?![A-Za-z0-9])",
            None,
        ),
    )
    for pattern, canonical in operators:
        match = re.search(pattern, text, re.I)
        if match:
            return canonical or _normalize_operator(match.group(0))
    return None


def _number_is_in_request(request: str, number: int) -> bool:
    text = unicodedata.normalize("NFKC", request)
    return bool(re.search(r"(?<!\d)%s(?!\d)" % re.escape(str(number)), text))


def _validated_number(request: str, value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid %s" % name) from exc
    if number < 1 or not _number_is_in_request(request, number):
        raise ValueError(
            "recognizer proposed %s=%d without matching numeric evidence"
            % (name, number)
        )
    return number


def _validated_dimensions(request: str, value: Any) -> dict[str, int]:
    if value in (None, {}):
        return {}
    if not isinstance(value, dict):
        raise ValueError("dimensions must be a mapping")
    return {
        str(name): _validated_number(request, number, "dimension %s" % name)
        for name, number in value.items()
    }


def _validated_shapes(request: str, value: Any) -> list[list[int]]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ValueError("shapes must be a list")
    shapes: list[list[int]] = []
    for shape in value:
        if not isinstance(shape, (list, tuple)):
            raise ValueError("each shape must be a list")
        shapes.append(
            [_validated_number(request, number, "shape") for number in shape]
        )
    return shapes


def _dimensions_from_text(text: str) -> dict[str, int]:
    dimensions: dict[str, int] = {}
    pattern = re.compile(
        r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9_]*)\s*"
        r"(?:=|:|(?<![A-Za-z0-9])is(?![A-Za-z0-9])|为)\s*"
        r"(\d+)(?!\d)",
        re.I,
    )
    ignored = {"mi", "gfx", "block", "block_size"}
    for match in pattern.finditer(unicodedata.normalize("NFKC", text)):
        name = match.group(1).lower()
        if name not in ignored:
            dimensions[name] = int(match.group(2))
    return dimensions


def _gemm_dimensions_from_text(text: str) -> dict[str, int]:
    normalized = unicodedata.normalize("NFKC", text)
    result = {
        name: int(value)
        for name, value in re.findall(
            r"(?<![A-Za-z0-9])([mnk])\s*(?:=|:)\s*(\d+)(?!\d)",
            normalized,
            re.I,
        )
    }
    if len(result) == 3:
        return result
    ordered = list(
        re.finditer(
            r"(?<![A-Za-z0-9])([mnk]{3})(?![A-Za-z0-9])[^\d]{0,20}"
            r"(\d+)\s*[/xX×,]\s*(\d+)\s*[/xX×,]\s*(\d+)",
            normalized,
            re.I,
        )
    )
    if ordered:
        match = ordered[-1]
        return {
            name.lower(): int(value)
            for name, value in zip(match.group(1), match.groups()[1:])
        }
    matches = list(
        re.finditer(
            r"(?:dimensions?|shape|参数)[^\d]{0,20}"
            r"(\d+)\s*[/xX×,]\s*(\d+)\s*[/xX×,]\s*(\d+)",
            normalized,
            re.I,
        )
    )
    if matches:
        return dict(zip(("m", "n", "k"), map(int, matches[-1].groups())))
    return result


def _shapes_from_text(text: str) -> list[list[int]]:
    shapes: list[list[int]] = []
    pattern = re.compile(r"(?<!\d)\d+(?:\s*[xX×,/]\s*\d+)+(?!\d)")
    for match in pattern.finditer(unicodedata.normalize("NFKC", text)):
        shape = [int(value) for value in re.findall(r"\d+", match.group(0))]
        if len(shape) > 1:
            shapes.append(shape)
    return shapes


def _block_size_from_text(text: str) -> int | None:
    match = re.search(
        r"(?<![A-Za-z0-9])block(?:[_\s-]?size)?\s*"
        r"(?:=|:|(?<![A-Za-z0-9])is(?![A-Za-z0-9]))?\s*(\d+)(?!\d)",
        text,
        re.I,
    )
    return int(match.group(1)) if match else None


def _validate_model_result(
    request: str, payload: dict[str, Any]
) -> dict[str, Any]:
    operator = str(payload.get("operator") or "").lower().replace("-", "_")
    if operator not in {"gemm", "dense_gemm"}:
        raise ValueError("request was not recognized as dense GEMM")
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("recognizer returned no confidence") from exc
    if confidence < 0.5:
        raise ValueError("recognizer confidence is too low: %.3f" % confidence)

    text = unicodedata.normalize("NFKC", request)
    dimensions: dict[str, int] = {}
    for name in ("m", "n", "k"):
        try:
            number = int(payload.get(name))
        except (TypeError, ValueError) as exc:
            raise ValueError("recognizer could not identify %s" % name.upper()) from exc
        if number < 1:
            raise ValueError("%s must be positive" % name.upper())
        if not re.search(r"(?<!\d)%s(?!\d)" % re.escape(str(number)), text):
            raise ValueError(
                "recognizer proposed %s=%d without matching numeric evidence"
                % (name.upper(), number)
            )
        dimensions[name] = number

    # Model format/language values are only preselection hints. Never let them
    # introduce a contract that has no evidence in the original request.
    dtype = _format_from_text(request) or "fp16"

    target_gpu = str(payload.get("target_gpu") or "AMD-Instinct").upper()
    if target_gpu == "MI308":
        target_gpu = "MI308X"
    language = _language_from_text(request)
    return {
        "target_gpu": target_gpu,
        "dtype": dtype,
        **dimensions,
        "request": request.strip(),
        "language": language or "triton",
        "confidence": confidence,
        "recognition": "model",
    }


def _language_from_text(text: str) -> str | None:
    matches = list(
        re.finditer(
            r"(?<![A-Za-z0-9])(triton|hip|rocm\s*c\+\+|fly\s*dsl|flydsl)"
            r"(?![A-Za-z0-9])",
            text,
            re.I,
        )
    )
    return _normalize_language(matches[-1].group(1)) if matches else None


def _format_from_text(text: str) -> str | None:
    aliases = "|".join(
        sorted((re.escape(value) for value in FORMAT_ALIASES), key=len, reverse=True)
    )
    match = re.search(
        r"(?<![A-Za-z0-9])(%s)(?![A-Za-z0-9])" % aliases,
        unicodedata.normalize("NFKC", text),
        re.I,
    )
    return normalize_gemm_format(match.group(1)) if match else None


def _normalize_language(value: Any) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[\s_-]+", "", str(value).lower())
    aliases = {
        "triton": "triton",
        "hip": "hip",
        "rocmc++": "hip",
        "flydsl": "flydsl",
    }
    if normalized not in aliases:
        raise ValueError("unsupported recognized language: %s" % value)
    return aliases[normalized]
