"""Model-assisted natural-language task recognition with strict validation."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

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
    match = re.search(r"\b(triton|hip|rocm\s*c\+\+|fly\s*dsl|flydsl)\b", text, re.I)
    return _normalize_language(match.group(1)) if match else None


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
