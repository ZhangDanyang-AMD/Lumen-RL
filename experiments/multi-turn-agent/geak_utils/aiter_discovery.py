"""Read-only discovery of relevant source files in an AITER checkout.

This module deliberately treats AITER as data.  It never imports the package,
loads shared libraries, or invokes a command from the checkout.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence


HIGH_CONFIDENCE_SCORE = 85
_TEXT_SUFFIXES = {
    ".cc",
    ".cpp",
    ".cuh",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
_SOURCE_SUFFIXES = {".cc", ".cpp", ".cuh", ".h", ".hpp", ".py"}
_MAX_TEXT_BYTES = 2 * 1024 * 1024
_TOKEN_RE = re.compile(r"[a-z]+|\d+")
_ARCH_RE = re.compile(r"(?<![a-z0-9])(?:gfx\d{3,4}|mi\d{3,4}x?)(?![a-z0-9])", re.I)
_REFERENCE_RE = re.compile(
    r"\b(?:reference|ref_impl|golden|naive)[a-z0-9_]*\b"
    r"|torch\.(?:matmul|mm|bmm|softmax)\b"
    r"|(?:torch\.nn\.functional|f)\.scaled_dot_product_attention\b",
    re.I,
)
_GRANULARITY_PATTERNS = {
    "per_tensor": re.compile(r"\bper[_ -]?tensor\b", re.I),
    "per_token": re.compile(r"\bper[_ -]?token\b", re.I),
    "per_channel": re.compile(r"\bper[_ -]?channel\b", re.I),
    "per_group": re.compile(r"\bper[_ -]?group\b", re.I),
    "per_block": re.compile(r"\b(?:per[_ -]?block|block[_ -]?wise)\b", re.I),
}
_ARCH_ALIASES = {
    "mi300": "gfx942",
    "mi300x": "gfx942",
    "mi308": "gfx942",
    "mi308x": "gfx942",
    "mi325": "gfx942",
    "mi325x": "gfx942",
    "mi350": "gfx950",
    "mi350x": "gfx950",
    "mi355": "gfx950",
    "mi355x": "gfx950",
}
_VALUE_ALIASES = {
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float8": "fp8",
    "int8_t": "int8",
    "blockwise": "per_block",
    "block_wise": "per_block",
    "tensorwise": "per_tensor",
    "tokenwise": "per_token",
    "channelwise": "per_channel",
    "groupwise": "per_group",
}
_OPERATOR_STOPWORDS = {"aiter", "op", "ops", "operator", "kernel"}
_GENERIC_FILE_TOKENS = {
    "aiter",
    "benchmark",
    "benchmarks",
    "config",
    "configs",
    "op",
    "ops",
    "test",
    "tests",
}


def _normalize_value(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not normalized:
        return None
    return _VALUE_ALIASES.get(normalized, normalized)


def _normalize_architecture(value: object | None) -> str | None:
    normalized = _normalize_value(value)
    if normalized is None:
        return None
    compact = normalized.replace("_", "")
    return _ARCH_ALIASES.get(compact, compact)


def _normalize_granularity(value: object | None) -> str | None:
    normalized = _normalize_value(value)
    if normalized in {"tensor", "token", "channel", "group", "block"}:
        return "per_" + normalized
    return normalized


def _normalize_shapes(
    shapes: Sequence[int] | Iterable[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...]:
    if shapes is None:
        return ()
    values = tuple(shapes)
    if not values:
        return ()
    if all(isinstance(value, int) for value in values):
        values = (values,)  # type: ignore[assignment]
    normalized: list[tuple[int, ...]] = []
    for shape in values:
        dimensions = tuple(int(dimension) for dimension in shape)  # type: ignore[arg-type]
        if not dimensions or any(dimension <= 0 for dimension in dimensions):
            raise ValueError("shapes must contain only positive dimensions")
        normalized.append(dimensions)
    return tuple(normalized)


@dataclass(frozen=True)
class AiterQuery:
    """Normalized description of an AITER operator to locate."""

    operator: str
    input_dtype: str | None = None
    weight_dtype: str | None = None
    input_format: str | None = None
    weight_format: str | None = None
    input_scale_granularity: str | None = None
    weight_scale_granularity: str | None = None
    block_size: int | tuple[int, ...] | None = None
    architecture: str | None = None
    shapes: tuple[tuple[int, ...], ...] | Sequence[int] | Iterable[Sequence[int]] = ()
    backend: str | None = "aiter"

    def __post_init__(self) -> None:
        operator = _normalize_value(self.operator)
        if operator is None:
            raise ValueError("operator must be non-empty")
        object.__setattr__(self, "operator", operator)
        for field_name in (
            "input_dtype",
            "weight_dtype",
            "input_format",
            "weight_format",
            "backend",
        ):
            object.__setattr__(
                self, field_name, _normalize_value(getattr(self, field_name))
            )
        for field_name in (
            "input_scale_granularity",
            "weight_scale_granularity",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_granularity(getattr(self, field_name)),
            )
        object.__setattr__(
            self, "architecture", _normalize_architecture(self.architecture)
        )
        object.__setattr__(self, "shapes", _normalize_shapes(self.shapes))

        block_size = self.block_size
        if block_size is not None:
            if isinstance(block_size, int):
                normalized_block: int | tuple[int, ...] = int(block_size)
                if normalized_block <= 0:
                    raise ValueError("block_size must be positive")
            else:
                normalized_block = tuple(int(value) for value in block_size)
                if not normalized_block or any(value <= 0 for value in normalized_block):
                    raise ValueError("block_size must contain positive integers")
            object.__setattr__(self, "block_size", normalized_block)


@dataclass(frozen=True)
class AiterCandidate:
    """Immutable evidence bundle for one possible AITER wrapper."""

    wrapper_path: Path
    test_paths: tuple[Path, ...] = ()
    reference_paths: tuple[Path, ...] = ()
    benchmark_paths: tuple[Path, ...] = ()
    config_paths: tuple[Path, ...] = ()
    supported_architectures: tuple[str, ...] = ()
    score: int = 0
    reasons: tuple[str, ...] = ()

    @property
    def high_confidence(self) -> bool:
        return self.score >= HIGH_CONFIDENCE_SCORE


@dataclass(frozen=True)
class _FileRecord:
    path: Path
    relative: str
    path_tokens: frozenset[str]
    text: str
    architectures: tuple[str, ...]
    granularities: frozenset[str]
    explicit_reference: bool


class AiterDiscoveryIndex:
    """A deterministic, in-memory index of a configured AITER root."""

    def __init__(self, aiter_root: Path | str) -> None:
        root = Path(aiter_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("AITER root is not a directory: %s" % root)
        records = tuple(self._scan(root))
        self._root = root
        self._records = records
        self._by_path: Mapping[Path, _FileRecord] = MappingProxyType(
            {record.path: record for record in records}
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(record.path for record in self._records)

    def _scan(self, root: Path) -> Iterable[_FileRecord]:
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            current = Path(directory)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not (current / name).is_symlink()
                and name not in {".git", "__pycache__", ".pytest_cache"}
            )
            for filename in sorted(filenames):
                lexical_path = current / filename
                if lexical_path.suffix.lower() not in _TEXT_SUFFIXES:
                    continue
                try:
                    path = lexical_path.resolve(strict=True)
                    path.relative_to(root)
                    if not path.is_file() or lexical_path.is_symlink():
                        continue
                    size = path.stat().st_size
                    if size > _MAX_TEXT_BYTES:
                        continue
                    text = path.read_text(encoding="utf-8", errors="replace")
                except (OSError, RuntimeError, ValueError):
                    continue
                relative = path.relative_to(root).as_posix()
                searchable = "%s\n%s" % (relative, text)
                architectures = tuple(
                    sorted(
                        {
                            architecture
                            for match in _ARCH_RE.findall(searchable)
                            if (architecture := _normalize_architecture(match)) is not None
                        }
                    )
                )
                granularities = frozenset(
                    name
                    for name, pattern in _GRANULARITY_PATTERNS.items()
                    if pattern.search(searchable)
                )
                yield _FileRecord(
                    path=path,
                    relative=relative,
                    path_tokens=frozenset(_tokens(relative)),
                    text=text,
                    architectures=architectures,
                    granularities=granularities,
                    explicit_reference=bool(_REFERENCE_RE.search(text)),
                )

    def search(
        self, query: AiterQuery, *, minimum_score: int = 0
    ) -> tuple[AiterCandidate, ...]:
        """Return matching candidates ordered by score, then relative path."""

        if not isinstance(query, AiterQuery):
            raise TypeError("query must be an AiterQuery")
        if minimum_score < 0:
            raise ValueError("minimum_score must be non-negative")

        operator_tokens = set(_tokens(query.operator)) - _OPERATOR_STOPWORDS
        if not operator_tokens:
            operator_tokens = set(_tokens(query.operator))
        relevant = tuple(
            record
            for record in self._records
            if operator_tokens <= record.path_tokens
        )
        wrappers = tuple(record for record in relevant if _is_wrapper(record))
        assets = tuple(record for record in relevant if not _is_wrapper(record))
        candidates: list[AiterCandidate] = []
        for wrapper in wrappers:
            related_assets = tuple(
                record
                for record in assets
                if _is_related(wrapper, record, operator_tokens)
            )
            evidence = (wrapper,) + related_assets
            architectures = tuple(
                sorted(
                    {
                        architecture
                        for record in evidence
                        for architecture in record.architectures
                    }
                )
            )
            architecture_gates = wrapper.architectures or architectures
            if (
                query.architecture
                and architecture_gates
                and query.architecture not in architecture_gates
            ):
                continue
            granularities = frozenset(
                granularity
                for record in evidence
                for granularity in record.granularities
            )
            requested_granularities = tuple(
                granularity
                for granularity in (
                    query.input_scale_granularity,
                    query.weight_scale_granularity,
                )
                if granularity
            )
            scale_gates = wrapper.granularities or granularities
            if (
                requested_granularities
                and scale_gates
                and any(value not in scale_gates for value in requested_granularities)
            ):
                continue

            tests = _paths(related_assets, _is_test)
            references = _paths(
                related_assets, _is_reference
            )
            benchmarks = _paths(related_assets, _is_benchmark)
            configs = _paths(related_assets, _is_config)
            score = 30
            reasons = ["operator tokens match wrapper path"]
            if tests:
                score += 25
                reasons.append("dedicated mirrored op test")
            if references:
                score += 35
                reasons.append("explicit independent reference in test")
            if benchmarks:
                score += 5
                reasons.append("matching benchmark")
            if configs:
                score += 3
                reasons.append("matching config")

            format_tokens = {
                value
                for value in (
                    query.input_dtype,
                    query.weight_dtype,
                    query.input_format,
                    query.weight_format,
                )
                if value
            }
            matched_formats = sorted(
                value
                for value in format_tokens
                if any(set(_tokens(value)) <= record.path_tokens for record in evidence)
            )
            if matched_formats:
                score += min(2 * len(matched_formats), 6)
                reasons.append("format tokens match: %s" % ", ".join(matched_formats))
            if query.block_size is not None:
                block_values = (
                    (query.block_size,)
                    if isinstance(query.block_size, int)
                    else query.block_size
                )
                if all(
                    any(str(value) in record.path_tokens for record in evidence)
                    for value in block_values
                ):
                    score += 2
                    reasons.append("block-size tokens match")

            candidate = AiterCandidate(
                wrapper_path=wrapper.path,
                test_paths=tests,
                reference_paths=references,
                benchmark_paths=benchmarks,
                config_paths=configs,
                supported_architectures=architectures,
                score=min(score, 100),
                reasons=tuple(reasons),
            )
            if candidate.score >= minimum_score:
                candidates.append(candidate)
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.wrapper_path.relative_to(self._root).as_posix(),
                ),
            )
        )


def build_aiter_index(aiter_root: Path | str) -> AiterDiscoveryIndex:
    """Build a read-only index for *aiter_root*."""

    return AiterDiscoveryIndex(aiter_root)


def discover_aiter(
    aiter_root: Path | str,
    query: AiterQuery,
    *,
    minimum_score: int = 0,
) -> tuple[AiterCandidate, ...]:
    """Build an index and search it in one call."""

    return AiterDiscoveryIndex(aiter_root).search(query, minimum_score=minimum_score)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(value.lower()))


def _is_wrapper(record: _FileRecord) -> bool:
    parts = record.relative.lower().split("/")
    return (
        record.path.suffix.lower() in _SOURCE_SUFFIXES
        and "aiter" in parts
        and "ops" in parts
        and not _is_test(record)
        and not _looks_like_reference_path(record)
        and record.path.name.lower() != "__init__.py"
    )


def _is_test(record: _FileRecord) -> bool:
    parts = record.relative.lower().split("/")
    return (
        "op_tests" in parts
        or "tests" in parts
        or record.path.name.lower().startswith(("test_", "test-"))
    ) and record.path.suffix.lower() in _SOURCE_SUFFIXES


def _is_benchmark(record: _FileRecord) -> bool:
    parts = record.relative.lower().split("/")
    name = record.path.stem.lower()
    return any(part in {"benchmark", "benchmarks", "bench"} for part in parts) or name.startswith(
        ("bench_", "benchmark_")
    )


def _is_config(record: _FileRecord) -> bool:
    parts = record.relative.lower().split("/")
    return record.path.suffix.lower() in {".json", ".yaml", ".yml"} and (
        any(part in {"config", "configs"} for part in parts)
        or "config" in record.path.stem.lower()
    )


def _looks_like_reference_path(record: _FileRecord) -> bool:
    parts = record.relative.lower().split("/")
    stem = record.path.stem.lower()
    return any(part in {"reference", "references"} for part in parts) or stem.startswith(
        ("ref_", "reference_")
    )


def _is_reference(record: _FileRecord) -> bool:
    return record.explicit_reference and (
        _is_test(record) or _looks_like_reference_path(record)
    )


def _identity_tokens(record: _FileRecord) -> frozenset[str]:
    return record.path_tokens - _GENERIC_FILE_TOKENS


def _is_related(
    wrapper: _FileRecord, asset: _FileRecord, operator_tokens: set[str]
) -> bool:
    if not (
        _is_test(asset)
        or _is_benchmark(asset)
        or _is_config(asset)
        or _is_reference(asset)
    ):
        return False
    wrapper_identity = _identity_tokens(wrapper) - operator_tokens
    asset_identity = _identity_tokens(asset) - operator_tokens
    # Mirrored op_tests commonly retain the wrapper's operation-specific stem.
    return not wrapper_identity or not asset_identity or bool(
        wrapper_identity & asset_identity
    )


def _paths(
    records: Iterable[_FileRecord], predicate: Callable[[_FileRecord], bool]
) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (record.path for record in records if predicate(record)),
            key=lambda path: path.as_posix(),
        )
    )


__all__ = [
    "AiterCandidate",
    "AiterDiscoveryIndex",
    "AiterQuery",
    "HIGH_CONFIDENCE_SCORE",
    "build_aiter_index",
    "discover_aiter",
]
