"""Persistent registry for locally GPU-verified generated templates."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FILES = (
    "config.yaml",
    "kernel.py",
    "scripts/task_runner.py",
    "metadata.json",
)
_RECORD_FIELDS = {
    "contract_hash",
    "operator",
    "case_type",
    "template_path",
    "architecture",
    "language",
    "backend",
    "provenance",
    "direction",
}
_REQUIRED_RECORD_FIELDS = _RECORD_FIELDS - {"case_type"}


@dataclass(frozen=True)
class VerifiedTemplateRecord:
    """One immutable reference to a trusted, locally verified template."""

    contract_hash: str
    operator: str
    template_path: Path
    architecture: str
    language: str
    backend: str
    provenance: Any
    direction: str
    case_type: str = "aiter_generated"

    def __post_init__(self) -> None:
        if not isinstance(self.contract_hash, str) or not _HASH_RE.fullmatch(
            self.contract_hash
        ):
            raise ValueError("contract_hash must be 64 lowercase hexadecimal characters")
        object.__setattr__(self, "template_path", Path(self.template_path).expanduser())
        for name in (
            "operator",
            "case_type",
            "architecture",
            "language",
            "backend",
            "direction",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("%s must be a non-empty string" % name)
            object.__setattr__(self, name, value.strip())
        if self.provenance is None:
            raise ValueError("provenance must not be null")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: Path | str | None = None,
        verified_root: Path | str | None = None,
    ) -> "VerifiedTemplateRecord":
        """Parse and validate a registry entry."""

        if not isinstance(value, Mapping):
            raise ValueError("verified template entry must be a mapping")
        keys = set(value)
        missing = sorted(_REQUIRED_RECORD_FIELDS - keys)
        unknown = sorted(keys - _RECORD_FIELDS)
        if missing:
            raise ValueError(
                "verified template entry is missing field(s): %s" % ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "verified template entry has unknown field(s): %s" % ", ".join(unknown)
            )
        raw_path = value.get("template_path")
        if not isinstance(raw_path, (str, os.PathLike)) or not str(raw_path).strip():
            raise ValueError("template_path must be a non-empty path")
        template_path = Path(raw_path).expanduser()
        if not template_path.is_absolute():
            template_path = Path(base_dir or Path.cwd()) / template_path
        record = cls(
            contract_hash=value.get("contract_hash"),  # type: ignore[arg-type]
            operator=value.get("operator"),  # type: ignore[arg-type]
            case_type=value.get("case_type", "aiter_generated"),  # type: ignore[arg-type]
            template_path=template_path,
            architecture=value.get("architecture"),  # type: ignore[arg-type]
            language=value.get("language"),  # type: ignore[arg-type]
            backend=value.get("backend"),  # type: ignore[arg-type]
            provenance=value.get("provenance"),
            direction=value.get("direction"),  # type: ignore[arg-type]
        )
        return _validated_record(record, verified_root=verified_root)

    def to_mapping(self) -> dict[str, Any]:
        """Return the stable YAML representation of this record."""

        return {
            "contract_hash": self.contract_hash,
            "operator": self.operator,
            "case_type": self.case_type,
            "template_path": str(self.template_path),
            "architecture": self.architecture,
            "language": self.language,
            "backend": self.backend,
            "provenance": self.provenance,
            "direction": self.direction,
        }


def load_verified_templates(
    path: Path | str, *, verified_root: Path | str | None = None
) -> list[VerifiedTemplateRecord]:
    """Load, validate, and deterministically order a local template registry."""

    registry_path = _registry_path(path)
    payload = _load_registry_payload(registry_path)
    raw_records = payload.get("templates", [])
    if not isinstance(raw_records, list):
        raise ValueError("verified template registry 'templates' must be a list")

    by_hash: dict[str, VerifiedTemplateRecord] = {}
    for index, raw_record in enumerate(raw_records):
        try:
            record = VerifiedTemplateRecord.from_mapping(
                raw_record,
                base_dir=registry_path.parent,
                verified_root=verified_root,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                "invalid verified template entry %d: %s" % (index, exc)
            ) from exc
        existing = by_hash.get(record.contract_hash)
        if existing is not None:
            if existing != record:
                raise ValueError(
                    "conflicting duplicate contract_hash: %s" % record.contract_hash
                )
            continue
        by_hash[record.contract_hash] = record
    return [by_hash[key] for key in sorted(by_hash)]


def find_verified_template(
    path: Path | str,
    contract_hash: str,
    *,
    verified_root: Path | str | None = None,
) -> VerifiedTemplateRecord | None:
    """Return the trusted record for *contract_hash*, if present."""

    if not isinstance(contract_hash, str) or not _HASH_RE.fullmatch(contract_hash):
        raise ValueError("contract_hash must be 64 lowercase hexadecimal characters")
    return next(
        (
            record
            for record in load_verified_templates(path, verified_root=verified_root)
            if record.contract_hash == contract_hash
        ),
        None,
    )


def register_verified_template(
    path: Path | str,
    record: VerifiedTemplateRecord,
    *,
    verified_root: Path | str | None = None,
) -> VerifiedTemplateRecord:
    """Atomically add one trusted record, or leave an identical entry unchanged."""

    if not isinstance(record, VerifiedTemplateRecord):
        raise TypeError("record must be a VerifiedTemplateRecord")
    registry_path = _registry_path(path)
    normalized_path = record.template_path
    if not normalized_path.is_absolute():
        normalized_path = registry_path.parent / normalized_path
    normalized = _validated_record(
        VerifiedTemplateRecord(
            contract_hash=record.contract_hash,
            operator=record.operator,
            case_type=record.case_type,
            template_path=normalized_path,
            architecture=record.architecture,
            language=record.language,
            backend=record.backend,
            provenance=record.provenance,
            direction=record.direction,
        ),
        verified_root=verified_root,
    )

    payload = _load_registry_payload(registry_path)
    existing_records = load_verified_templates(
        registry_path, verified_root=verified_root
    )
    by_hash = {item.contract_hash: item for item in existing_records}
    existing = by_hash.get(normalized.contract_hash)
    if existing is not None:
        if existing != normalized:
            raise ValueError(
                "conflicting duplicate contract_hash: %s" % normalized.contract_hash
            )
        return existing

    by_hash[normalized.contract_hash] = normalized
    payload["templates"] = [
        by_hash[key].to_mapping() for key in sorted(by_hash)
    ]
    _atomic_write_yaml(registry_path, payload)
    return normalized


def _registry_path(path: Path | str) -> Path:
    registry_path = Path(path).expanduser().absolute()
    if registry_path.is_symlink():
        raise ValueError("registry path must not be a symlink")
    return registry_path


def _load_registry_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError("verified template registry is not a regular file: %s" % path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("invalid verified template registry YAML: %s" % exc) from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("verified template registry root must be a mapping")
    return dict(payload)


def _validated_record(
    record: VerifiedTemplateRecord,
    *,
    verified_root: Path | str | None,
) -> VerifiedTemplateRecord:
    supplied = record.template_path
    _reject_symlink_components(supplied, "template path")
    try:
        template_path = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("template path cannot be resolved: %s" % supplied) from exc
    if not template_path.is_dir():
        raise ValueError("template path is not a directory: %s" % template_path)

    if verified_root is not None:
        root_supplied = Path(verified_root).expanduser()
        _reject_symlink_components(root_supplied, "verified_root")
        try:
            root = root_supplied.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("verified_root cannot be resolved") from exc
        if not root.is_dir():
            raise ValueError("verified_root is not a directory: %s" % root)
        try:
            template_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "template path escapes verified_root: %s" % template_path
            ) from exc

    for entry in template_path.rglob("*"):
        if entry.is_symlink():
            raise ValueError("template directory contains a symlink: %s" % entry)
    for relative in _REQUIRED_FILES:
        required = template_path / relative
        if required.is_symlink() or not required.is_file():
            raise ValueError("template is missing required regular file: %s" % relative)

    metadata_path = template_path / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("template metadata.json is invalid: %s" % exc) from exc
    if not isinstance(metadata, dict):
        raise ValueError("template metadata.json must contain an object")
    if metadata.get("contract_hash") != record.contract_hash:
        raise ValueError("template metadata contract_hash does not match registry record")
    trust = metadata.get("trust")
    if not isinstance(trust, dict) or trust.get("trusted") is not True:
        raise ValueError("template metadata must contain trust.trusted=true")

    return VerifiedTemplateRecord(
        contract_hash=record.contract_hash,
        operator=record.operator,
        case_type=record.case_type,
        template_path=template_path,
        architecture=record.architecture,
        language=record.language,
        backend=record.backend,
        provenance=record.provenance,
        direction=record.direction,
    )


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("%s must not contain symlinks: %s" % (label, current))


def _atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("registry path must not be a symlink")
    text = yaml.safe_dump(
        dict(payload), sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".%s." % path.name,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


__all__ = [
    "VerifiedTemplateRecord",
    "find_verified_template",
    "load_verified_templates",
    "register_verified_template",
]
