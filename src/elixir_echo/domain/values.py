"""Immutable semantic values and deterministic content identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


class DomainContractError(ValueError):
    """Raised when a value violates a canonical domain boundary."""


class _FrozenMap(Mapping[str, Any]):
    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any] | None = None) -> None:
        self._data = MappingProxyType(
            {str(key): freeze(item) for key, item in (value or {}).items()}
        )

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def freeze(value: Any) -> Any:
    """Recursively freeze collection values used by domain contracts."""

    if isinstance(value, (_FrozenMap, ObservedValue)):
        return value
    if isinstance(value, Mapping):
        return _FrozenMap(value)
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((freeze(item) for item in value), key=repr))
    return value


def frozen_map(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Return an immutable recursively frozen mapping."""

    return _FrozenMap(value)


def jsonable(value: Any) -> Any:
    """Convert canonical domain values to deterministic JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, _FrozenMap):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return jsonable(to_dict())
    if is_dataclass(value):
        return {item.name: jsonable(getattr(value, item.name)) for item in fields(value)}
    return value


def canonical_json(value: Any) -> str:
    """Encode a domain value with stable ordering and no non-finite numbers."""

    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_sha256(value: Any) -> str:
    """Return the SHA-256 identity of a value's canonical JSON representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservedValue:
    """A fact that is known, unknown, or inapplicable to the current entity."""

    status: str
    value: Any = None

    def __post_init__(self) -> None:
        if self.status not in {"known", "unknown", "not_applicable"}:
            raise DomainContractError(f"invalid observed value status {self.status!r}")
        if self.status != "known" and self.value is not None:
            raise DomainContractError(
                "unknown/not_applicable observed values cannot carry a value"
            )
        object.__setattr__(self, "value", freeze(self.value))

    @classmethod
    def known(cls, value: Any) -> "ObservedValue":
        return cls("known", value)

    @classmethod
    def unknown(cls) -> "ObservedValue":
        return cls("unknown")

    @classmethod
    def not_applicable(cls) -> "ObservedValue":
        return cls("not_applicable")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.status == "known":
            payload["value"] = jsonable(self.value)
        return payload
