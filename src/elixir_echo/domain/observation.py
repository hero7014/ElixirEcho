"""Actor-conditioned current-state observation contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .values import (
    DomainContractError,
    ObservedValue,
    content_sha256,
    frozen_map,
    jsonable,
)


OBSERVATION_SCHEMA = "elixir_echo.observation@1"

_FORBIDDEN_OBSERVATION_KEYS = frozenset(
    {
        "episode_id",
        "tick",
        "actor",
        "action_mask",
        "expert_action",
        "event_delta",
        "validation",
        "fingerprint",
        "fingerprints",
        "provenance",
        "runtime_provenance",
        "field_evidence",
        "source_fields",
        "observed_tick",
        "native_digest",
        "metadata",
        "internal",
        "raster",
        "opponent_belief",
        "pending_actions",
        "events",
        "deck",
        "cycle",
        "history",
        "tracker",
        "belief",
        "memory",
    }
)


def _assert_observation_keys(value: Any, path: str = "observation") -> None:
    if isinstance(value, ObservedValue):
        _assert_observation_keys(value.value, path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_OBSERVATION_KEYS or "provenance" in normalized:
                raise DomainContractError(f"forbidden Observation field {path}.{key}")
            _assert_observation_keys(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_observation_keys(item, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class Observation:
    """Current game semantics available to one actor at one decision time."""

    match: Mapping[str, Any]
    players: tuple[Mapping[str, Any], ...]
    towers: tuple[Mapping[str, Any], ...]
    entities: tuple[Mapping[str, Any], ...]
    terminal: Mapping[str, Any]
    perspective: str = "actor_fair"

    def __post_init__(self) -> None:
        if self.perspective != "actor_fair":
            raise DomainContractError("Observation perspective must be actor_fair")
        object.__setattr__(self, "match", frozen_map(self.match))
        object.__setattr__(
            self,
            "players",
            tuple(frozen_map(item) for item in self.players),
        )
        object.__setattr__(
            self,
            "towers",
            tuple(frozen_map(item) for item in self.towers),
        )
        object.__setattr__(
            self,
            "entities",
            tuple(frozen_map(item) for item in self.entities),
        )
        object.__setattr__(self, "terminal", frozen_map(self.terminal))
        _assert_observation_keys(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "perspective": self.perspective,
            "match": jsonable(self.match),
            "players": jsonable(self.players),
            "towers": jsonable(self.towers),
            "entities": jsonable(self.entities),
            "terminal": jsonable(self.terminal),
        }

    def content_sha256(self) -> str:
        return content_sha256(self.to_dict())
