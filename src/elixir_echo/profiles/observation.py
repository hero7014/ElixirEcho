"""Approved model-facing field selections over an Observation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from elixir_echo.domain import Observation


class ObservationProfile(str, Enum):
    """Stable public names for the two initial observation selections."""

    CORE = "core"
    FULL = "full"


_MATCH = ("phase", "remaining_ms", "elixir_multiplier")
_PLAYER_BASE = ("owner", "relation", "crowns", "tower_ids", "self_private")
_EVOLUTION_CORE = ("card_id", "current_form_id", "ready", "active")
_TOWER_CORE = (
    "entity_id",
    "owner",
    "tower_kind",
    "position",
    "hitpoints",
    "max_hitpoints",
    "tower_troop_id",
    "active",
    "shield",
)
_TOWER_FULL = _TOWER_CORE + ("attack", "effects", "visibility")
_ENTITY_CORE = (
    "entity_id",
    "owner",
    "card_id",
    "current_form",
    "entity_kind",
    "position",
    "hitpoints",
    "max_hitpoints",
    "shield",
    "visibility_basic",
)
_ENTITY_FULL = _ENTITY_CORE + (
    "velocity",
    "age_ms",
    "public_target_id",
    "source_entity_id",
    "causal_relation",
    "attack",
    "effects",
    "projectile",
    "visibility",
    "abilities",
    "evolution",
)
_STABLE_CAUSAL_RELATION = ("kind", "source_card_id", "parent_entity_id")


def _select(source: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: source[name] for name in names if name in source}


def _player(source: Mapping[str, Any], *, full: bool) -> dict[str, Any]:
    result = _select(source, _PLAYER_BASE)
    evolution = source.get("current_evolution")
    if isinstance(evolution, (tuple, list)):
        result["current_evolution"] = (
            list(evolution)
            if full
            else [_select(item, _EVOLUTION_CORE) for item in evolution]
        )
    if full and "current_abilities" in source:
        result["current_abilities"] = source["current_abilities"]
    return result


def _entity(source: Mapping[str, Any], *, full: bool) -> dict[str, Any]:
    result = _select(source, _ENTITY_FULL if full else _ENTITY_CORE)
    relation = source.get("causal_relation")
    if full and isinstance(relation, Mapping):
        result["causal_relation"] = _select(relation, _STABLE_CAUSAL_RELATION)
    return result


def select_observation(
    observation: Observation, profile: ObservationProfile
) -> Mapping[str, object]:
    """Select a model-facing view without access to episode context or history."""

    profile = ObservationProfile(profile)
    payload = observation.to_dict()
    full = profile is ObservationProfile.FULL
    return {
        "profile": profile.value,
        "observation_schema": payload["schema"],
        "match": _select(payload["match"], _MATCH),
        "players": [_player(item, full=full) for item in payload["players"]],
        "towers": [
            _select(item, _TOWER_FULL if full else _TOWER_CORE)
            for item in payload["towers"]
        ],
        "entities": [_entity(item, full=full) for item in payload["entities"]],
    }
