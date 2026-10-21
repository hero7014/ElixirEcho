"""Translate one FirstLight FAIR observation into the canonical Observation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

from elixir_echo.domain import Observation, ObservedValue


class FirstLightAdapterError(ValueError):
    """Raised when FirstLight input cannot satisfy the actor-fair contract."""


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    candidate = getattr(value, "value", value)
    if isinstance(candidate, (str, int, float, bool, type(None))):
        return candidate
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    return value


def _evidence_name(provenance: Any, field_name: str) -> str | None:
    if provenance is None:
        return None
    evidence = getattr(provenance, "field_evidence", {}).get(field_name)
    return str(_enum_value(evidence)) if evidence is not None else None


def _allowed_value(value: Any, name: str) -> tuple[bool, Any]:
    evidence = _evidence_name(getattr(value, "provenance", None), name)
    if evidence in {"unknown", "not_applicable"}:
        return False, None
    result = getattr(value, name, None)
    return result is not None, _plain(result)


def _select_runtime(value: Any, fields: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in fields:
        present, item = _allowed_value(value, name)
        if present:
            result[name] = item
    return result


def _block_status(parent: Any, field_name: str, value: Any, builder) -> ObservedValue:
    evidence = _evidence_name(getattr(parent, "runtime_provenance", None), field_name)
    if evidence == "unknown":
        return ObservedValue.unknown()
    if evidence == "not_applicable":
        return ObservedValue.not_applicable()
    if value is None or value == ():
        return ObservedValue.not_applicable() if evidence is None else ObservedValue.unknown()
    return ObservedValue.known(builder(value))


_SHIELD_FIELDS = ("hitpoints", "max_hitpoints", "kind", "source_entity", "broken")
_EFFECT_FIELDS = (
    "effect_id",
    "kind",
    "source_entity",
    "source_card_id",
    "source_owner",
    "remaining_ms",
    "stacks",
    "magnitude",
    "stage",
    "active",
)
_ATTACK_FIELDS = (
    "phase",
    "target_entity",
    "target_position",
    "phase_remaining_ms",
    "cooldown_remaining_ms",
    "sequence_index",
    "sequence_progress",
    "sequence_progress_limit",
    "sequence_decay_remaining_ms",
    "sequence_decay_duration_ms",
    "charge_stage",
    "charge_elapsed_ms",
    "damage_multiplier",
    "locked",
    "interrupted",
)
_PROJECTILE_FIELDS = (
    "projectile_id",
    "phase",
    "source_entity",
    "source_card_id",
    "target_entity",
    "target_position",
    "velocity",
    "homing",
    "deflected",
    "drag_stage",
)
_VISIBILITY_FIELDS = (
    "phase",
    "public_by_owner",
    "targetable_by_owner",
    "area_damage_eligible",
    "transition_remaining_ms",
)
_ABILITY_FIELDS = (
    "ability_id",
    "source_entity",
    "phase",
    "elixir_cost",
    "cooldown_ms",
    "remaining_cooldown_ms",
    "charges",
    "available",
    "target_entity",
    "target_position",
)
_EVOLUTION_FIELDS = (
    "card_id",
    "deck_slot",
    "phase",
    "base_form_id",
    "current_form_id",
    "next_form_id",
    "cycle_required",
    "cycle_remaining",
    "ready",
    "active",
    "deployments_in_cycle",
)
_RESOURCE_FIELDS = ("kind", "current_raw", "capacity_raw", "normalized")
_CAPTURE_FIELDS = (
    "phase",
    "configured_cooldown_ms",
    "cooldown_remaining_ms",
    "hit_frequency_ms",
    "hit_accumulator_ms",
)
_RELOCATION_FIELDS = (
    "phase",
    "stage",
    "relocation_index",
    "thresholds_percent",
    "hide_duration_ms",
    "remaining_ms",
    "burrowed",
)
_PERIODIC_FIELDS = (
    "phase",
    "period_attacks",
    "completed_attacks",
    "source_entity",
    "linger_remaining_ms",
    "linger_duration_ms",
)
_TARGET_BEARING_ATTACK_PHASES = frozenset(
    {"acquiring", "windup", "release", "channel", "charging"}
)


def _shield(value: Any, fallback: float | None) -> dict[str, Any] | None:
    if value is not None:
        return _select_runtime(value, _SHIELD_FIELDS)
    if fallback is not None and fallback > 0:
        return {"hitpoints": float(fallback)}
    return None


def _visibility(value: Any, actor: int) -> tuple[str | None, dict[str, Any] | None]:
    if value is None:
        return None, None
    payload = _select_runtime(value, _VISIBILITY_FIELDS)
    phase = payload.get("phase")
    if phase == "visible":
        basic = "visible"
    elif phase in {"hidden", "burrowed"}:
        basic = "hidden"
    elif phase in {"hiding", "revealing"}:
        basic = "transitioning"
    elif phase == "unknown":
        basic = "unknown"
    else:
        public = payload.get("public_by_owner", {})
        is_public = public.get(str(actor)) if isinstance(public, Mapping) else None
        basic = "visible" if is_public is True else "hidden" if is_public is False else None
    return basic, payload


def _attack(value: Any) -> tuple[int | None, dict[str, Any] | None]:
    if value is None:
        return None, None
    payload = _select_runtime(value, _ATTACK_FIELDS)
    payload.pop("target_entity", None)
    payload.pop("target_position", None)
    target: int | None = None
    if payload.get("phase") in _TARGET_BEARING_ATTACK_PHASES:
        target_known, target_value = _allowed_value(value, "target_entity")
        if target_known:
            target = int(target_value)
            payload["target_entity"] = target
        position_known, target_position = _allowed_value(value, "target_position")
        if position_known:
            payload["target_position"] = target_position
    return target, payload


def _current_form(
    *,
    kind: str,
    card_id: int | None,
    evolution: Mapping[str, Any] | None,
    hero_form_id: str | None,
) -> ObservedValue:
    if hero_form_id is not None:
        return ObservedValue.known({"variant": "hero", "form_id": hero_form_id})
    if kind not in {"troop", "building"}:
        return ObservedValue.not_applicable()
    if evolution is not None:
        current = evolution.get("current_form_id")
        if current is None:
            return ObservedValue.unknown()
        phase = str(evolution.get("phase") or "unknown")
        variant = "evolved" if phase == "evolved" or evolution.get("active") is True else phase
        return ObservedValue.known({"variant": variant, "form_id": str(current)})
    if card_id is not None:
        return ObservedValue.known({"variant": "base", "form_id": "base"})
    return ObservedValue.unknown()


def _tower_runtime(value: Any) -> dict[str, Any]:
    names = (
        "charge_count",
        "max_charge_count",
        "recharge_elapsed_ms",
        "recharge_duration_ms",
        "start_delay_remaining_ms",
        "start_delay_duration_ms",
        "cooking_contribution",
        "contribution_needed",
        "surviving_side_towers",
        "throw_delay_remaining_ms",
        "target_entity",
    )
    return {
        name: _plain(getattr(value, name)) for name in names if hasattr(value, name)
    }


def _capture(value: Any) -> dict[str, Any]:
    payload = _select_runtime(value, _CAPTURE_FIELDS)
    payload["targets"] = tuple(
        {
            name: _plain(getattr(item, name))
            for name in (
                "phase",
                "elapsed_ms",
                "phase_budget_remaining_ms",
                "target_entity",
            )
            if getattr(item, name, None) is not None
        }
        for item in getattr(value, "targets", ())
    )
    return payload


class FirstLightObservationAdapter:
    """Pure adapter from one FirstLight FAIR object to an Observation."""

    def __init__(
        self, *, runtime_hero_forms: Mapping[int, tuple[int, str]] | None = None
    ) -> None:
        normalized: dict[int, tuple[int, str]] = {}
        for raw_runtime_id, raw_identity in (runtime_hero_forms or {}).items():
            runtime_id = int(raw_runtime_id)
            base_card_id, form_id = raw_identity
            base_card_id = int(base_card_id)
            form_id = str(form_id)
            if runtime_id <= 0 or base_card_id <= 0 or not form_id:
                raise ValueError("runtime Hero form bindings require positive IDs and a form name")
            normalized[runtime_id] = (base_card_id, form_id)
        self._runtime_hero_forms = normalized

    def adapt(self, source: object, actor: int) -> Observation:
        tier = _enum_value(getattr(source, "tier", None))
        source_actor = getattr(source, "owner", None)
        if tier != "fair":
            raise FirstLightAdapterError(
                f"adapter requires a FirstLight FAIR observation, got {tier!r}"
            )
        if actor not in (0, 1) or source_actor != actor:
            raise FirstLightAdapterError(
                f"configured actor {actor!r} does not match FAIR observation actor {source_actor!r}"
            )

        players = tuple(
            self._player(item, actor)
            for item in sorted(source.players, key=lambda item: item.owner)
        )
        if {item["owner"] for item in players} != {0, 1}:
            raise FirstLightAdapterError("FAIR observation must contain both players")
        towers = tuple(
            self._tower(item, actor)
            for item in sorted(source.towers, key=lambda item: item.entity_id)
        )
        entities = tuple(
            self._entity(item, actor)
            for item in sorted(source.entities, key=lambda item: item.entity_id)
        )
        terminal = source.terminal
        return Observation(
            match={
                "phase": str(source.phase),
                "elapsed_ms": int(source.time.elapsed_ms),
                "remaining_ms": source.time.remaining_ms,
                "elixir_multiplier": float(source.time.elixir_multiplier),
            },
            players=players,
            towers=towers,
            entities=entities,
            terminal={
                "ended": bool(terminal.ended),
                "winner": terminal.winner,
                "result_by_owner": tuple(terminal.result_by_owner),
                "reason": terminal.reason,
                "terminal_tick": terminal.terminal_tick,
            },
        )

    def _player(self, player: Any, actor: int) -> dict[str, Any]:
        own = int(player.owner) == actor
        result: dict[str, Any] = {
            "owner": int(player.owner),
            "relation": "self" if own else "opponent",
            "crowns": int(player.crowns),
            "tower_ids": tuple(int(item) for item in player.tower_ids),
        }
        if own:
            if player.elixir_exact is None or not player.hand or player.next_card is None:
                raise FirstLightAdapterError("actor-private current state is incomplete")
            result["self_private"] = {
                "elixir": float(player.elixir_exact),
                "hand": tuple(int(item) for item in player.hand),
                "next_card": int(player.next_card),
            }
            result["current_evolution"] = tuple(
                _select_runtime(item, _EVOLUTION_FIELDS)
                for item in player.evolution_runtime_states
            )
            result["current_abilities"] = tuple(
                _select_runtime(item, _ABILITY_FIELDS)
                for item in player.ability_runtime_states
            )
        return result

    def _tower(self, tower: Any, actor: int) -> dict[str, Any]:
        shield = _shield(tower.shield_state, tower.shield)
        _basic_visibility, visibility = _visibility(tower.visibility_state, actor)
        _target, attack = _attack(tower.attack_state)
        result: dict[str, Any] = {
            "entity_id": int(tower.entity_id),
            "owner": int(tower.owner),
            "tower_kind": str(tower.tower_kind),
            "position": tuple(float(item) for item in tower.position),
            "hitpoints": float(tower.hitpoints),
            "max_hitpoints": float(tower.max_hitpoints),
            "tower_troop_id": tower.tower_troop_id,
            "active": bool(tower.active),
            "tower_troop_runtime": _block_status(
                tower,
                "tower_troop_runtime",
                tower.tower_troop_runtime,
                _tower_runtime,
            ),
        }
        if shield is not None:
            result["shield"] = shield
        if attack:
            result["attack"] = attack
        if tower.effect_states:
            result["effects"] = tuple(
                _select_runtime(item, _EFFECT_FIELDS) for item in tower.effect_states
            )
        if visibility is not None:
            result["visibility"] = visibility
        return result

    def _entity(self, entity: Any, actor: int) -> dict[str, Any]:
        raw_card_id = entity.card_id
        raw_kind = str(entity.entity_kind)
        hero_binding = (
            self._runtime_hero_forms.get(int(raw_card_id))
            if raw_card_id is not None
            else None
        )
        if hero_binding is not None and raw_kind in {"unknown", "hero"}:
            card_id, hero_form_id = hero_binding
            entity_kind = "hero"
        else:
            card_id, hero_form_id, entity_kind = raw_card_id, None, raw_kind

        shield = _shield(entity.shield_state, entity.shield)
        visibility_basic, visibility = _visibility(entity.visibility_state, actor)
        if visibility_basic is None:
            visibility_basic = "visible"
        evolution = (
            _select_runtime(entity.evolution_state, _EVOLUTION_FIELDS)
            if entity.evolution_state is not None
            else None
        )
        attack_target, attack = _attack(entity.attack_state)
        result: dict[str, Any] = {
            "entity_id": int(entity.entity_id),
            "owner": entity.owner,
            "card_id": card_id,
            "current_form": _current_form(
                kind=entity_kind,
                card_id=card_id,
                evolution=evolution,
                hero_form_id=hero_form_id,
            ),
            "entity_kind": entity_kind,
            "position": tuple(float(item) for item in entity.position),
            "hitpoints": entity.hitpoints,
            "max_hitpoints": entity.max_hitpoints,
            "resources": _block_status(
                entity,
                "resource_states",
                entity.resource_states,
                lambda values: tuple(
                    _select_runtime(item, _RESOURCE_FIELDS) for item in values
                ),
            ),
            "capture_runtime": _block_status(
                entity, "capture_runtime", entity.capture_runtime, _capture
            ),
            "threshold_relocation": _block_status(
                entity,
                "threshold_relocation_runtime",
                entity.threshold_relocation_runtime,
                lambda item: _select_runtime(item, _RELOCATION_FIELDS),
            ),
            "periodic_attack_modifier": _block_status(
                entity,
                "periodic_attack_modifier",
                entity.periodic_attack_modifier,
                lambda item: _select_runtime(item, _PERIODIC_FIELDS),
            ),
        }
        for name in ("velocity", "age_ms"):
            value = getattr(entity, name)
            if value is not None:
                result[name] = _plain(value)
        result["visibility_basic"] = visibility_basic
        if attack_target is not None:
            result["public_target_id"] = attack_target
        if entity.source_entity is not None:
            result["source_entity_id"] = int(entity.source_entity)
        if entity.causal_group is not None:
            result["causal_relation"] = {
                "kind": _enum_value(entity.causal_group.kind),
                "handle": entity.causal_group.handle,
                "source_card_id": entity.causal_group.source_card_id,
                "parent_entity_id": entity.causal_group.parent_entity_id,
            }
        if shield is not None:
            result["shield"] = shield
        if attack:
            result["attack"] = attack
        if entity.effect_states:
            result["effects"] = tuple(
                _select_runtime(item, _EFFECT_FIELDS) for item in entity.effect_states
            )
        if entity.projectile_state is not None:
            result["projectile"] = _select_runtime(
                entity.projectile_state, _PROJECTILE_FIELDS
            )
        if visibility is not None:
            result["visibility"] = visibility
        if entity.ability_states:
            result["abilities"] = tuple(
                _select_runtime(item, _ABILITY_FIELDS) for item in entity.ability_states
            )
        if evolution is not None:
            result["evolution"] = evolution
        return result
