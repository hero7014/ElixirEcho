"""Semantic Observation to frozen FirstLight V4 tensor-record adapter.

This module constructs V4 records directly from ElixirEcho Observation/Profile
contracts. It does not import or invoke any legacy tensorizer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

import torch

from elixir_echo.domain import DecisionIdentity
from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
from elixir_echo.temporal import PreviousActionStatus, TemporalContext


class TensorizationError(ValueError):
    """The selected semantic input cannot be represented by frozen V4 tensors."""


@dataclass(frozen=True, slots=True)
class TensorizedDecision:
    batch: Any
    candidate_index_by_slot: Mapping[int, int]
    candidate_uid_by_slot: Mapping[int, int]
    actor: int


def load_v4_components(runtime: FirstLightRuntime) -> Any:
    """Load provider catalogs, config, and record types through the read-only gate."""

    factory = runtime.import_module("native_runner.training.v4.factory")
    config_module = runtime.import_module("native_runner.training.v4.config")
    tensors = runtime.import_module("native_runner.training.v4.tensors")
    decoding = runtime.import_module("native_runner.training.v4.decoding")
    catalog = runtime.import_module("native_runner.training.v4.catalog")
    # The formal Checkpoint V1 contract was trained with 64 group rows.
    config = config_module.ModelConfigV4(max_battle_groups=64)
    names = (
        "CardSetV4", "TowerSetV4", "BattleGroupSetV4", "ActiveEffectSetV4",
        "EventSetV4", "RelationEdgesV4", "PreviousActionV4", "ActionCandidatesV4",
        "UniversalSemanticBatchV4", "ActionSequenceV4", "GATE_WAIT", "GATE_ACT", "CANDIDATE_DEPLOY",
        "TARGET_GRID", "EFFECT_PARENT_CHILD", "EFFECT_PARENT_TOWER",
    )
    record_types = SimpleNamespace(**{name: getattr(tensors, name) for name in names})
    record_types.candidate_uid_v4 = decoding.candidate_uid_v4
    record_types.target_cell_from_xy = tensors.target_cell_from_xy
    record_types.UNKNOWN_CARD_VOCAB_ID = catalog.UNKNOWN_CARD_VOCAB_ID
    record_types.UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID = catalog.UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
    return SimpleNamespace(
        provider_root=runtime.root,
        bundle=factory.production_semantic_bundle(),
        config=config,
        types=record_types,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _known(value: Any) -> Any:
    data = _mapping(value)
    return data.get("value") if data.get("status") == "known" else None


def _known_or_raw(value: Any) -> Any:
    """Unwrap known semantic values while preserving legacy raw scalar fields."""

    if isinstance(value, Mapping) and "status" in value:
        return value.get("value") if value.get("status") == "known" else None
    return value


def _number(value: Any, scale: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value / scale if scale else value


def _float(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32)


def _long(*shape: int, fill: int = 0) -> torch.Tensor:
    return torch.full(shape, fill, dtype=torch.long)


def _bool(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.bool)


class ObservationTensorizer:
    """Build one batch-size-one FirstLight V4 input from a selected profile."""

    WORLD_WIDTH = 18_000.0
    WORLD_HEIGHT = 32_000.0
    OWNER_SELF, OWNER_ENEMY, OWNER_NEUTRAL = 0, 1, 2
    TOWER_TYPES = {"king": 0, "princess_left": 1, "princess_right": 2}
    TOWER_TROOPS = {159_000_000: 1, 159_000_001: 2, 159_000_002: 3, 159_000_004: 4}
    CHILD_TYPES = {"troop": 0, "building": 1, "hero": 2, "projectile": 3, "area": 4, "effect": 5, "character": 6, "unknown": 7}
    GROUP_TYPES = {"deployment": 0, "spawn_wave": 1, "volley": 2, "persistent_effect": 3, "deploy": 0}
    FORM_NORMAL, FORM_EVOLUTION, FORM_HERO, FORM_HERO_ACTIVE = 0, 1, 2, 3
    PREVIOUS_ACTION_STATUS_SCALAR = 17

    def __init__(self, components: Any) -> None:
        self.components = load_v4_components(components) if isinstance(components, FirstLightRuntime) else components
        self.config = self.components.config
        self.types = self.components.types
        self.bundle = self.components.bundle

    def _owner(self, owner: Any, actor: int) -> int:
        if owner not in (0, 1):
            return self.OWNER_NEUTRAL
        return self.OWNER_SELF if int(owner) == actor else self.OWNER_ENEMY

    def _tile(self, point: Sequence[Any], actor: int) -> tuple[float, float]:
        if len(point) != 2:
            raise TensorizationError("position must contain x and y")
        x, y = float(point[0]), float(point[1])
        if actor == 1:
            x, y = self.WORLD_WIDTH - x, self.WORLD_HEIGHT - y
        return (
            min(max(x / (self.WORLD_WIDTH / self.config.board_width), 0.0), self.config.board_width - 1e-4),
            min(max(y / (self.WORLD_HEIGHT / self.config.board_height), 0.0), self.config.board_height - 1e-4),
        )

    @staticmethod
    def _self_player(profile: Mapping[str, Any]) -> Mapping[str, Any]:
        for player in profile["players"]:
            if player.get("relation") == "self":
                return player
        raise TensorizationError("selected Observation has no self player")

    @staticmethod
    def _opponent_player(profile: Mapping[str, Any]) -> Mapping[str, Any]:
        for player in profile["players"]:
            if player.get("relation") == "opponent":
                return player
        raise TensorizationError("selected Observation has no opponent player")

    def _cards(self, profile: Mapping[str, Any], actor: int, temporal: TemporalContext | None) -> tuple[Any, Any, dict[int, int]]:
        player = self._self_player(profile)
        private = _mapping(player.get("self_private"))
        hand = tuple(int(value) for value in private.get("hand", ()))
        if not 1 <= len(hand) <= 4:
            raise TensorizationError("current own hand must contain one to four cards")
        next_card = private.get("next_card")
        ordered = list(hand)
        if isinstance(next_card, int) and not isinstance(next_card, bool) and next_card not in ordered:
            ordered.append(int(next_card))
        if len(ordered) > self.config.max_own_cards:
            raise TensorizationError("current own hand and next card exceed V4 capacity")

        count = self.config.max_own_cards
        card_ids, forms = _long(1, count), _long(1, count)
        roles, runtime = _bool(1, count, 2), _float(1, count, self.config.card_runtime_feature_dim)
        mask = _bool(1, count)
        card_to_row: dict[int, int] = {}
        evolutions = {int(item["card_id"]): item for item in player.get("current_evolution", ()) if isinstance(item, Mapping) and isinstance(item.get("card_id"), int)}
        abilities = {int(item["card_id"]): item for item in player.get("current_abilities", ()) if isinstance(item, Mapping) and isinstance(item.get("card_id"), int)}
        for row, card_id in enumerate(ordered):
            card_ids[0, row] = self.bundle.card_catalog.require_vocab_id(card_id, context="current own card")
            mask[0, row] = True
            card_to_row[card_id] = row
            runtime[0, row, 0] = float(card_id in hand)
            runtime[0, row, 1] = float(card_id == next_card)
            runtime[0, row, 2] = 1.0
            evolution = evolutions.get(card_id)
            if evolution is not None:
                runtime[0, row, 3] = 1.0
                runtime[0, row, 4] = float(evolution.get("ready") is True)
                runtime[0, row, 5] = _number(evolution.get("cycle_remaining"), 4.0)
                if evolution.get("active") is True:
                    forms[0, row] = self.FORM_EVOLUTION
            ability = abilities.get(card_id)
            if ability is not None:
                runtime[0, row, 9] = 1.0
                runtime[0, row, 10] = float(ability.get("available") is True)
                runtime[0, row, 11] = _number(ability.get("remaining_cooldown_ms"), 30_000.0)
                runtime[0, row, 13] = float(ability.get("phase") == "casting")
                runtime[0, row, 14] = float(ability.get("phase") == "active")
                forms[0, row] = self.FORM_HERO_ACTIVE if ability.get("phase") == "active" else self.FORM_HERO
        own = self.types.CardSetV4(card_ids, forms, roles, runtime, mask)

        revealed = () if temporal is None else temporal.public_state.revealed_cards
        if len(revealed) > self.config.max_opponent_cards - 1 or len(set(revealed)) != len(revealed):
            raise TensorizationError("publicly revealed opponent cards exceed V4 capacity")
        opponent_count = self.config.max_opponent_cards
        opp_ids, opp_forms = _long(1, opponent_count), _long(1, opponent_count)
        opp_roles, opp_runtime = _bool(1, opponent_count, 2), _float(1, opponent_count, self.config.card_runtime_feature_dim)
        opp_mask = _bool(1, opponent_count)
        for row, card_id in enumerate(revealed):
            opp_ids[0, row] = self.bundle.card_catalog.require_vocab_id(int(card_id), context="publicly revealed opponent card")
            opp_runtime[0, row, 2] = 1.0
            opp_mask[0, row] = True
        if temporal is not None:
            for item in temporal.public_state.public_plays_since_last_use:
                row = next((i for i, value in enumerate(revealed) if value == item.card_id), None)
                if row is not None:
                    opp_runtime[0, row, 7] = 0.0 if item.capped else max(0, 4 - item.count) / 4.0
                    opp_runtime[0, row, 8] = 1.0
        unknown = self.config.max_opponent_cards - 1 - len(revealed)
        if unknown:
            opp_ids[0, -1] = self.types.UNKNOWN_CARD_VOCAB_ID
            opp_runtime[0, -1, 15] = unknown / float(self.config.max_opponent_cards - 1)
            opp_mask[0, -1] = True
        opponent = self.types.CardSetV4(opp_ids, opp_forms, opp_roles, opp_runtime, opp_mask)
        return own, opponent, card_to_row

    def _towers(self, profile: Mapping[str, Any], actor: int) -> tuple[Any, dict[int, int]]:
        rows = sorted(profile["towers"], key=lambda item: (self._owner(item.get("owner"), actor), self.TOWER_TYPES.get(self._tower_kind(str(item.get("tower_kind")), actor), 99), int(item.get("entity_id", -1))))
        if len(rows) > self.config.max_towers:
            raise TensorizationError("tower count exceeds V4 capacity")
        n = self.config.max_towers
        kind, troop = _long(1, n), _long(1, n)
        owner = _long(1, n, fill=self.OWNER_NEUTRAL)
        features, position, extent, mask = _float(1, n, self.config.tower_feature_dim), _float(1, n, 2), _float(1, n, 4), _bool(1, n)
        indices: dict[int, int] = {}
        for row, tower in enumerate(rows):
            tower_kind = self._tower_kind(str(tower.get("tower_kind")), actor)
            if tower_kind not in self.TOWER_TYPES:
                raise TensorizationError(f"unsupported tower kind {tower_kind!r}")
            xy = self._tile(tower["position"], actor)
            hp, max_hp = tower.get("hitpoints"), tower.get("max_hitpoints")
            shield = _mapping(tower.get("shield"))
            kind[0, row] = self.TOWER_TYPES[tower_kind]
            troop_id = tower.get("tower_troop_id")
            troop[0, row] = 0 if troop_id is None else self.TOWER_TROOPS.get(int(troop_id), 0)
            owner[0, row] = self._owner(tower.get("owner"), actor)
            features[0, row, 0] = _number(hp, float(max_hp or 1.0))
            shield_hp = _known_or_raw(shield.get("hitpoints", shield))
            features[0, row, 1] = _number(shield_hp, float(max_hp or 1.0))
            features[0, row, 2] = float(tower.get("active") is True)
            attack = _mapping(tower.get("attack"))
            features[0, row, 3] = float(attack.get("target_entity") is not None)
            features[0, row, 5] = _number(hp, 5_000.0)
            features[0, row, 6] = _number(max_hp, 5_000.0)
            phase = str(attack.get("phase", ""))
            features[0, row, 7] = _number(attack.get("cooldown_remaining_ms"), 5_000)
            features[0, row, 8] = _number(attack.get("phase_remaining_ms"), 5_000)
            features[0, row, 9] = float(phase == "windup")
            features[0, row, 10] = float(phase in {"release", "channel"})
            features[0, row, 11] = float(phase == "cooldown")
            features[0, row, 12] = float(phase == "interrupted" or attack.get("interrupted") is True)
            features[0, row, 13] = _number(attack.get("sequence_index"), 10)
            features[0, row, 14] = _number(attack.get("damage_multiplier"), 5)
            position[0, row] = torch.tensor(xy)
            half = 2.0 if tower_kind == "king" else 1.5
            extent[0, row] = torch.tensor((xy[0] - half, xy[1] - half, xy[0] + half, xy[1] + half))
            mask[0, row] = True
            indices[int(tower["entity_id"])] = row
        return self.types.TowerSetV4(kind, troop, owner, features, position, extent, mask), indices

    @staticmethod
    def _tower_kind(kind: str, actor: int) -> str:
        if actor == 1 and kind == "princess_left":
            return "princess_right"
        if actor == 1 and kind == "princess_right":
            return "princess_left"
        return kind

    def _entities(self, profile: Mapping[str, Any], actor: int) -> tuple[Any, dict[int, int]]:
        entities = sorted(profile["entities"], key=lambda item: int(item.get("entity_id", -1)))
        if len(entities) > min(self.config.max_battle_groups, self.config.max_children_total):
            raise TensorizationError("entity count exceeds one-entity-per-group V4 capacity")
        groups_n, children_n = self.config.max_battle_groups, self.config.max_children_total
        source_card, group_type = _long(1, groups_n), _long(1, groups_n)
        owner, forms = _long(1, groups_n, fill=self.OWNER_NEUTRAL), _long(1, groups_n)
        group_features, group_position, group_extent, group_mask = _float(1, groups_n, self.config.group_feature_dim), _float(1, groups_n, 2), _float(1, groups_n, 4), _bool(1, groups_n)
        child_archetype, child_group, child_type = _long(1, children_n), _long(1, children_n), _long(1, children_n)
        child_features, child_position, child_extent, child_radius = _float(1, children_n, self.config.child_feature_dim), _float(1, children_n, 2), _float(1, children_n, 4), _float(1, children_n)
        child_mask = _bool(1, children_n)
        group_by_id: dict[int, int] = {}
        for row, entity in enumerate(entities):
            entity_id = int(entity["entity_id"])
            relation = _mapping(entity.get("causal_relation"))
            card_id = relation.get("source_card_id", entity.get("card_id"))
            source_card[0, row] = self.bundle.card_catalog.vocab_id(card_id)
            group_type[0, row] = self.GROUP_TYPES.get(str(relation.get("kind", "deployment")), 0)
            owner[0, row] = self._owner(entity.get("owner"), actor)
            kind = str(entity.get("entity_kind", "unknown"))
            form_value = _mapping(_known(entity.get("current_form")))
            variant = form_value.get("variant", "base")
            if variant in {"evolution", "evolved"}:
                forms[0, row] = self.FORM_EVOLUTION
            elif kind == "hero":
                active = any(
                    _mapping(ability).get("phase") == "active"
                    for ability in entity.get("abilities", ())
                )
                forms[0, row] = self.FORM_HERO_ACTIVE if active else self.FORM_HERO
            xy = self._tile(entity["position"], actor)
            hp, max_hp = entity.get("hitpoints"), entity.get("max_hitpoints")
            shield = _mapping(entity.get("shield"))
            shield_hp = _known_or_raw(shield.get("hitpoints", shield))
            group_features[0, row, 0] = 1 / 16
            group_features[0, row, 2] = _number(hp, 10_000)
            group_features[0, row, 3] = _number(shield_hp, 10_000)
            ratio = _number(hp, float(max_hp or 1.0))
            group_features[0, row, 4:7] = ratio
            group_features[0, row, 9] = float(entity.get("public_target_id") is not None)
            velocity = entity.get("velocity")
            if isinstance(velocity, Sequence) and len(velocity) == 2:
                vx, vy = float(velocity[0]), float(velocity[1])
                if actor == 1:
                    vx, vy = -vx, -vy
                group_features[0, row, 10] = _number(vx / 1_000, 10)
                group_features[0, row, 11] = _number(vy / 1_000, 10)
                group_features[0, row, 12] = _number(math.hypot(vx, vy) / 1_000, 10)
            if kind == "projectile":
                group_features[0, row, 17] = 1 / 16
            group_features[0, row, 20] = float(bool(relation))
            group_position[0, row] = torch.tensor(xy)
            group_extent[0, row] = torch.tensor((xy[0], xy[1], xy[0], xy[1]))
            group_mask[0, row] = True

            child_archetype[0, row] = self.types.UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID
            child_group[0, row] = row
            child_type[0, row] = self.CHILD_TYPES.get(kind, self.CHILD_TYPES["unknown"])
            child_features[0, row, 0] = ratio
            child_features[0, row, 1] = _number(hp, 10_000)
            child_features[0, row, 2] = _number(max_hp, 10_000)
            child_features[0, row, 3] = _number(shield_hp, 10_000)
            child_features[0, row, 4] = _number(entity.get("age_ms"), 60_000)
            child_features[0, row, 5] = float(entity.get("public_target_id") is not None)
            child_features[0, row, 7] = xy[0] / self.config.board_width
            child_features[0, row, 8] = xy[1] / self.config.board_height
            if isinstance(velocity, Sequence) and len(velocity) == 2:
                child_features[0, row, 9] = _number(vx / 1_000, 10)
                child_features[0, row, 10] = _number(vy / 1_000, 10)
            attack = _mapping(entity.get("attack"))
            phase = str(attack.get("phase", ""))
            if phase == "windup": child_features[0, row, 16] = 1.0
            if phase in {"release", "channel"}: child_features[0, row, 17] = 1.0
            if phase == "cooldown": child_features[0, row, 18] = 1.0
            if phase == "charging": child_features[0, row, 19] = 1.0
            if attack:
                child_features[0, row, 11] = _number(attack.get("cooldown_remaining_ms"), 5_000)
                child_features[0, row, 12] = _number(attack.get("phase_remaining_ms"), 5_000)
                child_features[0, row, 20] = float(phase == "interrupted" or attack.get("interrupted") is True)
                child_features[0, row, 21] = _number(attack.get("sequence_index"), 10)
                child_features[0, row, 22] = _number(attack.get("charge_stage"), 10)
                child_features[0, row, 23] = _number(attack.get("charge_elapsed_ms"), 5_000)
                child_features[0, row, 24] = _number(attack.get("damage_multiplier"), 5)
                child_features[0, row, 25] = float(attack.get("locked") is True)
            visibility = _mapping(entity.get("visibility"))
            hidden = entity.get("visibility_basic") in {"hidden", "burrowed"}
            child_features[0, row, 30] = float(hidden)
            if visibility:
                child_features[0, row, 31] = _number(visibility.get("transition_remaining_ms"), 5_000)
                targetable = _mapping(visibility.get("targetable_by_owner"))
                if str(actor) in targetable:
                    child_features[0, row, 36] = float(targetable[str(actor)] is True)
                    child_features[0, row, 37] = 1.0
                area_eligible = visibility.get("area_damage_eligible")
                if isinstance(area_eligible, bool):
                    child_features[0, row, 38] = float(area_eligible)
                    child_features[0, row, 39] = 1.0
            projectile = _mapping(entity.get("projectile"))
            if projectile:
                child_features[0, row, 13] = _number(projectile.get("damage"), 5_000)
                child_features[0, row, 14] = _number(projectile.get("radius_tiles"), 5)
                child_features[0, row, 15] = float(projectile.get("homing") is True)
                child_features[0, row, 32] = float(projectile.get("phase") == "in_flight")
                child_features[0, row, 33] = float(projectile.get("damage") is not None)
                child_features[0, row, 34] = float(projectile.get("radius_tiles") is not None)
                target_position = projectile.get("target_position")
                if isinstance(target_position, Sequence) and len(target_position) == 2:
                    target_xy = self._tile(target_position, actor)
                    child_features[0, row, 40] = target_xy[0] / self.config.board_width
                    child_features[0, row, 41] = target_xy[1] / self.config.board_height
                    child_features[0, row, 42] = 1.0
                child_features[0, row, 43] = float(projectile.get("drag_back_active") is True)
                child_features[0, row, 44] = float(projectile.get("drag_stage") is not None)
            child_position[0, row] = torch.tensor(xy)
            child_extent[0, row] = torch.tensor((xy[0], xy[1], xy[0], xy[1]))
            child_mask[0, row] = True
            group_by_id[entity_id] = row
        return self.types.BattleGroupSetV4(
            source_card, group_type, owner, forms, group_features, group_position, group_extent, group_mask,
            child_archetype, child_group, child_type, child_features, child_position, child_extent, child_radius, child_mask,
        ), group_by_id

    def _effects(
        self,
        profile: Mapping[str, Any],
        actor: int,
        tower_rows: Mapping[int, int],
        group_rows: Mapping[int, int],
    ) -> Any:
        rows: list[tuple[int, int, int, int, tuple[float, ...]]] = []
        for parent_type, items, indices in (
            (self.types.EFFECT_PARENT_TOWER, profile.get("towers", ()), tower_rows),
            (self.types.EFFECT_PARENT_CHILD, profile.get("entities", ()), group_rows),
        ):
            for item in items:
                parent_index = indices.get(int(item.get("entity_id", -1)))
                if parent_index is None:
                    continue
                for effect in item.get("effects", ()):
                    if not isinstance(effect, Mapping) or effect.get("active") is False:
                        continue
                    raw_id = effect.get("effect_id", "")
                    if isinstance(raw_id, str) and raw_id.startswith("buff:"):
                        suffix = raw_id.split(":", 1)[1]
                        raw_id = int(suffix) if suffix.isdigit() else -1
                    if type(raw_id) is not int:
                        raw_id = -1
                    vocab = int(self.bundle.effect_catalog.vocab_id(raw_id))
                    if vocab == 0:
                        continue
                    remaining, stacks, magnitude = effect.get("remaining_ms"), effect.get("stacks"), effect.get("magnitude")
                    runtime = (
                        min(_number(remaining, 10_000), 30.0),
                        float(isinstance(remaining, (int, float)) and not isinstance(remaining, bool)),
                        float(remaining is None),
                        _number(stacks, 10),
                        float(isinstance(stacks, int) and not isinstance(stacks, bool)),
                        _number(magnitude),
                        float(isinstance(magnitude, (int, float)) and not isinstance(magnitude, bool)),
                    )
                    rows.append((vocab, parent_type, parent_index, self._owner(effect.get("source_owner"), actor), runtime))
        if len(rows) > 64:
            raise TensorizationError("active-effect count exceeds V4 adapter capacity")
        count = 64
        effect_id, parent_type, parent_index, source_owner = _long(1, count), _long(1, count), _long(1, count, fill=-1), _long(1, count)
        features, mask = _float(1, count, 7), _bool(1, count)
        for row, values in enumerate(sorted(rows)):
            effect_id[0, row], parent_type[0, row], parent_index[0, row], source_owner[0, row] = values[:4]
            features[0, row] = torch.tensor(values[4])
            mask[0, row] = True
        return self.types.ActiveEffectSetV4(effect_id, parent_type, parent_index, source_owner, features, mask)

    def _relations(
        self,
        profile: Mapping[str, Any],
        actor: int,
        own_cards: Any,
        towers: Any,
        group_rows: Mapping[int, int],
        tower_rows: Mapping[int, int],
    ) -> Any:
        group_start = self.config.max_towers + self.config.max_own_cards + self.config.max_opponent_cards
        tokens = dict(tower_rows)
        tokens.update({entity_id: group_start + row for entity_id, row in group_rows.items()})
        edges: set[tuple[int, int, int]] = set()
        for entity in profile.get("entities", ()):
            source = tokens.get(int(entity.get("entity_id", -1)))
            if source is None:
                continue
            for raw_id, forward, reverse in (
                (entity.get("public_target_id"), 1, 2),
                (entity.get("source_entity_id"), 4, 3),
            ):
                target = tokens.get(int(raw_id)) if raw_id is not None else None
                if target is not None and source != target:
                    edges.add((source, target, forward))
                    edges.add((target, source, reverse))
            parent_id = _mapping(entity.get("causal_relation")).get("parent_entity_id")
            parent = tokens.get(int(parent_id)) if parent_id is not None else None
            if parent is not None and parent != source:
                edges.add((source, parent, 8))

        own_start = self.config.max_towers
        own_card_tokens = {
            int(own_cards.card_vocab_id[0, row]): own_start + row
            for row in range(own_cards.mask.shape[1])
            if bool(own_cards.mask[0, row])
        }
        for entity in profile.get("entities", ()):
            if self._owner(entity.get("owner"), actor) != self.OWNER_SELF:
                continue
            row = group_rows.get(int(entity.get("entity_id", -1)))
            if row is None:
                continue
            vocab = int(self.bundle.card_catalog.vocab_id(_mapping(entity.get("causal_relation")).get("source_card_id", entity.get("card_id"))))
            card_token = own_card_tokens.get(vocab)
            if card_token is not None:
                edges.add((card_token, group_start + row, 7))
        if len(edges) > 256:
            raise TensorizationError("semantic relation count exceeds V4 capacity")
        source, target, relation_type, mask = _long(1, 256), _long(1, 256), _long(1, 256), _bool(1, 256)
        for row, (left, right, kind) in enumerate(sorted(edges)):
            source[0, row], target[0, row], relation_type[0, row], mask[0, row] = left, right, kind, True
        return self.types.RelationEdgesV4(source, target, relation_type, mask)

    def _candidates(self, profile: Mapping[str, Any], actor: int, action_mask: Mapping[str, Any], card_rows: Mapping[int, int]) -> Any:
        player = self._self_player(profile)
        hand = tuple(int(value) for value in _mapping(player.get("self_private")).get("hand", ()))
        legal_slots = tuple(bool(value) for value in action_mask.get("hand_slots", ()))
        if len(legal_slots) != 4:
            raise TensorizationError("ActionMask hand_slots must contain four entries")
        n = self.config.max_action_candidates
        mask, uid, variant = _bool(1, n), _long(1, n), _long(1, n)
        visible, effective, form = _long(1, n), _long(1, n), _long(1, n)
        own_row, source_group, source_child = _long(1, n, fill=-1), _long(1, n, fill=-1), _long(1, n, fill=-1)
        ability, cost = _long(1, n), _float(1, n)
        runtime, target_mode = _float(1, n, self.config.candidate_runtime_feature_dim), _long(1, n)
        placement = _bool(1, n, self.config.board_height, self.config.board_width)
        is_building, half_width, half_height, offset_x, offset_y = _bool(1, n), _float(1, n), _float(1, n), _float(1, n), _float(1, n)
        exclusion, native_slot, native_source, native_visible = _long(1, n, fill=-1), _long(1, n, fill=-1), _long(1, n, fill=-1), _long(1, n, fill=-1)
        index_by_slot: dict[int, int] = {}
        uid_by_slot: dict[int, int] = {}
        placements = _mapping(action_mask.get("placement_masks"))
        candidate_row = 0
        for slot, allowed in enumerate(legal_slots):
            if not allowed:
                continue
            entry = _mapping(placements.get(str(slot)))
            card_id = entry.get("visible_card_id", entry.get("card_id"))
            if type(card_id) is not int or card_id not in hand or card_id not in card_rows:
                raise TensorizationError("legal hand-slot candidate does not match current hand")
            effective_id = entry.get("effective_card_id", card_id)
            if type(effective_id) is not int:
                raise TensorizationError("candidate effective card identity is not an integer")
            card_vocab = self.bundle.card_catalog.require_vocab_id(card_id, context="legal candidate")
            effective_vocab = self.bundle.card_catalog.require_vocab_id(effective_id, context="effective candidate")
            candidate_uid = int(self.types.candidate_uid_v4(
                variant=self.types.CANDIDATE_DEPLOY,
                visible_card_vocab_id=card_vocab,
                effective_card_vocab_id=effective_vocab,
                native_hand_slot=slot,
            ))
            grid = entry.get("row_major")
            if not isinstance(grid, Sequence) or len(grid) != self.config.board_height or any(not isinstance(row, Sequence) or len(row) != self.config.board_width for row in grid):
                raise TensorizationError("candidate placement mask must be row-major [32][18]")
            values = [list(map(bool, row)) for row in grid]
            if actor == 1:
                values = [list(reversed(row)) for row in reversed(values)]
            placement[0, candidate_row] = torch.tensor(values, dtype=torch.bool)
            footprint_width = _number(entry.get("footprint_width_tiles"))
            footprint_height = _number(entry.get("footprint_height_tiles"))
            building = footprint_width > 0.0 and footprint_height > 0.0
            is_building[0, candidate_row] = building
            half_width[0, candidate_row] = footprint_width / 2.0 if building else 0.0
            half_height[0, candidate_row] = footprint_height / 2.0 if building else 0.0
            offset = entry.get("model_subcell_offset")
            if isinstance(offset, Sequence) and len(offset) == 2:
                sign = -1.0 if actor == 1 else 1.0
                offset_x[0, candidate_row] = sign * _number(offset[0])
                offset_y[0, candidate_row] = sign * _number(offset[1])
            raw_cost = entry.get("effective_cost")
            if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)) or not math.isfinite(float(raw_cost)):
                raise TensorizationError("legal candidate requires a finite effective_cost")
            actual_cost = float(raw_cost)
            mask[0, candidate_row], uid[0, candidate_row] = True, candidate_uid
            visible[0, candidate_row], effective[0, candidate_row] = card_vocab, effective_vocab
            form[0, candidate_row] = int(entry.get("form_code", 0))
            own_row[0, candidate_row], cost[0, candidate_row] = card_rows[card_id], actual_cost
            runtime[0, candidate_row, 0] = actual_cost / 10.0
            runtime[0, candidate_row, 1] = placement[0, candidate_row].float().mean()
            target_mode[0, candidate_row] = self.types.TARGET_GRID
            exclusion[0, candidate_row] = 1000 + slot
            native_slot[0, candidate_row], native_visible[0, candidate_row] = slot, card_id
            index_by_slot[slot], uid_by_slot[slot] = candidate_row, candidate_uid
            candidate_row += 1
        if candidate_row == 0:
            # V4 requires one valid candidate-shaped row even when ACT is illegal.
            mask[0, 0] = False
        raw_elixir = _mapping(action_mask.get("reasons")).get("effective_elixir", _mapping(player.get("self_private")).get("elixir", 0.0))
        candidates = self.types.ActionCandidatesV4(
            mask, _float(1).fill_(min(max(_number(raw_elixir), 0.0), 10.0)), uid, variant, visible, effective, form,
            own_row, source_group, source_child, ability, cost, runtime, target_mode, placement, is_building,
            half_width, half_height, offset_x, offset_y, exclusion, native_slot, native_source, native_visible,
        )
        return candidates, index_by_slot, uid_by_slot

    def _previous_action(self, temporal: TemporalContext | None) -> tuple[Any, float]:
        count = self.config.max_micro_actions
        shape = (1, count)
        gate, n_actions = _long(1, fill=self.types.GATE_WAIT), _long(1)
        variant, visible, effective, form, ability = (_long(1, count) for _ in range(5))
        target, delay = _long(1, count, fill=-1), _long(1, count, fill=-1)
        source_position, source_mask, action_mask = _float(1, count, 2), _bool(1, count), _bool(1, count)
        status_scalar = 0.0
        if temporal is not None:
            previous = temporal.previous_action
            if previous.status is PreviousActionStatus.UNKNOWN_MASKED:
                status_scalar = 0.5
            elif previous.status is PreviousActionStatus.KNOWN:
                if previous.actions:
                    action = previous.actions[0] if len(previous.actions) == 1 else None
                    card_id = None if action is None else action.get("card_id")
                    grid = None if action is None else action.get("target_grid")
                    delay_ticks = None if action is None else action.get("execute_offset_ticks")
                    coordinates_are_exact = (
                        isinstance(grid, Sequence)
                        and len(grid) == 2
                        and all(type(value) is int for value in grid)
                        and 0 <= grid[0] < self.config.board_width
                        and 0 <= grid[1] < self.config.board_height
                    )
                    representable = (
                        action is not None
                        and action.get("kind") == "play_card"
                        and action.get("owner") == temporal.identity.actor
                        and type(card_id) is int
                        and action.get("target_kind") == "grid"
                        and coordinates_are_exact
                        and type(delay_ticks) is int
                        and delay_ticks * 50 in self.config.delay_offset_ms
                    )
                    if representable:
                        try:
                            visible_vocab = self.bundle.card_catalog.require_vocab_id(card_id, context="previous action")
                            effective_id = action.get("effective_card_id", card_id)
                            if type(effective_id) is not int:
                                raise ValueError("prior effective card ID is not an integer")
                            effective_vocab = self.bundle.card_catalog.require_vocab_id(effective_id, context="previous effective card")
                            form_code = action.get("form_code", 0)
                            if type(form_code) is not int:
                                raise ValueError("prior action form code is not an integer")
                        except ValueError:
                            status_scalar = 0.5
                        else:
                            x, y = grid
                            if temporal.identity.actor == 1:
                                x, y = self.config.board_width - 1 - x, self.config.board_height - 1 - y
                            gate[0], n_actions[0] = self.types.GATE_ACT, 1
                            variant[0, 0], visible[0, 0], effective[0, 0] = self.types.CANDIDATE_DEPLOY, visible_vocab, effective_vocab
                            form[0, 0] = form_code
                            target[0, 0] = y * self.config.board_width + x
                            delay[0, 0] = self.config.delay_offset_ms.index(delay_ticks * 50)
                            action_mask[0, 0] = True
                            status_scalar = 1.0
                    else:
                        status_scalar = 0.5
                else:
                    status_scalar = 1.0
        if self.PREVIOUS_ACTION_STATUS_SCALAR >= self.config.generic_scalar_dim:
            raise TensorizationError("V4 lacks a reserved previous-action status scalar")
        previous_batch = self.types.PreviousActionV4(gate, n_actions, variant, visible, effective, form, ability, target, delay, source_position, source_mask, action_mask)
        return previous_batch, status_scalar

    def tensorize(
        self,
        profile: Mapping[str, Any],
        *,
        actor: int,
        action_mask: Mapping[str, Any],
        temporal_context: TemporalContext | None = None,
        validate: bool = True,
    ) -> TensorizedDecision:
        if actor not in (0, 1):
            raise TensorizationError("actor must be 0 or 1")
        if temporal_context is not None and temporal_context.identity.actor != actor:
            raise TensorizationError("temporal context actor differs from profile actor")
        own, opponent, card_rows = self._cards(profile, actor, temporal_context)
        towers, tower_rows = self._towers(profile, actor)
        groups, group_rows = self._entities(profile, actor)
        candidates, slot_rows, slot_uids = self._candidates(profile, actor, action_mask, card_rows)
        previous, previous_status = self._previous_action(temporal_context)
        config = self.config
        scalars = _float(1, config.generic_scalar_dim)
        match = _mapping(profile.get("match"))
        player, opponent_player = self._self_player(profile), self._opponent_player(profile)
        private = _mapping(player.get("self_private"))
        scalars[0, 0] = _number(match.get("remaining_ms"), 300_000)
        scalars[0, 1] = _number(match.get("elixir_multiplier"), 3)
        scalars[0, 2] = float(match.get("phase") in {"overtime", "sudden_death"})
        scalars[0, 3] = float(match.get("phase") == "tiebreak")
        scalars[0, 4] = _number(private.get("elixir"), 10)
        scalars[0, 5] = _number(player.get("crowns"), 3)
        scalars[0, 6] = _number(opponent_player.get("crowns"), 3)
        scalars[0, 7] = sum(bool(item) for item in profile.get("entities", ())) / config.max_children_total
        scalars[0, 13] = float(candidates.mask.sum()) / config.max_action_candidates
        scalars[0, 14] = _number(_mapping(action_mask.get("reasons")).get("reserved_elixir"), 10)
        scalars[0, 15:17] = 1.0
        scalars[0, self.PREVIOUS_ACTION_STATUS_SCALAR] = previous_status

        # Construct empty V4 leaves with canonical shapes/padding. Current event
        # deltas are deliberately not forwarded to V4 in this initial policy.
        effect_n, relation_n = 64, 256
        effects = self._effects(profile, actor, tower_rows, group_rows)
        events = self.types.EventSetV4(_long(1, 0), _long(1, 0), _long(1, 0), _long(1, 0), _long(1, 0, fill=-1), _long(1, 0, fill=-1), _float(1, 0, config.event_feature_dim), _bool(1, 0))
        relations = self._relations(profile, actor, own, towers, group_rows, tower_rows)
        spatial = _float(1, config.explicit_spatial_channels, config.board_height, config.board_width)
        for entity in profile.get("entities", ()):
            if entity.get("owner") is None:
                continue
            xy = self._tile(entity["position"], actor)
            x, y = min(config.board_width - 1, max(0, int(xy[0]))), min(config.board_height - 1, max(0, int(xy[1])))
            own_side = self._owner(entity.get("owner"), actor) == self.OWNER_SELF
            base = 0 if own_side else 7
            kind = str(entity.get("entity_kind", "unknown"))
            category = 3 if kind == "projectile" else 2 if kind == "building" else 0
            spatial[0, base + category, y, x] += 1.0
            spatial[0, base + 4, y, x] += _number(entity.get("hitpoints"), 10_000)
            shield_hp = _known_or_raw(
                _mapping(entity.get("shield")).get("hitpoints")
            )
            spatial[0, base + 5, y, x] += _number(shield_hp, 10_000)
        batch = self.types.UniversalSemanticBatchV4(
            match_scalars=scalars, own_cards=own, opponent_cards=opponent, towers=towers, groups=groups,
            active_effects=effects, events=events, relation_edges=relations, explicit_spatial_planes=spatial,
            previous_action=previous, candidates=candidates,
        )
        if validate:
            batch.validate(
                config,
                card_vocab_size=self.bundle.card_catalog.vocab_size,
                ability_vocab_size=self.bundle.ability_catalog.vocab_size,
                entity_archetype_vocab_size=self.bundle.entity_archetype_catalog.vocab_size,
                effect_vocab_size=self.bundle.effect_catalog.vocab_size,
            )
        return TensorizedDecision(batch, slot_rows, slot_uids, actor)


class PolicyInputBuilder:
    """Bind the selected Observation, same-tick ActionMask, and TemporalContext."""

    def __init__(self, runtime: FirstLightRuntime, *, components: Any | None = None) -> None:
        self.components = components or load_v4_components(runtime)
        self.tensorizer = ObservationTensorizer(self.components)

    def build(
        self,
        observation: Any,
        action_mask: Any,
        temporal_context: TemporalContext,
        profile: Any,
    ) -> Any:
        from elixir_echo.policies.protocol import PolicyInput
        from elixir_echo.profiles import select_observation

        identity: DecisionIdentity = temporal_context.identity
        if action_mask.identity != identity:
            raise TensorizationError("ActionMask identity differs from current temporal context")
        selected = select_observation(observation, profile)
        tensors = self.tensorizer.tensorize(
            selected,
            actor=identity.actor,
            action_mask=action_mask.constraints,
            temporal_context=temporal_context,
        )
        return PolicyInput(
            identity=identity,
            profile=profile.value,
            tensors=tensors,
            action_mask=action_mask.constraints,
            temporal_context=temporal_context,
        )
