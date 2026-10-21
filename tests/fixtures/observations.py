"""Synthetic FirstLight-shaped observations used without importing FirstLight."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def namespace(**values: Any) -> SimpleNamespace:
    return SimpleNamespace(**values)


def visibility(phase: str) -> SimpleNamespace:
    return namespace(
        phase=phase,
        public_by_owner={},
        targetable_by_owner={},
        area_damage_eligible=None,
        transition_remaining_ms=None,
        provenance=None,
    )


def attack(phase: str, target_entity: int = 99) -> SimpleNamespace:
    return namespace(
        phase=phase,
        target_entity=target_entity,
        target_position=None,
        phase_remaining_ms=None,
        cooldown_remaining_ms=None,
        sequence_index=0,
        sequence_progress=None,
        sequence_progress_limit=None,
        sequence_decay_remaining_ms=None,
        sequence_decay_duration_ms=None,
        charge_stage=None,
        charge_elapsed_ms=None,
        damage_multiplier=None,
        locked=None,
        interrupted=False,
        provenance=None,
    )


def entity(
    *,
    entity_id: int = 9,
    card_id: int = 26_000_025,
    entity_kind: str = "troop",
    visibility_phase: str = "visible",
    attack_phase: str = "unknown",
    evolution_state: Any = None,
) -> SimpleNamespace:
    return namespace(
        entity_id=entity_id,
        owner=0,
        card_id=card_id,
        entity_kind=entity_kind,
        position=(1.0, 2.0),
        velocity=(0.0, 0.0),
        hitpoints=100.0,
        max_hitpoints=100.0,
        shield=None,
        age_ms=100,
        visible_target=99,
        source_entity=None,
        causal_group=None,
        shield_state=None,
        effect_states=(),
        attack_state=attack(attack_phase),
        projectile_state=None,
        visibility_state=visibility(visibility_phase),
        ability_states=(),
        evolution_state=evolution_state,
        resource_states=(),
        capture_runtime=None,
        threshold_relocation_runtime=None,
        periodic_attack_modifier=None,
        runtime_provenance=None,
    )


def tower(*, attack_phase: str = "unknown") -> SimpleNamespace:
    return namespace(
        entity_id=1,
        owner=0,
        tower_kind="princess_left",
        position=(3.0, 4.0),
        hitpoints=1000.0,
        max_hitpoints=1000.0,
        tower_troop_id=159_000_000,
        shield=0.0,
        active=True,
        visible_target=99,
        shield_state=None,
        effect_states=(),
        attack_state=attack(attack_phase),
        visibility_state=visibility("visible"),
        tower_troop_runtime=None,
        runtime_provenance=None,
    )


def fair_observation(*, actor: int = 0) -> SimpleNamespace:
    own = namespace(
        owner=actor,
        crowns=0,
        tower_ids=(1,),
        elixir_exact=5.0,
        hand=(1, 2, 3, 4),
        next_card=5,
        ability_runtime_states=(),
        evolution_runtime_states=(),
    )
    opponent = namespace(
        owner=1 - actor,
        crowns=0,
        tower_ids=(2,),
        elixir_exact=9.0,
        elixir_visible=None,
        hand=(6, 7, 8, 9),
        next_card=10,
        ability_runtime_states=(),
        evolution_runtime_states=(),
    )
    return namespace(
        tier=namespace(value="fair"),
        owner=actor,
        tick=90,
        phase="normal",
        time=namespace(
            elapsed_ms=4500,
            remaining_ms=175_500,
            elixir_multiplier=1.0,
        ),
        players=(own, opponent),
        towers=(),
        entities=(),
        events=(namespace(secret="x"),),
        action_mask=namespace(),
        opponent_belief=namespace(),
        pending_actions=(),
        terminal=namespace(
            ended=False,
            winner=None,
            result_by_owner=(0.0, 0.0),
            reason=None,
            terminal_tick=None,
        ),
        native_digest="secret",
        raster=namespace(),
        metadata={"secret": True},
    )
