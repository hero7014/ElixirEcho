from __future__ import annotations

import copy

import pytest

from elixir_echo.domain import ObservedValue
from elixir_echo.integrations.firstlight.observation import (
    FirstLightAdapterError,
    FirstLightObservationAdapter,
)
from tests.fixtures.observations import (
    entity,
    fair_observation,
    namespace,
    tower,
)


def test_adapter_is_read_only_actor_fair_and_drops_provider_private_fields() -> None:
    source = fair_observation(actor=0)
    before = copy.deepcopy(source)

    adapted = FirstLightObservationAdapter().adapt(source, actor=0)
    payload = adapted.to_dict()

    assert source.__dict__ == before.__dict__
    assert payload["players"][0]["relation"] == "self"
    assert "self_private" in payload["players"][0]
    assert "self_private" not in payload["players"][1]
    encoded = str(payload).casefold()
    for forbidden in (
        "native_digest",
        "metadata",
        "raster",
        "opponent_belief",
        "action_mask",
        "events",
        "provenance",
    ):
        assert forbidden not in encoded


def test_adapter_rejects_wrong_tier_or_actor() -> None:
    source = fair_observation(actor=0)
    source.tier = namespace(value="oracle")
    with pytest.raises(FirstLightAdapterError, match="FAIR"):
        FirstLightObservationAdapter().adapt(source, actor=0)

    source = fair_observation(actor=0)
    with pytest.raises(FirstLightAdapterError, match="actor"):
        FirstLightObservationAdapter().adapt(source, actor=1)


@pytest.mark.parametrize(
    ("phase", "exposes_target"),
    [
        ("acquiring", True),
        ("windup", True),
        ("release", True),
        ("channel", True),
        ("charging", True),
        ("idle", False),
        ("cooldown", False),
        ("backswing", False),
        ("interrupted", False),
        ("unknown", False),
    ],
)
def test_attack_target_requires_a_proven_current_attack_phase(
    phase: str, exposes_target: bool
) -> None:
    source = fair_observation()
    source.entities = (entity(attack_phase=phase),)
    source.towers = (tower(attack_phase=phase),)

    payload = FirstLightObservationAdapter().adapt(source, actor=0).to_dict()

    adapted_entity = payload["entities"][0]
    adapted_tower = payload["towers"][0]
    assert ("target_entity" in adapted_entity.get("attack", {})) is exposes_target
    assert ("public_target_id" in adapted_entity) is exposes_target
    assert ("target_entity" in adapted_tower.get("attack", {})) is exposes_target


@pytest.mark.parametrize(
    ("phase", "coarse"),
    [
        ("visible", "visible"),
        ("hidden", "hidden"),
        ("burrowed", "hidden"),
        ("hiding", "transitioning"),
        ("revealing", "transitioning"),
        ("unknown", "unknown"),
    ],
)
def test_coarse_visibility_is_derived_from_known_phase(
    phase: str, coarse: str
) -> None:
    source = fair_observation()
    source.entities = (entity(visibility_phase=phase),)

    adapted = FirstLightObservationAdapter().adapt(source, actor=0).to_dict()[
        "entities"
    ][0]

    assert adapted["visibility_basic"] == coarse
    assert adapted["visibility"]["phase"] == phase


def test_projectile_presence_in_fair_list_proves_coarse_visibility() -> None:
    projectile = entity(entity_kind="projectile")
    projectile.visibility_state = None
    projectile.projectile_state = namespace(provenance=None)
    source = fair_observation()
    source.entities = (projectile,)

    adapted = FirstLightObservationAdapter().adapt(source, actor=0).to_dict()[
        "entities"
    ][0]

    assert adapted["visibility_basic"] == "visible"
    assert "visibility" not in adapted


def test_current_form_distinguishes_base_evolved_and_not_applicable() -> None:
    evolved = namespace(
        card_id=26_000_025,
        deck_slot=None,
        phase="evolved",
        base_form_id="SkeletonWarriors",
        current_form_id="SkeletonWarriors_EV1",
        next_form_id=None,
        cycle_required=None,
        cycle_remaining=None,
        ready=None,
        active=True,
        deployments_in_cycle=None,
        provenance=None,
    )
    source = fair_observation()
    source.entities = (
        entity(entity_id=9),
        entity(entity_id=10, evolution_state=evolved),
        entity(entity_id=11, entity_kind="projectile"),
    )

    values = FirstLightObservationAdapter().adapt(source, actor=0).entities

    assert values[0]["current_form"] == ObservedValue.known(
        {"variant": "base", "form_id": "base"}
    )
    assert values[1]["current_form"] == ObservedValue.known(
        {"variant": "evolved", "form_id": "SkeletonWarriors_EV1"}
    )
    assert values[2]["current_form"] == ObservedValue.not_applicable()


def test_hero_mapping_cannot_reclassify_a_projectile_id_collision() -> None:
    adapter = FirstLightObservationAdapter(
        runtime_hero_forms={203_000_034: (26_000_034, "Bowler_hero")}
    )
    source = fair_observation()
    source.entities = (
        entity(card_id=203_000_034, entity_kind="projectile"),
        entity(entity_id=10, card_id=203_000_034, entity_kind="unknown"),
    )

    projectile, hero = adapter.adapt(source, actor=0).entities

    assert projectile["card_id"] == 203_000_034
    assert projectile["entity_kind"] == "projectile"
    assert projectile["current_form"] == ObservedValue.not_applicable()
    assert hero["card_id"] == 26_000_034
    assert hero["entity_kind"] == "hero"
    assert hero["current_form"] == ObservedValue.known(
        {"variant": "hero", "form_id": "Bowler_hero"}
    )
