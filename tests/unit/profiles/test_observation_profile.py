from __future__ import annotations

import inspect

from elixir_echo.domain import Observation, ObservedValue
from elixir_echo.profiles.observation import (
    ObservationProfile,
    select_observation,
)


def observation() -> Observation:
    return Observation(
        match={
            "phase": "normal",
            "elapsed_ms": 10_000,
            "remaining_ms": 170_000,
            "elixir_multiplier": 1.0,
        },
        players=(
            {
                "owner": 0,
                "relation": "self",
                "crowns": 0,
                "tower_ids": (1,),
                "self_private": {
                    "elixir": 5.0,
                    "hand": (1, 2, 3, 4),
                    "next_card": 5,
                },
                "current_evolution": (
                    {
                        "card_id": 1,
                        "phase": "base",
                        "base_form_id": "Knight",
                        "current_form_id": "Knight",
                        "next_form_id": "Knight_EV1",
                        "cycle_required": 2,
                        "cycle_remaining": 1,
                        "ready": False,
                        "active": False,
                        "deployments_in_cycle": 1,
                    },
                ),
                "current_abilities": (
                    {
                        "ability_id": "KnightAbility",
                        "source_entity": 9,
                        "phase": "cooldown",
                        "remaining_cooldown_ms": 500,
                        "available": False,
                    },
                ),
            },
            {"owner": 1, "relation": "opponent", "crowns": 0, "tower_ids": (2,)},
        ),
        towers=(
            {
                "entity_id": 1,
                "owner": 0,
                "tower_kind": "king",
                "position": (9.0, 4.0),
                "hitpoints": 4000.0,
                "max_hitpoints": 4000.0,
                "tower_troop_id": None,
                "active": True,
                "attack": {"phase": "idle"},
            },
        ),
        entities=(
            {
                "entity_id": 9,
                "owner": 0,
                "card_id": 26_000_000,
                "current_form": ObservedValue.known(
                    {"variant": "base", "form_id": "base"}
                ),
                "entity_kind": "troop",
                "position": (9.0, 12.0),
                "hitpoints": 100.0,
                "max_hitpoints": 100.0,
                "visibility_basic": "visible",
                "velocity": (1.0, 0.0),
                "causal_relation": {
                    "kind": "deploy",
                    "handle": "deploy:6:6:1",
                    "source_card_id": 26_000_000,
                    "parent_entity_id": 4,
                },
                "attack": {"phase": "windup"},
                "visibility": {
                    "phase": "visible",
                    "targetable_by_owner": {"0": True},
                },
                "resources": ObservedValue.known(()),
            },
        ),
        terminal={"ended": False},
    )


def test_min_rich_are_canonical_names_with_core_full_compatibility_aliases() -> None:
    assert ObservationProfile.MIN.name == "MIN"
    assert ObservationProfile.RICH.name == "RICH"
    assert ObservationProfile.CORE is ObservationProfile.MIN
    assert ObservationProfile.FULL is ObservationProfile.RICH
    assert ObservationProfile.MIN.value == "core"
    assert ObservationProfile.RICH.value == "full"


def test_core_is_combat_complete_and_full_is_strictly_additive() -> None:
    core = select_observation(observation(), ObservationProfile.MIN)
    full = select_observation(observation(), ObservationProfile.RICH)

    assert core["players"][0]["self_private"]["hand"] == [1, 2, 3, 4]
    assert "tower_troop_id" in core["towers"][0]
    assert core["entities"][0]["visibility_basic"] == "visible"
    assert "velocity" not in core["entities"][0]
    assert "visibility" not in core["entities"][0]
    assert full["entities"][0]["velocity"] == [1.0, 0.0]
    assert full["entities"][0]["visibility"]["phase"] == "visible"
    assert "resources" not in full["entities"][0]


def test_core_player_runtime_is_coarse_and_full_is_detailed() -> None:
    core_player = select_observation(observation(), ObservationProfile.MIN)[
        "players"
    ][0]
    full_player = select_observation(observation(), ObservationProfile.RICH)[
        "players"
    ][0]

    assert set(core_player["current_evolution"][0]) == {
        "card_id",
        "current_form_id",
        "ready",
        "active",
    }
    assert "current_abilities" not in core_player
    assert full_player["current_evolution"][0]["cycle_remaining"] == 1
    assert full_player["current_abilities"][0]["remaining_cooldown_ms"] == 500


def test_full_relations_exclude_provider_temporary_handles() -> None:
    relation = select_observation(observation(), ObservationProfile.RICH)["entities"][
        0
    ]["causal_relation"]

    assert relation == {
        "kind": "deploy",
        "source_card_id": 26_000_000,
        "parent_entity_id": 4,
    }


def test_profile_api_accepts_only_observation_and_profile() -> None:
    assert tuple(inspect.signature(select_observation).parameters) == (
        "observation",
        "profile",
    )
