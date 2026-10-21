from __future__ import annotations

import re

import pytest

from elixir_echo.domain.observation import Observation
from elixir_echo.domain.values import DomainContractError


def _observation() -> Observation:
    return Observation(
        match={"phase": "normal", "remaining_ms": 120_000},
        players=(
            {
                "owner": 0,
                "relation": "self",
                "self_private": {"elixir": 5.0, "hand": [1, 2, 3, 4]},
            },
            {"owner": 1, "relation": "opponent"},
        ),
        towers=({"owner": 0, "kind": "king", "hp": 4_000},),
        entities=({"owner": 1, "card_id": 26_000_004, "position": [9.5, 18.0]},),
        terminal={"ended": False},
    )


def test_observation_is_actor_conditioned_current_state() -> None:
    observation = _observation()
    payload = observation.to_dict()

    assert payload["schema"] == "elixir_echo.observation@1"
    assert payload["perspective"] == "actor_fair"
    assert "actor" not in payload
    assert "action_mask" not in payload
    assert "expert_action" not in payload
    assert observation.content_sha256() == observation.content_sha256()


@pytest.mark.parametrize(
    "section,value",
    [
        ("match", {"provenance": {"native": "internal"}}),
        ("players", ({"deck": [1, 2, 3, 4, 5, 6, 7, 8]},)),
        ("entities", ({"action_mask": {"wait": True}},)),
        ("terminal", {"expert_action": {"kind": "wait"}}),
    ],
)
def test_observation_rejects_cache_private_and_policy_fields(
    section: str,
    value: object,
) -> None:
    values = {
        "match": {},
        "players": (),
        "towers": (),
        "entities": (),
        "terminal": {},
    }
    values[section] = value

    with pytest.raises(DomainContractError, match="forbidden Observation field"):
        Observation(**values)


def test_observation_requires_actor_fair_perspective() -> None:
    with pytest.raises(DomainContractError, match="perspective must be actor_fair"):
        Observation(
            match={},
            players=(),
            towers=(),
            entities=(),
            terminal={},
            perspective="omniscient",
        )


def test_observation_is_recursively_immutable_and_unversioned_in_python() -> None:
    observation = _observation()

    with pytest.raises(TypeError):
        observation.match["phase"] = "overtime"

    with pytest.raises(TypeError):
        observation.players[0]["self_private"]["hand"][0] = 9

    assert re.search(r"V\d+$", type(observation).__name__) is None
