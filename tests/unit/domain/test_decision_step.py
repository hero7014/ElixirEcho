from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from elixir_echo.domain.action import ActionMask, ExpertAction
from elixir_echo.domain.observation import Observation
from elixir_echo.domain.trajectory import DecisionIdentity, DecisionStep
from elixir_echo.domain.values import ObservedValue


def _identity(*, tick: int = 90, actor: int = 0) -> DecisionIdentity:
    return DecisionIdentity("episode-1", tick, actor)


def _observation() -> Observation:
    return Observation(
        match={"remaining_ms": 120_000, "phase": "normal"},
        players=({"owner": 0, "relation": "self"},),
        towers=(),
        entities=(),
        terminal={"ended": False},
    )


def _step(*, identity: DecisionIdentity | None = None) -> DecisionStep:
    identity = identity or _identity()
    return DecisionStep(
        identity=identity,
        episode_context_ref="context-sha256",
        observation=_observation(),
        action_mask=ActionMask(
            identity=identity,
            constraints={"kinds": {"wait": True, "play_card": True}},
        ),
        expert_action=ExpertAction.from_legacy(
            ({"kind": "play_card", "slot": 2, "position": [9.0, 12.0]},)
        ),
        event_delta=({"kind": "unit_spawned", "owner": 1},),
        validation={"accepted": True},
        fingerprints={"observation_schema": "fixture"},
    )


def test_decision_identity_validates_atomic_coordinates() -> None:
    assert _identity().to_dict() == {"episode_id": "episode-1", "tick": 90, "actor": 0}

    for invalid in (
        ("", 90, 0),
        ("episode-1", -1, 0),
        ("episode-1", 90, 2),
    ):
        with pytest.raises(ValueError, match="invalid decision identity"):
            DecisionIdentity(*invalid)


def test_decision_step_requires_action_mask_identity_match() -> None:
    step_identity = _identity()
    mask_identity = _identity(tick=95)

    with pytest.raises(ValueError, match="action mask identity"):
        DecisionStep(
            identity=step_identity,
            episode_context_ref="context-sha256",
            observation=_observation(),
            action_mask=ActionMask(mask_identity, {"kinds": {"wait": True}}),
            expert_action=ExpertAction.wait(),
            event_delta=(),
            validation={},
            fingerprints={},
        )


def test_decision_step_is_atomic_immutable_and_hash_reproducible() -> None:
    first = _step()
    second = _step()
    payload = first.to_dict()

    assert payload["schema"] == "elixir_echo.decision-step@1"
    assert payload["observation"]["schema"] == "elixir_echo.observation@1"
    assert payload["action_mask"]["schema"] == "elixir_echo.action-mask@1"
    assert payload["identity"] == payload["action_mask"]["identity"]
    assert first.content_sha256() == second.content_sha256()

    with pytest.raises(TypeError):
        first.validation["accepted"] = False


def test_wait_and_action_labels_preserve_legacy_wire_values() -> None:
    assert ExpertAction.wait().to_legacy() is None
    assert ExpertAction.from_legacy(None).is_wait
    assert ExpertAction.from_legacy(({"kind": "play_card"},)).to_legacy() == [
        {"kind": "play_card"}
    ]


def test_canonical_step_matches_explicitly_renamed_legacy_fixture() -> None:
    canonical = _step().to_dict()
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_decision_step.json"
    legacy = json.loads(fixture.read_text(encoding="utf-8"))
    normalized = dict(legacy)
    normalized["schema"] = "elixir_echo.decision-step@1"
    normalized.pop("version")
    normalized["observation"] = normalized.pop("frame")
    normalized["observation"]["schema"] = "elixir_echo.observation@1"
    normalized["observation"].pop("version")
    normalized["action_mask"]["schema"] = "elixir_echo.action-mask@1"

    assert canonical == normalized


def test_public_domain_names_have_no_generation_suffix() -> None:
    public_types = (
        ActionMask,
        DecisionIdentity,
        DecisionStep,
        ExpertAction,
        Observation,
        ObservedValue,
    )

    assert all(re.search(r"V\d+$", item.__name__) is None for item in public_types)
