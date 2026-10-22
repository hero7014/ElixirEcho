from __future__ import annotations

import json

import pytest

from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation
from elixir_echo.temporal.history import PreviousActionStatus
from elixir_echo.temporal.public_state import PublicStateTracker


def step(tick: int, events=(), *, actor: int = 0, episode_id: str = "episode-a", action=()):
    identity = DecisionIdentity(episode_id, tick, actor)
    return DecisionStep(
        identity,
        "context-ref",
        Observation(
            match={"phase": "normal", "remaining_ms": 180_000 - tick * 50},
            players=({"owner": actor, "relation": "self"},),
            towers=(),
            entities=(),
            terminal={"ended": False},
        ),
        ActionMask(identity, {"kinds": {"wait": True}}),
        ExpertAction(tuple(action)),
        tuple(events),
        {"accepted": True},
        {},
    )


def play_event(tick: int, owner: int, card_id: int, **extra):
    return {
        "tick": tick,
        "event_type": "action_executed",
        "owner": owner,
        "card_id": card_id,
        "position": [8.0, 12.0],
        **extra,
    }


def test_public_reveals_play_history_and_strict_cycle_facts():
    tracker = PublicStateTracker()
    first = step(90, (
        play_event(90, 1, 26_000_101, native_handle=998),
        play_event(90, 0, 26_000_201),
    ))
    at_90 = tracker.before_step(first)
    assert at_90.public_state.revealed_cards == (26_000_101,)
    assert [(play.owner, play.card_id) for play in at_90.public_state.public_play_history] == [
        (1, 26_000_101), (0, 26_000_201)
    ]
    assert at_90.public_state.public_plays_since_last_use[0].count == 0
    tracker.after_action(first.expert_action)

    second = step(95, (play_event(95, 1, 26_000_102),))
    at_95 = tracker.before_step(second)
    assert at_95.public_state.revealed_cards == (26_000_101, 26_000_102)
    assert [(item.card_id, item.count, item.capped) for item in at_95.public_state.public_plays_since_last_use] == [
        (26_000_101, 1, False), (26_000_102, 0, False)
    ]


def test_spawn_can_reveal_a_card_but_does_not_prove_a_play_cycle():
    tracker = PublicStateTracker()
    current = step(90, ({"tick": 90, "event_type": "spawn", "owner": 1, "card_id": 26_000_301},))
    context = tracker.before_step(current)
    assert context.public_state.revealed_cards == (26_000_301,)
    assert context.public_state.public_play_history == ()
    assert context.public_state.public_plays_since_last_use == ()


def test_unknown_previous_action_is_not_faked_as_wait_and_handle_is_not_retained():
    tracker = PublicStateTracker()
    first = step(
        90,
        (play_event(90, 1, 26_000_101, native_handle=123),),
        action=({"kind": "activate_ability", "owner": 0},),
    )
    tracker.before_step(first)
    tracker.after_action(first.expert_action)
    next_context = tracker.before_step(step(95))
    assert next_context.previous_action.status is PreviousActionStatus.UNKNOWN_MASKED
    assert next_context.previous_action.actions == ()
    encoded = json.dumps(next_context.public_state.to_dict(), sort_keys=True)
    assert "native_handle" not in encoded
    assert "deck" not in encoded
    assert "elixir" not in encoded


def test_current_expert_label_does_not_change_current_temporal_context():
    wait = step(90)
    ability = step(90, action=({"kind": "activate_ability", "owner": 0},))
    wait_context = PublicStateTracker().before_step(wait)
    ability_context = PublicStateTracker().before_step(ability)
    assert wait_context == ability_context


def test_future_and_late_events_and_backward_decisions_are_rejected():
    tracker = PublicStateTracker()
    with pytest.raises(ValueError, match="publicly available"):
        tracker.before_step(step(90, (play_event(91, 1, 26_000_101),)))

    first = step(90, (play_event(90, 1, 26_000_101),))
    tracker.before_step(first)
    tracker.after_action(first.expert_action)
    with pytest.raises(ValueError, match="expected tick"):
        tracker.before_step(step(100))


def test_reset_starts_a_new_actor_sequence_without_carrying_public_facts():
    tracker = PublicStateTracker()
    first = step(90, (play_event(90, 1, 26_000_101),))
    tracker.before_step(first)
    tracker.after_action(first.expert_action)

    identity = DecisionIdentity("episode-b", 90, 1)
    tracker.reset(identity)
    second = step(90, actor=1, episode_id="episode-b")
    context = tracker.before_step(second)
    assert context.previous_action.status is PreviousActionStatus.START
    assert context.public_state.revealed_cards == ()
    assert context.history.prior_entries == ()


def test_future_suffix_does_not_change_an_earlier_tracker_snapshot():
    prefix = (play_event(90, 1, 26_000_101),)
    snapshots = []
    for card_id in (26_000_102, 26_000_103):
        tracker = PublicStateTracker()
        context = tracker.before_step(step(90, prefix))
        snapshots.append(context.public_state.to_dict())
        tracker.after_action(step(90).expert_action)
        tracker.before_step(step(95, (play_event(95, 1, card_id),)))
    assert snapshots[0] == snapshots[1]
