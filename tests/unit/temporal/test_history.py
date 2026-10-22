from __future__ import annotations

import pytest

from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation
from elixir_echo.temporal.history import History, PreviousActionStatus


def step(tick: int, *, action=(), actor: int = 0, episode_id: str = "episode-a", events=()):
    identity = DecisionIdentity(episode_id, tick, actor)
    return DecisionStep(
        identity=identity,
        episode_context_ref="context-ref",
        observation=Observation(
            match={"phase": "normal", "remaining_ms": 180_000 - tick * 50},
            players=({"owner": actor, "relation": "self"},),
            towers=(),
            entities=(),
            terminal={"ended": False},
        ),
        action_mask=ActionMask(identity, {"kinds": {"wait": True}}),
        expert_action=ExpertAction(tuple(action)),
        event_delta=tuple(events),
        validation={"accepted": True},
        fingerprints={},
    )


def play_action(actor: int = 0):
    return {
        "owner": actor,
        "kind": "play_card",
        "hand_slot": 2,
        "card_id": 26_000_021,
        "target_kind": "grid",
        "target_grid": [8, 12],
        "execute_offset_ticks": 1,
    }


def test_current_label_is_hidden_until_committed_and_then_becomes_previous():
    first = step(90, action=(play_action(),))
    history = History()
    history.reset(first.identity)

    current = history.before_step(first)
    assert current.previous_action.status is PreviousActionStatus.START
    assert current.prior_entries == ()

    history.after_action(first.expert_action)
    second = step(95, action=({"kind": "activate_ability", "owner": 0},))
    following = history.before_step(second)
    assert following.previous_action.status is PreviousActionStatus.KNOWN
    assert following.previous_action.source_tick == 90
    assert following.previous_action.actions[0]["kind"] == "play_card"
    assert [entry.identity.tick for entry in following.prior_entries] == [90]


def test_wait_and_unrepresentable_real_action_have_distinct_statuses():
    history = History()
    first = step(90)
    history.reset(first.identity)
    history.before_step(first)
    history.after_action(first.expert_action)
    second = step(95, action=({"kind": "activate_ability", "owner": 0},))
    after_wait = history.before_step(second)
    assert after_wait.previous_action.status is PreviousActionStatus.KNOWN
    assert after_wait.previous_action.actions == ()

    history.after_action(second.expert_action)
    after_ability = history.before_step(step(100))
    assert after_ability.previous_action.status is PreviousActionStatus.UNKNOWN_MASKED
    assert after_ability.previous_action.actions == ()
    assert after_ability.previous_action.source_tick == 95


def test_reset_backward_time_and_missing_action_commit_fail_closed():
    history = History()
    first = step(90)
    history.reset(first.identity)
    history.before_step(first)
    with pytest.raises(ValueError, match="committed"):
        history.before_step(step(95))
    history.after_action(first.expert_action)
    history.before_step(step(95))
    with pytest.raises(ValueError, match="committed"):
        history.before_step(step(90))

    reset = step(90, actor=1, episode_id="episode-b")
    history.reset(reset.identity)
    view = history.before_step(reset)
    assert view.previous_action.status is PreviousActionStatus.START
    assert view.prior_entries == ()


def test_future_or_malformed_public_event_is_rejected():
    history = History()
    current = step(90, events=({"tick": 91, "event_type": "future"},))
    history.reset(current.identity)
    with pytest.raises(ValueError, match="causally"):
        history.before_step(current)
