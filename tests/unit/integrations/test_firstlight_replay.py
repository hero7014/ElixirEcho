from __future__ import annotations

from types import SimpleNamespace

import pytest

from elixir_echo.domain import Observation
from elixir_echo.integrations.firstlight.replay import (
    FirstLightReplayProvider,
    ReplayAdapterError,
    ReplayLabelLeakageError,
    ReplayReconstructionError,
    TimedExpertAction,
    group_timed_expert_actions,
)


class _Adapter:
    def adapt(self, source, *, actor):
        if getattr(source, "reject", False):
            raise ValueError("fixture rejected")
        return Observation(
            match={"phase": "normal", "remaining_ms": 120_000 - source.tick * 100},
            players=({"owner": actor, "relation": "self"},),
            towers=(),
            entities=(),
            terminal={"ended": False},
        )


def _observation(actor: int, tick: int):
    return SimpleNamespace(
        owner=actor,
        tick=tick,
        action_mask={"identity": {"tick": tick, "actor": actor}, "kinds": {"wait": True, "play_card": True}},
        events=({"tick": tick, "kind": "public_tick_event", "native_handle": 44},),
    )


def _step_wait_factory(*, terminal_after=3):
    calls = []

    def step_wait(ticks):
        calls.append(ticks)
        current_tick = 90 + len(calls) * 5
        observations = {actor: _observation(actor, current_tick) for actor in (0, 1)}
        is_terminal = len(calls) >= terminal_after
        flags = {actor: is_terminal for actor in (0, 1)}
        return observations, {0: 0, 1: 0}, flags, {0: False, 1: False}, {0: {"terminal_reason": "fixture"}, 1: {"terminal_reason": "fixture"}}

    return step_wait, calls


def test_capture_is_continuous_actor_paired_and_labels_stay_after_current_input():
    step_wait, calls = _step_wait_factory()
    labels = {
        (90, 0): (TimedExpertAction(
            decision_tick=90,
            source_tick=92,
            observable_tick=93,
            actor=0,
            action_id="label-action-0",
            action={"kind": "play_card", "slot": 2},
        ),)
    }
    captured = FirstLightReplayProvider(_Adapter()).capture_timeline(
        episode_id="ep-1",
        episode_context_ref="context-ref",
        initial_observations={actor: _observation(actor, 90) for actor in (0, 1)},
        expert_actions=labels,
        step_wait=step_wait,
        fingerprints={"source": "fixture"},
    )

    assert captured.outcome == "terminated"
    assert captured.decision_tick_count == 3
    assert captured.label_leakage_checks == 6
    assert calls == [5, 5, 5]
    assert [(item.identity.tick, item.identity.actor) for item in captured.records] == [
        (90, 0), (90, 1), (95, 0), (95, 1), (100, 0), (100, 1)
    ]
    assert captured.records[0].expert_action.actions[0]["kind"] == "play_card"
    assert "label-action-0" not in str(captured.records[0].observation.to_dict())
    assert "native_handle" not in str(captured.records[0].event_delta)
    assert all(record.validation["label_leakage_asserted"] for record in captured.records)


def test_current_label_observable_at_input_tick_is_rejected():
    step_wait, _ = _step_wait_factory()
    labels = {
        (90, 0): (TimedExpertAction(90, 90, 90, 0, "too-early", {"kind": "play_card"}),)
    }
    with pytest.raises(ReplayLabelLeakageError, match="observable"):
        FirstLightReplayProvider(_Adapter()).capture_timeline(
            episode_id="ep-1",
            episode_context_ref="context-ref",
            initial_observations={actor: _observation(actor, 90) for actor in (0, 1)},
            expert_actions=labels,
            step_wait=step_wait,
            fingerprints={},
            max_decision_steps=1,
        )


def test_adapter_and_native_reconstruction_failures_remain_distinct():
    step_wait, _ = _step_wait_factory()
    observations = {actor: _observation(actor, 90) for actor in (0, 1)}
    observations[1].reject = True
    with pytest.raises(ReplayAdapterError):
        FirstLightReplayProvider(_Adapter()).capture_timeline(
            episode_id="ep-1",
            episode_context_ref="context-ref",
            initial_observations=observations,
            expert_actions={},
            step_wait=step_wait,
            fingerprints={},
            max_decision_steps=1,
        )

    def broken_step(_ticks):
        raise ConnectionError("native unavailable")

    with pytest.raises(ReplayReconstructionError, match="step failed"):
        FirstLightReplayProvider(_Adapter()).capture_timeline(
            episode_id="ep-1",
            episode_context_ref="context-ref",
            initial_observations={actor: _observation(actor, 90) for actor in (0, 1)},
            expert_actions={},
            step_wait=broken_step,
            fingerprints={},
            max_decision_steps=1,
        )


def test_replay_action_grouping_preserves_decision_window_without_raw_action_id():
    action = SimpleNamespace(
        kind="play_card",
        owner=1,
        hand_slot=2,
        card_id=26_000_025,
        source_entity=None,
        ability_id=None,
        target_kind="grid",
        target_grid=(8, 10),
        target_entity=None,
        subcell_offset=None,
        action_id="provider-only-label-id",
        metadata={"source_command_tick": 97, "source_index": 3, "native_observable_tick": 98},
    )
    replay = SimpleNamespace(operations=(SimpleNamespace(actions=(action,)),))

    grouped = group_timed_expert_actions(replay)

    assert tuple(grouped) == ((95, 1),)
    label = grouped[(95, 1)][0]
    assert label.source_tick == 97
    assert label.observable_tick == 98
    assert label.action["execute_offset_ticks"] == 2
    assert "action_id" not in label.action
