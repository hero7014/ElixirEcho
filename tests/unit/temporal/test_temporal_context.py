from __future__ import annotations

from elixir_echo.datasets import TrainingTrajectoryReader, TrajectoryWriter
from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation
from elixir_echo.temporal.public_state import PublicStateTracker


def make_step(tick: int, actor: int, events=(), action=()):
    identity = DecisionIdentity("reader-episode", tick, actor)
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


def test_training_reader_only_drives_actor_local_temporal_context(tmp_path):
    writer = TrajectoryWriter(tmp_path / "cache")
    writer.begin({
        "episode_id": "reader-episode",
        "replay_tag": "fixture",
        "outcome": "terminated",
        "episode_context_ref": "context-ref",
        "deck0": [26_000_001] * 8,
        "deck1": [26_000_002] * 8,
    })
    for tick in (90, 95):
        for actor in (0, 1):
            events = (
                {"tick": 90, "event_type": "action_executed", "owner": 1 - actor, "card_id": 26_000_101},
            ) if tick == 90 else ()
            action = ({"kind": "activate_ability", "owner": actor},) if tick == 90 else ()
            writer.append(make_step(tick, actor, events, action))
    writer.commit()

    episode = TrainingTrajectoryReader(tmp_path / "cache").read_episode("reader-episode")
    assert not hasattr(episode, "metadata")
    assert "deck" not in str(episode.to_dict()).casefold()

    trackers = {actor: PublicStateTracker() for actor in (0, 1)}
    contexts = {}
    for step in episode.steps:
        tracker = trackers[step.identity.actor]
        contexts[(step.identity.tick, step.identity.actor)] = tracker.before_step(step)
        tracker.after_action(step.expert_action)

    for actor in (0, 1):
        current = contexts[(95, actor)]
        assert current.previous_action.status.value == "unknown_masked"
        assert current.previous_action.source_tick == 90
        assert current.public_state.revealed_cards == (26_000_101,)
        assert [entry.identity.tick for entry in current.history.prior_entries] == [90]
