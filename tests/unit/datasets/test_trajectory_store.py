from __future__ import annotations

import json

import pytest

from elixir_echo.datasets.trajectory import (
    TrainingTrajectoryReader,
    TrajectoryReader,
    TrajectoryWriter,
)
from elixir_echo.domain.action import ActionMask, ExpertAction
from elixir_echo.domain.observation import Observation
from elixir_echo.domain.trajectory import DecisionIdentity, DecisionStep


def _step(tick: int, actor: int) -> DecisionStep:
    identity = DecisionIdentity("episode-1", tick, actor)
    return DecisionStep(
        identity=identity,
        episode_context_ref="opaque-context-hash",
        observation=Observation(
            match={"phase": "normal", "remaining_ms": 120_000},
            players=({"owner": actor, "relation": "self", "self_private": {"elixir": 5}},),
            towers=(),
            entities=(),
            terminal={"ended": False},
        ),
        action_mask=ActionMask(identity, {"kinds": {"wait": True, "play_card": True}}),
        expert_action=ExpertAction.wait() if actor == 0 else ExpertAction.from_legacy({"kind": "play_card", "slot": 1}),
        event_delta=({"kind": "unit_spawned", "card_id": 7},),
        validation={"accepted": True, "provider_diagnostic": "do-not-train"},
        fingerprints={"provider_path": "C:/private/FirstLight"},
    )


def _writer(root, *, outcome="terminated"):
    return TrajectoryWriter(root).begin(
        {
            "episode_id": "episode-1",
            "replay_tag": "replay-identity",
            "outcome": outcome,
            "split": "train",
            "episode_context_ref": "opaque-context-hash",
            "episode_context": {"deck0": list(range(8)), "deck1": list(range(8, 16))},
            "fingerprints": {"provider_diagnostic": "private-path"},
        }
    )


def test_writer_commits_ordered_episode_with_reproducible_hashes_and_hidden_context(tmp_path):
    writer = _writer(tmp_path)
    for tick in (90, 95):
        for actor in (0, 1):
            writer.append(_step(tick, actor))
    manifest = writer.commit()

    assert manifest.record_count == 4
    assert manifest.outcome == "terminated"
    assert manifest.record_hashes == tuple(record.content_sha256() for record in TrajectoryReader(tmp_path).read_episode("episode-1").records)
    assert manifest.aggregate_hash == TrajectoryReader(tmp_path).read_episode("episode-1").manifest.aggregate_hash

    training_episode = TrainingTrajectoryReader(tmp_path).read_episode("episode-1")
    assert [(item.identity.tick, item.identity.actor) for item in training_episode.steps] == [
        (90, 0), (90, 1), (95, 0), (95, 1)
    ]
    serialized_training = json.dumps(training_episode.to_dict(), sort_keys=True)
    for forbidden in ("deck0", "deck1", "provider_diagnostic", "provider_path", "episode_context"):
        assert forbidden not in serialized_training
    assert "event_delta" in serialized_training


def test_writer_rejects_non_strict_order_and_duplicate_identity(tmp_path):
    writer = _writer(tmp_path)
    writer.append(_step(90, 1))
    with pytest.raises(ValueError, match=r"strict \(tick, actor\) order"):
        writer.append(_step(90, 0))
    writer = _writer(tmp_path / "duplicate")
    writer.append(_step(90, 1))
    with pytest.raises(ValueError, match="duplicate"):
        writer.append(_step(90, 1))


def test_episode_identity_mismatch_fails_without_visible_partial_episode(tmp_path):
    writer = _writer(tmp_path)
    identity = DecisionIdentity("another-episode", 90, 0)
    bad = _step(90, 0)
    bad = DecisionStep(
        identity=identity,
        episode_context_ref=bad.episode_context_ref,
        observation=bad.observation,
        action_mask=ActionMask(identity, bad.action_mask.constraints),
        expert_action=bad.expert_action,
        event_delta=bad.event_delta,
        validation=bad.validation,
        fingerprints=bad.fingerprints,
    )
    with pytest.raises(ValueError, match="episode identity"):
        writer.append(bad)
    assert tuple(TrajectoryReader(tmp_path).iter_episodes()) == ()
    writer.abort()


def test_commit_failure_cleans_staging_and_never_exposes_half_episode(tmp_path, monkeypatch):
    writer = _writer(tmp_path)
    writer.append(_step(90, 0))
    writer.append(_step(90, 1))

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated atomic rename failure")

    monkeypatch.setattr("elixir_echo.datasets.trajectory.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        writer.commit()
    assert tuple(TrajectoryReader(tmp_path).iter_episodes()) == ()
    assert not tuple(tmp_path.rglob(".staging-*"))


@pytest.mark.parametrize("outcome", ["terminated", "truncated"])
def test_manifest_records_terminal_or_truncated_outcome(tmp_path, outcome):
    writer = _writer(tmp_path, outcome=outcome)
    writer.append(_step(90, 0))
    writer.append(_step(90, 1))
    writer.commit()
    assert TrainingTrajectoryReader(tmp_path).read_episode("episode-1").outcome == outcome


def test_reader_rejects_record_tampering_and_aggregate_hash_mismatch(tmp_path):
    writer = _writer(tmp_path)
    writer.append(_step(90, 0))
    writer.append(_step(90, 1))
    writer.commit()
    episode = TrajectoryReader(tmp_path).episode_path("episode-1")
    record_file = episode / "records.jsonl"
    payload = json.loads(record_file.read_text(encoding="utf-8").splitlines()[0])
    payload["expert_action"] = [{"kind": "play_card", "slot": 7}]
    lines = record_file.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(payload)
    record_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record hash mismatch"):
        TrajectoryReader(tmp_path).read_episode("episode-1")
