from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from elixir_echo.datasets.trajectory import TrainingEpisode
from elixir_echo.integrations.firstlight.trainer import (
    FirstLightILTrainer,
    TrainingCheckpointRef,
    TrainingPassResult,
    TrainingRunProgress,
)


def _prefix_digest(episode_ids):
    digest = hashlib.sha256(b"").hexdigest()
    for episode_id in episode_ids:
        encoded = episode_id.encode("utf-8")
        digest = hashlib.sha256(
            bytes.fromhex(digest) + len(encoded).to_bytes(8, "big") + encoded
        ).hexdigest()
    return digest


def test_episode_checkpoint_runner_resumes_from_last_atomic_boundary(monkeypatch, tmp_path):
    trainer = object.__new__(FirstLightILTrainer)
    trainer.config = SimpleNamespace(fingerprint=lambda: "config-fingerprint")
    trainer.backend = SimpleNamespace(model=object())
    episodes = tuple(
        TrainingEpisode(f"episode-{index}", "terminated", "train", ())
        for index in range(1, 5)
    )
    trained_ids = []
    saves = []

    def train_one(items, *, input_builder, optimizer, max_episodes=None):
        episode = tuple(items)[0]
        trained_ids.append(episode.episode_id)
        return TrainingPassResult(1, 2, 2, 2, 0, 0, 0, 2, (0.4, 0.3))

    def save(path, *, optimizer, update_step, progress):
        reference = TrainingCheckpointRef(
            "checkpoint-id",
            path,
            update_step,
            "config-fingerprint",
            progress,
        )
        saves.append(reference)
        return reference

    monkeypatch.setattr(trainer, "train_episodes", train_one)
    monkeypatch.setattr(trainer, "save_checkpoint", save)

    def interrupted_source():
        yield from episodes[:2]
        raise RuntimeError("simulated process interruption")

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        trainer.train_episodes_resumable(
            interrupted_source(),
            input_builder=object(),
            optimizer=object(),
            dataset_fingerprint="cache-fingerprint",
            data_split_fingerprint="e" * 64,
            checkpoint_directory=tmp_path,
            checkpoint_every=2,
        )

    assert len(saves) == 1
    assert saves[0].progress.completed_episode_count == 2
    assert saves[0].progress.completed_episode_prefix_sha256 == _prefix_digest(
        ("episode-1", "episode-2")
    )
    assert trained_ids == ["episode-1", "episode-2"]

    resumed_checkpoint = saves[0]
    monkeypatch.setattr(
        trainer,
        "_read_checkpoint_progress",
        lambda path, *, expected_dataset_fingerprint, expected_data_split_fingerprint: resumed_checkpoint.progress,
    )
    restore_calls = []
    monkeypatch.setattr(
        trainer,
        "restore_checkpoint",
        lambda path, *, optimizer, expected_dataset_fingerprint, expected_data_split_fingerprint: (
            restore_calls.append(path) or resumed_checkpoint
        ),
    )

    with pytest.raises(ValueError, match="episode order"):
        trainer.train_episodes_resumable(
            (episodes[1], episodes[0], *episodes[2:]),
            input_builder=object(),
            optimizer=object(),
            dataset_fingerprint="cache-fingerprint",
            data_split_fingerprint="e" * 64,
            checkpoint_directory=tmp_path,
            checkpoint_every=2,
            resume_from=resumed_checkpoint.path,
        )
    assert restore_calls == []

    result = trainer.train_episodes_resumable(
        episodes,
        input_builder=object(),
        optimizer=object(),
        dataset_fingerprint="cache-fingerprint",
        data_split_fingerprint="e" * 64,
        checkpoint_directory=tmp_path,
        checkpoint_every=2,
        resume_from=resumed_checkpoint.path,
    )

    assert trained_ids == ["episode-1", "episode-2", "episode-3", "episode-4"]
    assert restore_calls == [resumed_checkpoint.path]
    assert result.resumed_episode_count == 2
    assert result.processed_episode_count == 2
    assert result.pass_result.episode_count == 2
    assert result.pass_result.optimizer_steps == 4
    assert result.last_checkpoint.progress == TrainingRunProgress(
        dataset_fingerprint="cache-fingerprint",
        completed_episode_count=4,
        completed_episode_prefix_sha256=_prefix_digest(
            ("episode-1", "episode-2", "episode-3", "episode-4")
        ),
        optimizer_steps=8,
        data_split_fingerprint="e" * 64,
    )

