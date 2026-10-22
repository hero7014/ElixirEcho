"""Safe planning and explicit execution for recurrent FirstLight training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from elixir_echo.datasets.trajectory import TrainingEpisode, TrainingTrajectoryReader
from elixir_echo.domain.values import content_sha256
from elixir_echo.training.config import TrainingConfig
from elixir_echo.training.labels import classify_expert_action


_FINGERPRINT_SCHEMA = b"elixir_echo.training-dataset@1\0"


@dataclass(frozen=True, slots=True)
class TrainingDatasetPlan:
    """A deterministic, safe projection of a capped train-split cache."""

    fingerprint: str
    selected_episode_ids: tuple[str, ...]
    episode_content_sha256: tuple[str, ...]
    selected_decision_count: int
    selected_label_counts: dict[str, int]
    excluded_split_episode_count: int
    excluded_truncated_episode_count: int

    @property
    def selected_episode_count(self) -> int:
        return len(self.selected_episode_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.fingerprint,
            "selected_episode_count": self.selected_episode_count,
            "selected_decision_count": self.selected_decision_count,
            "selected_label_counts": dict(sorted(self.selected_label_counts.items())),
            "excluded_split_episode_count": self.excluded_split_episode_count,
            "excluded_truncated_episode_count": self.excluded_truncated_episode_count,
            "selection": "first max_episodes terminated episodes in train split",
        }


def _safe_episode_digest(episode: TrainingEpisode) -> str:
    # Hash only the TrainingEpisode projection. Provider metadata and episode
    # context (including either player's deck) never enter this fingerprint.
    return content_sha256(episode.to_dict())


def plan_training_dataset(
    reader: TrainingTrajectoryReader,
    *,
    max_episodes: int,
) -> TrainingDatasetPlan:
    """Verify and fingerprint a deterministic prefix of eligible episodes."""

    if type(max_episodes) is not int or max_episodes <= 0:
        raise ValueError("max_episodes must be a positive integer")
    selected_ids: list[str] = []
    selected_hashes: list[str] = []
    labels: Counter[str] = Counter()
    selected_decisions = excluded_split = excluded_truncated = 0
    digest = hashlib.sha256(_FINGERPRINT_SCHEMA)
    seen: set[str] = set()

    for episode in reader.iter_episodes():
        if episode.episode_id in seen:
            raise ValueError(f"duplicate episode identity in training cache: {episode.episode_id}")
        seen.add(episode.episode_id)
        if episode.split is None:
            raise ValueError(f"episode {episode.episode_id!r} has no frozen dataset split")
        if episode.split != "train":
            excluded_split += 1
            continue
        if episode.outcome == "truncated":
            excluded_truncated += 1
            continue
        if episode.outcome != "terminated":
            raise ValueError(f"episode {episode.episode_id!r} has unsupported outcome {episode.outcome!r}")
        if len(selected_ids) >= max_episodes:
            continue
        episode_digest = _safe_episode_digest(episode)
        selected_ids.append(episode.episode_id)
        selected_hashes.append(episode_digest)
        payload = bytes.fromhex(episode_digest)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        selected_decisions += len(episode.steps)
        labels.update(classify_expert_action(step.expert_action).value for step in episode.steps)

    return TrainingDatasetPlan(
        fingerprint=digest.hexdigest(),
        selected_episode_ids=tuple(selected_ids),
        episode_content_sha256=tuple(selected_hashes),
        selected_decision_count=selected_decisions,
        selected_label_counts=dict(sorted(labels.items())),
        excluded_split_episode_count=excluded_split,
        excluded_truncated_episode_count=excluded_truncated,
    )


def iter_verified_training_episodes(
    reader: TrainingTrajectoryReader,
    plan: TrainingDatasetPlan,
) -> Iterator[TrainingEpisode]:
    """Yield the planned episodes while detecting cache changes before use."""

    index = 0
    seen: set[str] = set()
    for episode in reader.iter_episodes():
        if episode.episode_id in seen:
            raise ValueError(f"duplicate episode identity in training cache: {episode.episode_id}")
        seen.add(episode.episode_id)
        if episode.split != "train" or episode.outcome != "terminated":
            continue
        if index >= plan.selected_episode_count:
            continue
        if episode.episode_id != plan.selected_episode_ids[index]:
            raise ValueError("training cache order changed after dataset fingerprinting")
        if _safe_episode_digest(episode) != plan.episode_content_sha256[index]:
            raise ValueError(f"training episode changed after dataset fingerprinting: {episode.episode_id}")
        index += 1
        yield episode
    if index != plan.selected_episode_count:
        raise ValueError("training cache ended before the planned episode prefix")


def run_firstlight_training(
    reader: TrainingTrajectoryReader,
    plan: TrainingDatasetPlan,
    *,
    firstlight_root: str | Path,
    checkpoint_directory: str | Path,
    checkpoint_every: int,
    device: str,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Execute only the explicit capped plan through the resumable V4 trainer."""

    if type(checkpoint_every) is not int or checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be a positive integer")
    if not plan.selected_episode_count:
        raise ValueError("training cache contains no terminated train-split episodes")
    # Keep torch and provider imports out of planning/verification workflows.
    from elixir_echo.integrations.firstlight.policy import FirstLightPolicyBackend
    from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
    from elixir_echo.integrations.firstlight.tensorizer import PolicyInputBuilder
    from elixir_echo.integrations.firstlight.trainer import FirstLightILTrainer

    runtime = FirstLightRuntime(firstlight_root)
    backend = FirstLightPolicyBackend(runtime, device=device)
    config = TrainingConfig()
    input_builder = PolicyInputBuilder(runtime, components=backend.components)
    trainer = FirstLightILTrainer(backend, config)
    result = trainer.train_episodes_resumable(
        iter_verified_training_episodes(reader, plan),
        input_builder=input_builder,
        dataset_fingerprint=plan.fingerprint,
        checkpoint_directory=checkpoint_directory,
        checkpoint_every=checkpoint_every,
        resume_from=resume_from,
    )
    return {
        "mode": "training_run",
        **plan.to_dict(),
        "profile": config.profile.value,
        "training_config_fingerprint": config.fingerprint(),
        "resumed_episode_count": result.resumed_episode_count,
        "processed_episode_count": result.processed_episode_count,
        "optimizer_steps": result.last_checkpoint.update_step,
        "last_checkpoint": str(result.last_checkpoint.path),
    }
