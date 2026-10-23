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
_SPLIT_FINGERPRINT_SCHEMA = "elixir_echo.episode-split-assignment@1"


@dataclass(frozen=True, slots=True)
class TrainingDatasetPlan:
    """A deterministic, safe projection of a capped train-split cache."""

    fingerprint: str
    data_split_fingerprint: str
    selected_episode_ids: tuple[str, ...]
    episode_content_sha256: tuple[str, ...]
    selected_decision_count: int
    selected_label_counts: dict[str, int]
    excluded_split_episode_count: int
    excluded_truncated_episode_count: int
    source_episode_count: int
    source_split_counts: dict[str, int]
    source_outcome_counts: dict[str, int]
    source_split_outcome_counts: dict[str, int]
    assigned_without_episode_manifest_count: int

    @property
    def selected_episode_count(self) -> int:
        return len(self.selected_episode_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.fingerprint,
            "data_split_fingerprint": self.data_split_fingerprint,
            "selected_episode_count": self.selected_episode_count,
            "selected_decision_count": self.selected_decision_count,
            "selected_label_counts": dict(sorted(self.selected_label_counts.items())),
            "excluded_split_episode_count": self.excluded_split_episode_count,
            "excluded_truncated_episode_count": self.excluded_truncated_episode_count,
            "source_episode_count": self.source_episode_count,
            "source_split_counts": dict(sorted(self.source_split_counts.items())),
            "source_outcome_counts": dict(sorted(self.source_outcome_counts.items())),
            "source_split_outcome_counts": dict(sorted(self.source_split_outcome_counts.items())),
            "assigned_without_episode_manifest_count": self.assigned_without_episode_manifest_count,
            "selection": "first max_episodes terminated episodes in train split",
        }


def _safe_episode_digest(episode: TrainingEpisode) -> str:
    # Hash only the TrainingEpisode projection. Provider metadata and episode
    # context (including either player's deck) never enter this fingerprint.
    return content_sha256(episode.to_dict())


def _split_assignment_fingerprint(assignments: dict[str, str]) -> str:
    return content_sha256(
        {
            "schema": _SPLIT_FINGERPRINT_SCHEMA,
            "assignments": [
                {"episode_id": episode_id, "split": split}
                for episode_id, split in sorted(assignments.items())
            ],
        }
    )


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
    source_split_counts: Counter[str] = Counter()
    source_outcome_counts: Counter[str] = Counter()
    source_split_outcome_counts: Counter[str] = Counter()
    split_assignments: dict[str, str] = {}
    source_episode_count = 0
    assigned_without_episode_manifest_count = 0

    def include_episode(episode: TrainingEpisode) -> None:
        nonlocal selected_decisions
        if episode.split != "train" or episode.outcome != "terminated":
            raise ValueError("dataset planner selected an ineligible training episode")
        episode_digest = _safe_episode_digest(episode)
        selected_ids.append(episode.episode_id)
        selected_hashes.append(episode_digest)
        payload = bytes.fromhex(episode_digest)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        selected_decisions += len(episode.steps)
        labels.update(classify_expert_action(step.expert_action).value for step in episode.steps)

    catalog_factory = getattr(reader, "training_catalog", None)
    if callable(catalog_factory):
        catalog = catalog_factory(max_episodes=max_episodes)
        catalog_assignments = getattr(catalog, "split_assignments", None)
        if not isinstance(catalog_assignments, dict):
            raise ValueError("training catalog must expose the complete frozen split assignments")
        split_assignments.update(catalog_assignments)
        for episode_id in catalog.selected_episode_ids:
            include_episode(reader.read_episode(episode_id))
        source_split_counts.update(catalog.split_counts)
        source_outcome_counts.update(catalog.outcome_counts)
        source_split_outcome_counts.update(catalog.split_outcome_counts)
        source_episode_count = catalog.episode_count
        excluded_split = catalog.excluded_split_episode_count
        excluded_truncated = catalog.excluded_truncated_episode_count
        assigned_without_episode_manifest_count = catalog.assigned_without_episode_manifest_count
    else:
        seen: set[str] = set()
        for episode in reader.iter_episodes():
            source_episode_count += 1
            source_split_counts[episode.split or "unknown"] += 1
            source_outcome_counts[episode.outcome] += 1
            source_split_outcome_counts[f"{episode.split or 'unknown'}/{episode.outcome}"] += 1
            if episode.episode_id in seen:
                raise ValueError(f"duplicate episode identity in training cache: {episode.episode_id}")
            seen.add(episode.episode_id)
            if episode.split is None:
                raise ValueError(f"episode {episode.episode_id!r} has no frozen dataset split")
            split_assignments[episode.episode_id] = episode.split
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
            include_episode(episode)

    return TrainingDatasetPlan(
        fingerprint=digest.hexdigest(),
        data_split_fingerprint=_split_assignment_fingerprint(split_assignments),
        selected_episode_ids=tuple(selected_ids),
        episode_content_sha256=tuple(selected_hashes),
        selected_decision_count=selected_decisions,
        selected_label_counts=dict(sorted(labels.items())),
        excluded_split_episode_count=excluded_split,
        excluded_truncated_episode_count=excluded_truncated,
        source_episode_count=source_episode_count,
        source_split_counts=dict(sorted(source_split_counts.items())),
        source_outcome_counts=dict(sorted(source_outcome_counts.items())),
        source_split_outcome_counts=dict(sorted(source_split_outcome_counts.items())),
        assigned_without_episode_manifest_count=assigned_without_episode_manifest_count,
    )


def iter_verified_training_episodes(
    reader: TrainingTrajectoryReader,
    plan: TrainingDatasetPlan,
) -> Iterator[TrainingEpisode]:
    """Yield the planned episodes while detecting cache changes before use."""

    for episode_id, expected_digest in zip(
        plan.selected_episode_ids,
        plan.episode_content_sha256,
        strict=True,
    ):
        episode = reader.read_episode(episode_id)
        if episode.episode_id != episode_id:
            raise ValueError("training cache order changed after dataset fingerprinting")
        if _safe_episode_digest(episode) != expected_digest:
            raise ValueError(f"training episode changed after dataset fingerprinting: {episode.episode_id}")
        yield episode


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
    input_builder = PolicyInputBuilder(
        runtime,
        components=backend.components,
        legacy_checkpoint_input=config.input_mode == "legacy_checkpoint",
    )
    trainer = FirstLightILTrainer(backend, config)
    result = trainer.train_episodes_resumable(
        iter_verified_training_episodes(reader, plan),
        input_builder=input_builder,
        dataset_fingerprint=plan.fingerprint,
        data_split_fingerprint=plan.data_split_fingerprint,
        checkpoint_directory=checkpoint_directory,
        checkpoint_every=checkpoint_every,
        resume_from=resume_from,
    )
    return {
        "mode": "training_run",
        **plan.to_dict(),
        "profile": config.profile.value,
        "input_schema": config.to_dict()["input_schema"],
        "input_mode": config.input_mode,
        "training_config_fingerprint": config.fingerprint(),
        "resumed_episode_count": result.resumed_episode_count,
        "processed_episode_count": result.processed_episode_count,
        "optimizer_steps": result.last_checkpoint.update_step,
        "last_checkpoint": str(result.last_checkpoint.path),
    }
