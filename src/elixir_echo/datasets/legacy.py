"""Read-only compatibility with sealed SemanticFrame full-episode caches."""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator

from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation
from elixir_echo.domain.values import content_sha256
from .trajectory import (
    TrainingEpisode,
    TrajectoryEpisode,
    TrajectoryManifest,
    aggregate_record_hash,
    project_training_episode,
)


LEGACY_EPISODE_ENV = "ELIXIR_ECHO_LEGACY_EPISODE"
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class LegacyTrainingCatalog:
    """Manifest-only inventory plus the capped eligible training prefix."""

    selected_episode_ids: tuple[str, ...]
    split_counts: dict[str, int]
    outcome_counts: dict[str, int]
    split_outcome_counts: dict[str, int]
    assigned_without_episode_manifest_count: int

    @property
    def excluded_split_episode_count(self) -> int:
        return sum(
            count for key, count in self.split_outcome_counts.items()
            if not key.startswith("train/")
        )

    @property
    def excluded_truncated_episode_count(self) -> int:
        return self.split_outcome_counts.get("train/truncated", 0)

    @property
    def episode_count(self) -> int:
        return sum(self.split_counts.values())


def _legacy_step(payload: Mapping[str, Any]) -> DecisionStep:
    if payload.get("version") != "semantic-frame-cache-record.v1":
        raise ValueError("unsupported legacy cache record version")
    identity_payload = payload["identity"]
    identity = DecisionIdentity(
        episode_id=str(identity_payload["episode_id"]),
        tick=int(identity_payload["tick"]),
        actor=int(identity_payload["actor"]),
    )
    frame = payload["frame"]
    if frame.get("version") != "semantic-frame.v1":
        raise ValueError("unsupported legacy SemanticFrame version")
    observation = Observation(
        match=frame["match"],
        players=tuple(frame["players"]),
        towers=tuple(frame["towers"]),
        entities=tuple(frame["entities"]),
        terminal=frame["terminal"],
        perspective=str(frame.get("perspective") or "actor_fair"),
    )
    raw_mask = payload["action_mask"]
    if raw_mask.get("identity") != identity.to_dict():
        raise ValueError("legacy action mask identity mismatch")
    constraints = {key: value for key, value in raw_mask.items() if key not in {"schema", "identity"}}
    return DecisionStep(
        identity=identity,
        episode_context_ref=str(payload["episode_context_ref"]),
        observation=observation,
        action_mask=ActionMask(identity, constraints),
        expert_action=ExpertAction.from_legacy(payload.get("expert_action")),
        event_delta=tuple(payload.get("event_delta", ())),
        validation=payload["validation"],
        fingerprints=payload["fingerprints"],
    )


class LegacyTrajectoryReader:
    """Verify then map one legacy episode; never writes or repairs legacy files."""

    def __init__(self, episode_path: str | Path | None = None) -> None:
        raw = episode_path if episode_path is not None else os.environ.get(LEGACY_EPISODE_ENV)
        if raw is None:
            raise ValueError(f"set {LEGACY_EPISODE_ENV} or provide an episode path")
        self.episode_path = Path(raw).expanduser().resolve()

    def read_episode(self) -> TrajectoryEpisode:
        path = self.episode_path
        manifest_payload = json.loads((path / "episode_manifest.json").read_text(encoding="utf-8"))
        stored_manifest_hash = manifest_payload.pop("content_sha256", None)
        if not isinstance(stored_manifest_hash, str) or content_sha256(manifest_payload) != stored_manifest_hash:
            raise ValueError("legacy episode manifest hash mismatch")
        if manifest_payload.get("version") != "full-episode-manifest.v1":
            raise ValueError("unsupported legacy full-episode manifest")
        episode_id = str(manifest_payload["episode_id"])
        context_ref = str(manifest_payload["episode_context_ref"])
        if manifest_payload.get("outcome") not in {"terminated", "truncated"}:
            raise ValueError("legacy episode outcome is not terminated or truncated")
        if int(manifest_payload.get("decision_interval_ticks", -1)) != 5:
            raise ValueError("legacy episode decision interval is not five ticks")
        manifest_dir = path.parents[2]
        split_path = manifest_dir / "split_manifest.json"
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        if content_sha256(split_payload) != str(manifest_payload["split_manifest_hash"]):
            raise ValueError("legacy split manifest hash mismatch")
        assignment = split_payload.get("assignments", {}).get(episode_id)
        if assignment != manifest_payload.get("split"):
            raise ValueError("legacy episode split assignment mismatch")
        context_payload = json.loads((path / "episode_context.json").read_text(encoding="utf-8"))
        stored_context_hash = context_payload.pop("content_sha256", None)
        if stored_context_hash != context_ref or content_sha256(context_payload) != context_ref:
            raise ValueError("legacy episode context hash mismatch")
        if context_payload.get("episode_id") != episode_id:
            raise ValueError("legacy episode context identity mismatch")
        if len(context_payload.get("deck0", ())) != 8 or len(context_payload.get("deck1", ())) != 8:
            raise ValueError("legacy episode context must contain both complete decks")
        records: list[DecisionStep] = []
        hashes: list[str] = []
        with (path / "records.jsonl").open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                payload = json.loads(line)
                record_hash = payload.pop("content_sha256", None)
                if not isinstance(record_hash, str) or content_sha256(payload) != record_hash:
                    raise ValueError(f"legacy record hash mismatch at line {line_number}")
                step = _legacy_step(payload)
                if step.identity.episode_id != episode_id or step.episode_context_ref != context_ref:
                    raise ValueError("legacy record identity/context mismatch")
                records.append(step)
                hashes.append(record_hash)
        if tuple(hashes) != tuple(manifest_payload["record_hashes"]):
            raise ValueError("legacy record hash sequence mismatch")
        if aggregate_record_hash(episode_id, hashes) != manifest_payload["aggregate_hash"]:
            raise ValueError("legacy aggregate hash mismatch")
        keys = [(step.identity.tick, step.identity.actor) for step in records]
        if not keys or keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("legacy records are not in strict (tick, actor) order")
        expected_count = int(manifest_payload["record_count"])
        if len(records) != expected_count:
            raise ValueError("legacy record count mismatch")
        metadata = {
            "episode_id": episode_id,
            "replay_tag": str(manifest_payload["replay_tag"]),
            "outcome": str(manifest_payload["outcome"]),
            "split": str(manifest_payload["split"]),
            "episode_context_ref": context_ref,
            "source_manifest_sha256": stored_manifest_hash,
            "source_record_hashes": tuple(hashes),
        }
        ticks = tuple(sorted({step.identity.tick for step in records}))
        if not ticks or ticks[0] != int(manifest_payload["first_tick"]) or ticks[-1] != int(manifest_payload["last_tick"]):
            raise ValueError("legacy episode tick bounds mismatch")
        if ticks[0] != 90 or any(right - left != 5 for left, right in zip(ticks, ticks[1:])):
            raise ValueError("legacy episode timeline is not continuous from tick 90")
        if any([step.identity.actor for step in records if step.identity.tick == tick] != [0, 1] for tick in ticks):
            raise ValueError("legacy episode is missing an actor record at a decision tick")
        sequences = tuple(manifest_payload.get("sequences", ()))
        if tuple(item.get("actor") for item in sequences) != (0, 1):
            raise ValueError("legacy episode does not define actor 0/1 reset sequences")
        if any(item.get("episode_start_ordinal") != 0 or item.get("length") != len(ticks) for item in sequences):
            raise ValueError("legacy actor sequence reset/length boundary mismatch")
        new_manifest = TrajectoryManifest(
            episode_id=episode_id,
            replay_tag=metadata["replay_tag"],
            outcome=metadata["outcome"],
            first_tick=int(manifest_payload["first_tick"]),
            last_tick=int(manifest_payload["last_tick"]),
            record_count=expected_count,
            record_hashes=tuple(step.content_sha256() for step in records),
            aggregate_hash=aggregate_record_hash(
                episode_id, tuple(step.content_sha256() for step in records)
            ),
            metadata_sha256=content_sha256(metadata),
            split=str(manifest_payload["split"]),
        )
        return TrajectoryEpisode(new_manifest, tuple(records), metadata)


class LegacyTrainingTrajectoryReader:
    """Expose a verified legacy episode through the deck-free training view."""

    def __init__(self, episode_path: str | Path | None = None) -> None:
        self._reader = LegacyTrajectoryReader(episode_path)

    def read_episode(self) -> TrainingEpisode:
        return project_training_episode(self._reader.read_episode())


class LegacyTrainingTrajectoryStoreReader:
    """Read a sealed legacy episode store through the training-safe projection.

    Cataloging validates every episode manifest and the shared split manifest,
    then fully reads only the capped episodes that will actually train. Each
    selected episode still gets record, context, ordering, and aggregate-hash
    verification through :class:`LegacyTrainingTrajectoryReader`.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._paths_by_id: dict[str, Path] = {}

    def training_catalog(self, *, max_episodes: int) -> LegacyTrainingCatalog:
        if type(max_episodes) is not int or max_episodes <= 0:
            raise ValueError("max_episodes must be a positive integer")
        split_manifest_path = self.root / "split_manifest.json"
        split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(split_payload, dict) or not isinstance(split_payload.get("assignments"), dict):
            raise ValueError("legacy split manifest is invalid")
        split_hash = content_sha256(split_payload)
        split_counts: Counter[str] = Counter()
        outcome_counts: Counter[str] = Counter()
        split_outcome_counts: Counter[str] = Counter()
        selected: list[str] = []
        seen: set[str] = set()
        manifest_paths = sorted((self.root / "episodes").glob("*/*/episode_manifest.json"))
        if not manifest_paths:
            raise ValueError(f"legacy trajectory store has no episode manifests: {self.root}")

        for manifest_path in manifest_paths:
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"legacy episode manifest is unreadable: {manifest_path}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"legacy episode manifest is invalid: {manifest_path}")
            stored_hash = payload.pop("content_sha256", None)
            if not isinstance(stored_hash, str) or content_sha256(payload) != stored_hash:
                raise ValueError(f"legacy episode manifest hash mismatch: {manifest_path}")
            episode_id = payload.get("episode_id")
            split = payload.get("split")
            outcome = payload.get("outcome")
            if (
                payload.get("version") != "full-episode-manifest.v1"
                or not isinstance(episode_id, str)
                or not _SAFE_EPISODE_ID.fullmatch(episode_id)
                or split not in {"train", "validation", "test"}
                or outcome not in {"terminated", "truncated"}
            ):
                raise ValueError(f"legacy episode manifest contract is invalid: {manifest_path}")
            if episode_id != manifest_path.parent.name or episode_id in seen:
                raise ValueError(f"legacy episode identity is invalid or duplicated: {manifest_path}")
            if payload.get("split_manifest_hash") != split_hash:
                raise ValueError(f"legacy split manifest fingerprint mismatch: {manifest_path}")
            if split_payload["assignments"].get(episode_id) != split:
                raise ValueError(f"legacy split assignment mismatch: {manifest_path}")
            seen.add(episode_id)
            self._paths_by_id[episode_id] = manifest_path.parent
            split_counts[split] += 1
            outcome_counts[outcome] += 1
            split_outcome_counts[f"{split}/{outcome}"] += 1
            if split == "train" and outcome == "terminated" and len(selected) < max_episodes:
                selected.append(episode_id)

        if any(
            not isinstance(episode_id, str)
            or not _SAFE_EPISODE_ID.fullmatch(episode_id)
            or split not in {"train", "validation", "test"}
            for episode_id, split in split_payload["assignments"].items()
        ):
            raise ValueError("legacy split manifest contains invalid episode assignments")
        return LegacyTrainingCatalog(
            selected_episode_ids=tuple(selected),
            split_counts=dict(sorted(split_counts.items())),
            outcome_counts=dict(sorted(outcome_counts.items())),
            split_outcome_counts=dict(sorted(split_outcome_counts.items())),
            assigned_without_episode_manifest_count=(
                len(split_payload["assignments"]) - len(seen)
            ),
        )

    def read_episode(self, episode_id: str) -> TrainingEpisode:
        if not _SAFE_EPISODE_ID.fullmatch(episode_id):
            raise ValueError("legacy episode identity is invalid")
        path = self._paths_by_id.get(episode_id)
        if path is None:
            matches = tuple((self.root / "episodes").glob(f"*/{episode_id}"))
            if len(matches) != 1:
                raise ValueError(f"legacy episode identity is missing or ambiguous: {episode_id}")
            path = matches[0]
        return LegacyTrainingTrajectoryReader(path).read_episode()

    def iter_episodes(self) -> Iterator[TrainingEpisode]:
        for manifest_path in sorted((self.root / "episodes").glob("*/*/episode_manifest.json")):
            yield LegacyTrainingTrajectoryReader(manifest_path.parent).read_episode()
