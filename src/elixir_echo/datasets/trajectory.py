"""Atomic storage for complete, ordered replay trajectories."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation
from elixir_echo.domain.values import canonical_json, content_sha256, jsonable


TRAJECTORY_MANIFEST_SCHEMA = "elixir_echo.trajectory-manifest@1"
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_OUTCOMES = frozenset({"terminated", "truncated"})
_SPLITS = frozenset({"train", "validation", "test"})


def aggregate_record_hash(episode_id: str, record_hashes: Sequence[str]) -> str:
    """Return a stable hash over an episode identity and ordered record hashes."""

    if not episode_id or not record_hashes or any(not item for item in record_hashes):
        raise ValueError("aggregate hash requires an episode and non-empty record hashes")
    return content_sha256({"episode_id": episode_id, "record_hashes": list(record_hashes)})


@dataclass(frozen=True, slots=True)
class TrajectoryManifest:
    episode_id: str
    replay_tag: str
    outcome: str
    first_tick: int
    last_tick: int
    record_count: int
    record_hashes: tuple[str, ...]
    aggregate_hash: str
    metadata_sha256: str
    split: str | None = None
    schema: str = TRAJECTORY_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRAJECTORY_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported trajectory manifest schema {self.schema!r}")
        if not _SAFE_EPISODE_ID.fullmatch(self.episode_id) or not self.replay_tag:
            raise ValueError("invalid trajectory identity")
        if self.outcome not in _OUTCOMES:
            raise ValueError("trajectory outcome must be terminated or truncated")
        if self.split is not None and self.split not in _SPLITS:
            raise ValueError("trajectory split must be train, validation, test, or None")
        if self.first_tick < 0 or self.last_tick < self.first_tick or self.record_count <= 0:
            raise ValueError("invalid trajectory tick bounds or record count")
        if self.record_count != len(self.record_hashes):
            raise ValueError("record count and record hash count differ")
        if self.aggregate_hash != aggregate_record_hash(self.episode_id, self.record_hashes):
            raise ValueError("trajectory aggregate hash mismatch")
        if not self.metadata_sha256:
            raise ValueError("trajectory metadata hash is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "replay_tag": self.replay_tag,
            "outcome": self.outcome,
            "first_tick": self.first_tick,
            "last_tick": self.last_tick,
            "record_count": self.record_count,
            "record_hashes": list(self.record_hashes),
            "aggregate_hash": self.aggregate_hash,
            "metadata_sha256": self.metadata_sha256,
            "split": self.split,
        }

    def content_sha256(self) -> str:
        return content_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryManifest":
        return cls(
            schema=str(payload["schema"]),
            episode_id=str(payload["episode_id"]),
            replay_tag=str(payload["replay_tag"]),
            outcome=str(payload["outcome"]),
            first_tick=int(payload["first_tick"]),
            last_tick=int(payload["last_tick"]),
            record_count=int(payload["record_count"]),
            record_hashes=tuple(str(item) for item in payload["record_hashes"]),
            aggregate_hash=str(payload["aggregate_hash"]),
            metadata_sha256=str(payload["metadata_sha256"]),
            split=str(payload["split"]) if payload.get("split") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryEpisode:
    manifest: TrajectoryManifest
    records: tuple[DecisionStep, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingStep:
    """The permitted training projection of a decision record."""

    identity: DecisionIdentity
    observation: Observation
    action_mask: ActionMask
    expert_action: ExpertAction
    event_delta: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "observation": self.observation.to_dict(),
            "action_mask": self.action_mask.to_dict(),
            "expert_action": self.expert_action.to_legacy(),
            "event_delta": jsonable(self.event_delta),
        }


@dataclass(frozen=True, slots=True)
class TrainingEpisode:
    episode_id: str
    outcome: str
    split: str | None
    steps: tuple[TrainingStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "outcome": self.outcome,
            "split": self.split,
            "steps": [step.to_dict() for step in self.steps],
        }


def project_training_episode(episode: TrajectoryEpisode) -> TrainingEpisode:
    """Drop episode/provider audit metadata at the training boundary."""

    return TrainingEpisode(
        episode_id=episode.manifest.episode_id,
        outcome=episode.manifest.outcome,
        split=episode.manifest.split,
        steps=tuple(
            TrainingStep(
                identity=step.identity,
                observation=step.observation,
                action_mask=step.action_mask,
                expert_action=step.expert_action,
                event_delta=step.event_delta,
            )
            for step in episode.records
        ),
    )


def _write_fsynced(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_episode_id(value: str) -> str:
    if not _SAFE_EPISODE_ID.fullmatch(value):
        raise ValueError("episode identity is not safe for a trajectory directory")
    return value


class TrajectoryWriter:
    """Build and atomically publish exactly one complete trajectory episode."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._staging: Path | None = None
        self._metadata: dict[str, Any] | None = None
        self._records: list[DecisionStep] = []
        self._last_key: tuple[int, int] | None = None
        self._episode_id: str | None = None

    def begin(self, metadata: Mapping[str, Any]) -> "TrajectoryWriter":
        if self._staging is not None:
            raise RuntimeError("trajectory writer has already begun")
        values = jsonable(metadata)
        if not isinstance(values, dict):
            raise TypeError("trajectory metadata must be a mapping")
        episode_id = _safe_episode_id(str(values.get("episode_id") or ""))
        if values.get("outcome") not in _OUTCOMES:
            raise ValueError("trajectory metadata requires terminated or truncated outcome")
        if not str(values.get("replay_tag") or ""):
            raise ValueError("trajectory metadata requires replay_tag")
        if values.get("split") is not None and values["split"] not in _SPLITS:
            raise ValueError("trajectory split must be train, validation, test, or None")
        if not str(values.get("episode_context_ref") or ""):
            raise ValueError("trajectory metadata requires an episode_context_ref")
        self._episode_id = episode_id
        self._metadata = values
        final = TrajectoryReader(self.root).episode_path(episode_id)
        if final.exists():
            raise FileExistsError(f"trajectory episode already exists: {final}")
        final.parent.mkdir(parents=True, exist_ok=True)
        self._staging = Path(tempfile.mkdtemp(prefix=f".staging-{episode_id}-", dir=final.parent))
        return self

    def append(self, step: DecisionStep) -> None:
        if self._staging is None or self._metadata is None or self._episode_id is None:
            raise RuntimeError("call begin(metadata) before append(step)")
        key = (step.identity.tick, step.identity.actor)
        if step.identity.episode_id != self._episode_id:
            self.abort()
            raise ValueError("decision step episode identity does not match trajectory")
        if self._last_key is not None and key <= self._last_key:
            message = "duplicate" if key == self._last_key else "strict (tick, actor) order violation"
            self.abort()
            raise ValueError(message)
        try:
            if step.action_mask.identity != step.identity:
                raise ValueError("action mask identity does not match decision identity")
            if step.validation.get("accepted", True) is not True:
                raise ValueError("trajectory cannot contain a rejected decision record")
            self._records.append(step)
            self._last_key = key
        except BaseException:
            self.abort()
            raise

    def commit(self) -> TrajectoryManifest:
        if self._staging is None or self._metadata is None or self._episode_id is None:
            raise RuntimeError("call begin(metadata) before commit()")
        if not self._records:
            self.abort()
            raise ValueError("cannot commit an empty trajectory")
        try:
            metadata_hash = content_sha256(self._metadata)
            metadata_payload = {"metadata": self._metadata, "content_sha256": metadata_hash}
            _write_fsynced(self._staging / "episode_metadata.json", canonical_json(metadata_payload) + "\n")
            record_hashes = tuple(record.content_sha256() for record in self._records)
            lines = [
                canonical_json({**record.to_dict(), "content_sha256": record_hash})
                for record, record_hash in zip(self._records, record_hashes, strict=True)
            ]
            _write_fsynced(self._staging / "records.jsonl", "\n".join(lines) + "\n")
            manifest = TrajectoryManifest(
                episode_id=self._episode_id,
                replay_tag=str(self._metadata["replay_tag"]),
                outcome=str(self._metadata["outcome"]),
                first_tick=self._records[0].identity.tick,
                last_tick=self._records[-1].identity.tick,
                record_count=len(self._records),
                record_hashes=record_hashes,
                aggregate_hash=aggregate_record_hash(self._episode_id, record_hashes),
                metadata_sha256=metadata_hash,
                split=self._metadata.get("split"),
            )
            manifest_payload = {**manifest.to_dict(), "content_sha256": manifest.content_sha256()}
            _write_fsynced(self._staging / "trajectory_manifest.json", canonical_json(manifest_payload) + "\n")
            # Read back every byte-level contract before the one visible rename.
            staged = TrajectoryReader(self.root).read_staged(self._staging, manifest)
            if len(staged.records) != len(self._records):
                raise ValueError("staged trajectory verification changed record count")
            final = TrajectoryReader(self.root).episode_path(self._episode_id)
            os.replace(self._staging, final)
            self._staging = None
            return manifest
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        staging = self._staging
        self._staging = None
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _decision_from_dict(payload: Mapping[str, Any]) -> DecisionStep:
    identity_payload = payload["identity"]
    identity = DecisionIdentity(
        episode_id=str(identity_payload["episode_id"]),
        tick=int(identity_payload["tick"]),
        actor=int(identity_payload["actor"]),
    )
    observation_payload = payload["observation"]
    observation = Observation(
        match=observation_payload["match"],
        players=tuple(observation_payload["players"]),
        towers=tuple(observation_payload["towers"]),
        entities=tuple(observation_payload["entities"]),
        terminal=observation_payload["terminal"],
        perspective=str(observation_payload.get("perspective", "actor_fair")),
    )
    raw_mask = payload["action_mask"]
    mask_identity = raw_mask.get("identity")
    if mask_identity != identity.to_dict():
        raise ValueError("stored action mask identity mismatch")
    mask_constraints = {key: value for key, value in raw_mask.items() if key not in {"identity", "schema"}}
    return DecisionStep(
        identity=identity,
        episode_context_ref=str(payload["episode_context_ref"]),
        observation=observation,
        action_mask=ActionMask(identity, mask_constraints),
        expert_action=ExpertAction.from_legacy(payload.get("expert_action")),
        event_delta=tuple(payload.get("event_delta", ())),
        validation=payload.get("validation", {}),
        fingerprints=payload.get("fingerprints", {}),
    )


class TrajectoryReader:
    """Read only atomically committed and hash-verified episodes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def episode_path(self, episode_id: str) -> Path:
        safe = _safe_episode_id(episode_id)
        shard = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:2]
        return self.root / "episodes" / shard / safe

    def iter_episodes(self) -> Iterator[TrajectoryEpisode]:
        for manifest_path in sorted((self.root / "episodes").glob("*/*/trajectory_manifest.json")):
            yield self.read_episode(manifest_path.parent.name)

    def read_episode(self, episode_id: str) -> TrajectoryEpisode:
        path = self.episode_path(episode_id)
        manifest_path = path / "trajectory_manifest.json"
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stored_hash = raw_manifest.pop("content_sha256", None)
            if not isinstance(stored_hash, str) or content_sha256(raw_manifest) != stored_hash:
                raise ValueError("trajectory manifest hash mismatch")
            manifest = TrajectoryManifest.from_dict(raw_manifest)
            if manifest.episode_id != episode_id:
                raise ValueError("manifest episode identity mismatch")
            return self.read_staged(path, manifest)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid trajectory episode {path}: {error}") from error

    def read_staged(self, path: Path, manifest: TrajectoryManifest) -> TrajectoryEpisode:
        metadata_payload = json.loads((path / "episode_metadata.json").read_text(encoding="utf-8"))
        metadata = metadata_payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("episode metadata payload is invalid")
        if metadata_payload.get("content_sha256") != manifest.metadata_sha256:
            raise ValueError("episode metadata hash mismatch")
        if content_sha256(metadata) != manifest.metadata_sha256:
            raise ValueError("episode metadata content hash mismatch")
        if (
            metadata.get("episode_id") != manifest.episode_id
            or metadata.get("outcome") != manifest.outcome
            or metadata.get("split") != manifest.split
        ):
            raise ValueError("episode metadata identity/outcome mismatch")
        records: list[DecisionStep] = []
        hashes: list[str] = []
        with (path / "records.jsonl").open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                stored_hash = payload.pop("content_sha256", None)
                if not isinstance(stored_hash, str) or content_sha256(payload) != stored_hash:
                    raise ValueError(f"record hash mismatch at {path}:records.jsonl:{line_number}")
                step = _decision_from_dict(payload)
                if step.identity.episode_id != manifest.episode_id:
                    raise ValueError("record episode identity mismatch")
                if step.episode_context_ref != metadata.get("episode_context_ref"):
                    raise ValueError("record episode-context reference mismatch")
                records.append(step)
                hashes.append(stored_hash)
        if tuple(hashes) != manifest.record_hashes:
            raise ValueError("record hash order differs from manifest")
        if aggregate_record_hash(manifest.episode_id, hashes) != manifest.aggregate_hash:
            raise ValueError("trajectory aggregate hash mismatch")
        if len(records) != manifest.record_count:
            raise ValueError("trajectory record count mismatch")
        keys = [(item.identity.tick, item.identity.actor) for item in records]
        if not keys or keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("records are not in strict (tick, actor) order")
        if keys[0][0] != manifest.first_tick or keys[-1][0] != manifest.last_tick:
            raise ValueError("trajectory manifest tick bounds mismatch")
        grouped: dict[int, list[int]] = {}
        for tick, actor in keys:
            grouped.setdefault(tick, []).append(actor)
        ticks = tuple(grouped)
        if ticks[0] != 90 or any(grouped[tick] != [0, 1] for tick in ticks):
            raise ValueError("full trajectory requires actor 0 and actor 1 at every tick from tick 90")
        if any(right - left != 5 for left, right in zip(ticks, ticks[1:])):
            raise ValueError("full trajectory ticks must be continuous five-tick intervals")
        return TrajectoryEpisode(manifest, tuple(records), metadata)


class TrainingTrajectoryReader:
    """Expose only current observations, legal actions, public deltas and labels."""

    def __init__(self, root: str | Path) -> None:
        self._reader = TrajectoryReader(root)

    def iter_episodes(self) -> Iterator[TrainingEpisode]:
        for episode in self._reader.iter_episodes():
            yield project_training_episode(episode)

    def read_episode(self, episode_id: str) -> TrainingEpisode:
        return project_training_episode(self._reader.read_episode(episode_id))

    @staticmethod
    def _project(episode: TrajectoryEpisode) -> TrainingEpisode:
        return project_training_episode(episode)
