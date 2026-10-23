"""Deterministic, episode-isolated views over compiled actor sequences."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elixir_echo.datasets.trajectory import (
    TrajectoryEpisode,
    aggregate_record_hash,
    project_training_episode,
)
from elixir_echo.domain.values import canonical_json, content_sha256
from elixir_echo.profiles import ObservationProfile

if TYPE_CHECKING:
    from elixir_echo.training.sequence import CompiledActorSequence


SequenceFactory = Callable[[], Iterable["CompiledActorSequence"]]
ActorSequenceCompiler = Callable[[Any, int], "CompiledActorSequence"]
ActorSequenceEncoder = Callable[["CompiledActorSequence"], bytes]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_LAYER2_SCHEMA = "elixir_echo.layer2-sequence-episode@1"
_SPLITS = frozenset({"train", "validation", "test"})


class SequenceCacheError(ValueError):
    """Raised when a committed Layer 2 sequence cache violates its contract."""


@dataclass(frozen=True, slots=True)
class CachedSequenceEpisode:
    """Verified, model-ready actor payloads for one complete replay episode."""

    episode_id: str
    split: str
    outcome: str
    profile: str
    source_episode_sha256: str
    dataset_fingerprint: str
    training_config_fingerprint: str
    input_contract_fingerprint: str
    actor_payloads: tuple[bytes, bytes]
    ticks_by_actor: tuple[tuple[int, ...], tuple[int, ...]]
    label_counts_by_actor: tuple[Mapping[str, int], Mapping[str, int]]
    manifest_sha256: str


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_actor_keys(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and all(type(key) is int for key in value)
        and set(value) == {0, 1}
    )


def _validate_sequence_cache_payloads(actor_payloads: Mapping[int, bytes]) -> None:
    if not _has_actor_keys(actor_payloads):
        raise ValueError("sequence-cache episode requires payloads for actors 0 and 1")
    for actor in (0, 1):
        if not isinstance(actor_payloads[actor], bytes) or not actor_payloads[actor]:
            raise ValueError("sequence-cache actor payloads must be non-empty bytes")


def _validate_sequence_cache_inputs(
    *,
    episode_id: str,
    split: str,
    outcome: str,
    source_episode_sha256: str,
    dataset_fingerprint: str,
    training_config_fingerprint: str,
    input_contract_fingerprint: str,
    profile: str,
    ticks_by_actor: Mapping[int, Iterable[int]],
    label_counts_by_actor: Mapping[int, Mapping[str, int]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[dict[str, int], dict[str, int]]]:
    if not isinstance(episode_id, str) or not _SAFE_EPISODE_ID.fullmatch(episode_id):
        raise ValueError("invalid sequence-cache episode identity")
    if split not in _SPLITS:
        raise ValueError("sequence-cache split must be train, validation, or test")
    if outcome != "terminated":
        raise ValueError("Layer 2 training cache accepts terminated episodes only")
    for name, value in (
        ("source episode", source_episode_sha256),
        ("dataset", dataset_fingerprint),
        ("training config", training_config_fingerprint),
        ("input contract", input_contract_fingerprint),
    ):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{name} fingerprint must be a lowercase SHA-256 digest")
    if not isinstance(profile, str) or profile not in {"core", "full"}:
        raise ValueError("sequence-cache profile must be core/MIN or full/RICH")
    if not _has_actor_keys(ticks_by_actor):
        raise ValueError("sequence-cache episode requires ticks for actors 0 and 1")
    if not _has_actor_keys(label_counts_by_actor):
        raise ValueError("sequence-cache episode requires label counts for actors 0 and 1")
    ticks: list[tuple[int, ...]] = []
    counts: list[dict[str, int]] = []
    for actor in (0, 1):
        actor_ticks = tuple(ticks_by_actor[actor])
        if (
            not actor_ticks
            or any(type(tick) is not int or tick < 0 for tick in actor_ticks)
            or actor_ticks[0] != 90
            or any(right - left != 5 for left, right in zip(actor_ticks, actor_ticks[1:]))
        ):
            raise ValueError("sequence-cache actor ticks must start at 90 and advance by five")
        ticks.append(actor_ticks)
        actor_counts = label_counts_by_actor[actor]
        if not isinstance(actor_counts, Mapping) or any(
            not isinstance(name, str) or type(count) is not int or count < 0
            for name, count in actor_counts.items()
        ):
            raise ValueError("sequence-cache label counts must map names to non-negative integers")
        normalized_counts = {name: actor_counts[name] for name in sorted(actor_counts)}
        if sum(normalized_counts.values()) != len(actor_ticks):
            raise ValueError("sequence-cache label counts must account for every decision")
        counts.append(normalized_counts)
    if ticks[0] != ticks[1]:
        raise ValueError("actor sequences must contain the same decision ticks")
    return (ticks[0], ticks[1]), (counts[0], counts[1])


class Layer2SequenceCache:
    """Immutable, checksummed Layer 2 replay cache with episode-atomic commits.

    The cache stores opaque model-ready actor payloads and their public metadata;
    provider-specific tensor serialization belongs in the integration layer.
    A complete episode directory is made visible by one same-volume rename.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.episodes_root = self.root / "episodes"

    def _episode_dir(self, episode_id: str) -> Path:
        if not isinstance(episode_id, str) or not _SAFE_EPISODE_ID.fullmatch(episode_id):
            raise ValueError("invalid sequence-cache episode identity")
        return self.episodes_root / episode_id

    def commit_episode(
        self,
        *,
        episode_id: str,
        split: str,
        outcome: str,
        source_episode_sha256: str,
        dataset_fingerprint: str,
        training_config_fingerprint: str,
        input_contract_fingerprint: str,
        profile: str,
        actor_payloads: Mapping[int, bytes],
        ticks_by_actor: Mapping[int, Iterable[int]],
        label_counts_by_actor: Mapping[int, Mapping[str, int]],
    ) -> dict[str, Any]:
        _validate_sequence_cache_payloads(actor_payloads)
        ticks, counts = _validate_sequence_cache_inputs(
            episode_id=episode_id,
            split=split,
            outcome=outcome,
            source_episode_sha256=source_episode_sha256,
            dataset_fingerprint=dataset_fingerprint,
            training_config_fingerprint=training_config_fingerprint,
            input_contract_fingerprint=input_contract_fingerprint,
            profile=profile,
            ticks_by_actor=ticks_by_actor,
            label_counts_by_actor=label_counts_by_actor,
        )
        destination = self._episode_dir(episode_id)
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Layer 2 episode is already committed: {episode_id}")
        staging = Path(tempfile.mkdtemp(prefix=f".staging-{episode_id}-", dir=self.episodes_root))
        try:
            actors: dict[str, dict[str, Any]] = {}
            for actor in (0, 1):
                payload = actor_payloads[actor]
                name = f"actor-{actor}.bin"
                _write_fsynced(staging / name, payload)
                actors[str(actor)] = {
                    "file": name,
                    "ticks": list(ticks[actor]),
                    "frame_count": len(ticks[actor]),
                    "label_counts": counts[actor],
                    "byte_count": len(payload),
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                }
            manifest_body = {
                "schema": _LAYER2_SCHEMA,
                "episode_id": episode_id,
                "split": split,
                "outcome": outcome,
                "profile": profile,
                "source_episode_sha256": source_episode_sha256,
                "dataset_fingerprint": dataset_fingerprint,
                "training_config_fingerprint": training_config_fingerprint,
                "input_contract_fingerprint": input_contract_fingerprint,
                "actors": actors,
            }
            manifest_sha256 = content_sha256(manifest_body)
            manifest = {**manifest_body, "manifest_sha256": manifest_sha256}
            _write_fsynced(
                staging / "manifest.json",
                (canonical_json(manifest) + "\n").encode("utf-8"),
            )
            _write_fsynced(staging / "COMMITTED", (manifest_sha256 + "\n").encode("ascii"))
            os.replace(staging, destination)
            return manifest
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def _read_verified_manifest(
        self,
        episode_id: str,
        *,
        verify_payloads: bool,
    ) -> tuple[dict[str, Any], str]:
        episode_dir = self._episode_dir(episode_id)
        try:
            manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
            committed_hash = (episode_dir / "COMMITTED").read_text(encoding="ascii").strip()
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SequenceCacheError(f"Layer 2 episode {episode_id!r} is not committed/readable") from error
        if not isinstance(manifest, dict) or manifest.get("schema") != _LAYER2_SCHEMA:
            raise SequenceCacheError("unsupported Layer 2 sequence cache manifest")
        expected_fields = {
            "schema",
            "episode_id",
            "split",
            "outcome",
            "profile",
            "source_episode_sha256",
            "dataset_fingerprint",
            "training_config_fingerprint",
            "input_contract_fingerprint",
            "actors",
            "manifest_sha256",
        }
        if set(manifest) != expected_fields:
            raise SequenceCacheError("Layer 2 manifest fields do not match the frozen schema")
        claimed_hash = manifest.pop("manifest_sha256", None)
        actual_hash = content_sha256(manifest)
        manifest["manifest_sha256"] = claimed_hash
        if claimed_hash != actual_hash or committed_hash != actual_hash:
            raise SequenceCacheError("Layer 2 manifest/commit hash mismatch")
        if manifest.get("episode_id") != episode_id:
            raise SequenceCacheError("Layer 2 episode identity mismatch")
        actors = manifest.get("actors")
        if not isinstance(actors, dict) or set(actors) != {"0", "1"}:
            raise SequenceCacheError("Layer 2 manifest must contain exactly actors 0 and 1")
        for actor in (0, 1):
            entry = actors[str(actor)]
            if not isinstance(entry, dict) or set(entry) != {
                "file", "ticks", "frame_count", "label_counts", "byte_count", "payload_sha256"
            }:
                raise SequenceCacheError(f"Layer 2 actor {actor} descriptor fields are invalid")
            if (
                type(entry.get("frame_count")) is not int
                or entry["frame_count"] != len(entry.get("ticks", ()))
            ):
                raise SequenceCacheError(f"Layer 2 actor {actor} frame count differs from ticks")
        try:
            _validate_sequence_cache_inputs(
                episode_id=episode_id,
                split=manifest["split"],
                outcome=manifest["outcome"],
                source_episode_sha256=manifest["source_episode_sha256"],
                dataset_fingerprint=manifest["dataset_fingerprint"],
                training_config_fingerprint=manifest["training_config_fingerprint"],
                input_contract_fingerprint=manifest["input_contract_fingerprint"],
                profile=manifest["profile"],
                ticks_by_actor={
                    actor: actors[str(actor)]["ticks"] for actor in (0, 1)
                },
                label_counts_by_actor={
                    actor: actors[str(actor)]["label_counts"] for actor in (0, 1)
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SequenceCacheError(f"invalid Layer 2 episode manifest: {error}") from error
        for actor in (0, 1):
            entry = actors[str(actor)]
            name = f"actor-{actor}.bin"
            if not isinstance(entry, dict) or entry.get("file") != name:
                raise SequenceCacheError("Layer 2 manifest has an invalid actor payload path")
            if (
                type(entry.get("byte_count")) is not int
                or entry["byte_count"] <= 0
                or not isinstance(entry.get("payload_sha256"), str)
                or not _SHA256.fullmatch(entry["payload_sha256"])
            ):
                raise SequenceCacheError(f"Layer 2 actor {actor} payload identity is invalid")
            payload_path = episode_dir / name
            try:
                size = payload_path.stat().st_size
            except OSError as error:
                raise SequenceCacheError(f"Layer 2 actor {actor} payload is missing") from error
            if size != entry.get("byte_count"):
                raise SequenceCacheError(f"Layer 2 actor {actor} payload byte count mismatch")
            if verify_payloads and _sha256_file(payload_path) != entry.get("payload_sha256"):
                raise SequenceCacheError(f"Layer 2 actor {actor} payload SHA-256 mismatch")
        return manifest, actual_hash

    def read_episode(
        self,
        episode_id: str,
        *,
        expected_dataset_fingerprint: str | None = None,
        expected_training_config_fingerprint: str | None = None,
        expected_input_contract_fingerprint: str | None = None,
    ) -> CachedSequenceEpisode:
        manifest, manifest_sha256 = self._read_verified_manifest(episode_id, verify_payloads=False)
        expected = (
            ("dataset_fingerprint", expected_dataset_fingerprint),
            ("training_config_fingerprint", expected_training_config_fingerprint),
            ("input_contract_fingerprint", expected_input_contract_fingerprint),
        )
        for field, value in expected:
            if value is not None and manifest[field] != value:
                raise SequenceCacheError(f"Layer 2 {field} mismatch")
        episode_dir = self._episode_dir(episode_id)
        payloads = tuple((episode_dir / f"actor-{actor}.bin").read_bytes() for actor in (0, 1))
        for actor in (0, 1):
            entry = manifest["actors"][str(actor)]
            payload = payloads[actor]
            payload_sha256 = hashlib.sha256(payload).hexdigest()
            if len(payload) != entry["byte_count"] or payload_sha256 != entry["payload_sha256"]:
                raise SequenceCacheError(f"Layer 2 actor {actor} payload SHA-256 mismatch")
        ticks = tuple(tuple(manifest["actors"][str(actor)]["ticks"]) for actor in (0, 1))
        counts = tuple(manifest["actors"][str(actor)]["label_counts"] for actor in (0, 1))
        return CachedSequenceEpisode(
            episode_id=episode_id,
            split=manifest["split"],
            outcome=manifest["outcome"],
            profile=manifest["profile"],
            source_episode_sha256=manifest["source_episode_sha256"],
            dataset_fingerprint=manifest["dataset_fingerprint"],
            training_config_fingerprint=manifest["training_config_fingerprint"],
            input_contract_fingerprint=manifest["input_contract_fingerprint"],
            actor_payloads=(payloads[0], payloads[1]),
            ticks_by_actor=(ticks[0], ticks[1]),
            label_counts_by_actor=(counts[0], counts[1]),
            manifest_sha256=manifest_sha256,
        )

    def committed_episode_ids(self, *, split: str | None = None) -> tuple[str, ...]:
        if split is not None and split not in _SPLITS:
            raise ValueError("sequence-cache split must be train, validation, test, or None")
        if not self.episodes_root.exists():
            return ()
        episode_ids = []
        for candidate in sorted(self.episodes_root.iterdir(), key=lambda item: item.name):
            if candidate.name.startswith("."):
                continue
            if not candidate.is_dir():
                raise SequenceCacheError(f"unexpected file in Layer 2 episode directory: {candidate.name}")
            manifest, _ = self._read_verified_manifest(candidate.name, verify_payloads=True)
            if split is None or manifest["split"] == split:
                episode_ids.append(candidate.name)
        return tuple(episode_ids)


class SequenceDatasetCompiler:
    """Compile one verified Layer 1 episode into an atomic Layer 2 actor pair.

    Provider-specific tensorization, label compilation, and serialization are
    injected at this boundary. The compiler itself only projects the verified
    trajectory to the deck-free training episode and checks identity, split,
    profile, tick, and label-count alignment before committing the pair.
    """

    def __init__(
        self,
        *,
        cache: Layer2SequenceCache,
        dataset_fingerprint: str,
        training_config_fingerprint: str,
        input_contract_fingerprint: str,
        profile: ObservationProfile | str,
        actor_sequence_compiler: ActorSequenceCompiler,
        actor_sequence_encoder: ActorSequenceEncoder,
    ) -> None:
        if not isinstance(cache, Layer2SequenceCache):
            raise TypeError("cache must be a Layer2SequenceCache")
        for name, value in (
            ("dataset", dataset_fingerprint),
            ("training config", training_config_fingerprint),
            ("input contract", input_contract_fingerprint),
        ):
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{name} fingerprint must be a lowercase SHA-256 digest")
        if not callable(actor_sequence_compiler) or not callable(actor_sequence_encoder):
            raise TypeError("actor sequence compiler and encoder must be callable")
        self.cache = cache
        self.dataset_fingerprint = dataset_fingerprint
        self.training_config_fingerprint = training_config_fingerprint
        self.input_contract_fingerprint = input_contract_fingerprint
        self.profile = ObservationProfile(profile)
        self.actor_sequence_compiler = actor_sequence_compiler
        self.actor_sequence_encoder = actor_sequence_encoder

    @staticmethod
    def _verify_source_episode(episode: TrajectoryEpisode) -> str:
        if not isinstance(episode, TrajectoryEpisode):
            raise TypeError("SequenceDatasetCompiler consumes a verified TrajectoryEpisode")
        manifest = episode.manifest
        if len(episode.records) != manifest.record_count:
            raise ValueError("source record count differs from its manifest")
        record_hashes = tuple(record.content_sha256() for record in episode.records)
        if record_hashes != manifest.record_hashes:
            raise ValueError("source record hash sequence differs from its manifest")
        if aggregate_record_hash(manifest.episode_id, record_hashes) != manifest.aggregate_hash:
            raise ValueError("source aggregate hash differs from its manifest")
        if content_sha256(episode.metadata) != manifest.metadata_sha256:
            raise ValueError("source episode metadata hash differs from its manifest")
        keys = tuple((record.identity.tick, record.identity.actor) for record in episode.records)
        if not keys or keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("source records are not in strict (tick, actor) order")
        ticks_by_actor = {
            actor: tuple(tick for tick, owner in keys if owner == actor)
            for actor in (0, 1)
        }
        if (
            not ticks_by_actor[0]
            or ticks_by_actor[0] != ticks_by_actor[1]
            or ticks_by_actor[0][0] != 90
            or any(right - left != 5 for left, right in zip(ticks_by_actor[0], ticks_by_actor[0][1:]))
        ):
            raise ValueError("source actor records must share a continuous tick-90 five-tick timeline")
        if any(record.identity.episode_id != manifest.episode_id for record in episode.records):
            raise ValueError("source record episode identity differs from its manifest")
        if manifest.split not in _SPLITS:
            raise ValueError("source episode must have a frozen train/validation/test split")
        source_hash = episode.metadata.get("source_manifest_sha256", manifest.content_sha256())
        if not isinstance(source_hash, str) or not _SHA256.fullmatch(source_hash):
            raise ValueError("source manifest identity must be a lowercase SHA-256 digest")
        return source_hash

    def compile_episode(self, episode: TrajectoryEpisode) -> dict[str, Any]:
        """Compile and atomically publish one full, terminated source episode."""

        source_hash = self._verify_source_episode(episode)
        if episode.manifest.outcome != "terminated":
            raise ValueError("Layer 2 baseline compiler accepts terminated episodes only")
        training_episode = project_training_episode(episode)
        if training_episode.split != episode.manifest.split:
            raise ValueError("training projection changed the source episode split")

        sequences = tuple(
            self.actor_sequence_compiler(training_episode, actor)
            for actor in (0, 1)
        )
        payloads: dict[int, bytes] = {}
        ticks: dict[int, tuple[int, ...]] = {}
        label_counts: dict[int, dict[str, int]] = {}
        for actor, sequence in enumerate(sequences):
            expected_identities = tuple(
                step.identity for step in training_episode.steps if step.identity.actor == actor
            )
            sequence_profile = getattr(sequence.profile, "value", sequence.profile)
            if (
                sequence.episode_id != episode.manifest.episode_id
                or sequence.actor != actor
                or sequence_profile != self.profile.value
                or tuple(sequence.identities) != expected_identities
            ):
                raise ValueError(f"compiled actor {actor} identity/profile differs from the source episode")
            if not (
                len(sequence.inputs) == len(sequence.labels) == len(expected_identities)
            ):
                raise ValueError(f"compiled actor {actor} input/label count differs from source records")
            for identity, policy_input in zip(expected_identities, sequence.inputs, strict=True):
                if (
                    getattr(policy_input, "identity", None) != identity
                    or getattr(policy_input, "profile", None) != self.profile.value
                ):
                    raise ValueError(f"compiled actor {actor} PolicyInput identity/profile mismatch")
            actor_payload = self.actor_sequence_encoder(sequence)
            if not isinstance(actor_payload, bytes) or not actor_payload:
                raise ValueError(f"compiled actor {actor} encoder returned an empty/non-byte payload")
            payloads[actor] = actor_payload
            ticks[actor] = tuple(identity.tick for identity in expected_identities)
            category_counts = Counter(
                str(getattr(label.category, "value", label.category))
                for label in sequence.labels
            )
            label_counts[actor] = dict(sorted(category_counts.items()))

        return self.cache.commit_episode(
            episode_id=episode.manifest.episode_id,
            split=episode.manifest.split,
            outcome=episode.manifest.outcome,
            source_episode_sha256=source_hash,
            dataset_fingerprint=self.dataset_fingerprint,
            training_config_fingerprint=self.training_config_fingerprint,
            input_contract_fingerprint=self.input_contract_fingerprint,
            profile=self.profile.value,
            actor_payloads=payloads,
            ticks_by_actor=ticks,
            label_counts_by_actor=label_counts,
        )


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """Time-major padded lanes for a deterministic group of whole episodes.

    Padding is represented by ``None`` and marked false in ``valid_mask``.
    Consumers that train one lane at a time can use ``lanes`` directly; a
    vectorized consumer can use the masks without inferring episode boundaries.
    """

    episode_ids: tuple[str, ...]
    lanes: tuple[CompiledActorSequence, ...]
    valid_mask: tuple[tuple[bool, ...], ...]
    episode_start: tuple[tuple[bool, ...], ...]
    inputs_by_time: tuple[tuple[Any | None, ...], ...]
    labels_by_time: tuple[tuple[Any | None, ...], ...]

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_episode_groups(
        cls,
        groups: Iterable[tuple[CompiledActorSequence, CompiledActorSequence]],
    ) -> "SequenceBatch":
        groups = tuple(groups)
        if not groups:
            raise ValueError("sequence batch requires at least one complete episode")
        episode_ids: list[str] = []
        lanes: list[CompiledActorSequence] = []
        for group in groups:
            if len(group) != 2:
                raise ValueError("each sequence batch episode must have both actor lanes")
            left, right = group
            episode_ids.append(left.episode_id)
            lanes.extend((left, right))

        lengths = tuple(len(lane.inputs) for lane in lanes)
        if any(length <= 0 for length in lengths):
            raise ValueError("sequence batch lanes must contain at least one decision")
        time_steps = max(lengths)
        valid_mask = tuple(
            tuple(time < length for length in lengths)
            for time in range(time_steps)
        )
        episode_start = tuple(
            tuple(time == 0 and time < length for length in lengths)
            for time in range(time_steps)
        )
        inputs_by_time = tuple(
            tuple(lane.inputs[time] if time < length else None for lane, length in zip(lanes, lengths, strict=True))
            for time in range(time_steps)
        )
        labels_by_time = tuple(
            tuple(lane.labels[time] if time < length else None for lane, length in zip(lanes, lengths, strict=True))
            for time in range(time_steps)
        )
        return cls(
            episode_ids=tuple(episode_ids),
            lanes=tuple(lanes),
            valid_mask=valid_mask,
            episode_start=episode_start,
            inputs_by_time=inputs_by_time,
            labels_by_time=labels_by_time,
        )

    def validate(self) -> None:
        lane_count = len(self.lanes)
        if not self.episode_ids or len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("sequence batch episode IDs must be non-empty and unique")
        if lane_count != len(self.episode_ids) * 2:
            raise ValueError("sequence batch does not contain exactly two lanes per episode")
        profile = self.lanes[0].profile
        for episode_index, episode_id in enumerate(self.episode_ids):
            actor0, actor1 = self.lanes[episode_index * 2 : episode_index * 2 + 2]
            if (
                actor0.episode_id != episode_id
                or actor1.episode_id != episode_id
                or actor0.actor != 0
                or actor1.actor != 1
            ):
                raise ValueError("each sequence batch episode must be ordered actor 0 then actor 1")
            if actor0.profile != profile or actor1.profile != profile:
                raise ValueError("sequence batch mixes observation profiles")
            if (
                len(actor0.inputs) != len(actor0.labels)
                or len(actor1.inputs) != len(actor1.labels)
                or not actor0.inputs
                or not actor1.inputs
            ):
                raise ValueError("sequence batch lanes must have matching, non-empty inputs and labels")
        time_steps = len(self.valid_mask)
        if not time_steps or len(self.episode_start) != time_steps:
            raise ValueError("sequence batch time masks have invalid lengths")
        if time_steps != max(len(lane.inputs) for lane in self.lanes):
            raise ValueError("sequence batch does not use the maximum actor-lane length")
        if len(self.inputs_by_time) != time_steps or len(self.labels_by_time) != time_steps:
            raise ValueError("sequence batch padded values do not align with masks")
        for time in range(time_steps):
            if not all(
                len(matrix[time]) == lane_count
                for matrix in (self.valid_mask, self.episode_start, self.inputs_by_time, self.labels_by_time)
            ):
                raise ValueError("sequence batch rows have inconsistent lane counts")
            for lane_index, lane in enumerate(self.lanes):
                valid = self.valid_mask[time][lane_index]
                if self.episode_start[time][lane_index] != (valid and time == 0):
                    raise ValueError("sequence batch reset mask is inconsistent with lane starts")
                input_value = self.inputs_by_time[time][lane_index]
                label_value = self.labels_by_time[time][lane_index]
                if valid and (input_value is None or label_value is None):
                    raise ValueError("sequence batch padding mask is inconsistent with lane values")
                if not valid and (input_value is not None or label_value is not None):
                    raise ValueError("sequence batch padding mask is inconsistent with lane values")
                if valid and (input_value is not lane.inputs[time] or label_value is not lane.labels[time]):
                    raise ValueError("sequence batch lane content changed during collation")


class SequenceDataset:
    """Replay-ordered stream of precompiled actor sequences for one frozen split.

    The sequence factory is re-evaluated for each pass. Its output must be the
    two actor views for every selected episode, in the plan's exact order.
    Dataset fingerprints are supplied by the verified trajectory planner and
    are never recomputed from provider-only metadata here.
    """

    def __init__(
        self,
        sequence_factory: SequenceFactory,
        *,
        selected_episode_ids: tuple[str, ...],
        episode_splits: Mapping[str, str],
        dataset_fingerprint: str,
        resume_sequence_factory: Callable[[int], Iterable["CompiledActorSequence"]] | None = None,
    ) -> None:
        if not callable(sequence_factory):
            raise TypeError("sequence_factory must be callable")
        if not selected_episode_ids or any(not item for item in selected_episode_ids):
            raise ValueError("selected episode IDs must be non-empty")
        if len(set(selected_episode_ids)) != len(selected_episode_ids):
            raise ValueError("selected episode IDs contain duplicates")
        if not isinstance(episode_splits, Mapping):
            raise TypeError("episode_splits must map episode IDs to frozen splits")
        if any(episode_splits.get(item) != "train" for item in selected_episode_ids):
            raise ValueError("SequenceDataset accepts selected episodes from the train split only")
        if not isinstance(dataset_fingerprint, str) or not _SHA256.fullmatch(dataset_fingerprint):
            raise ValueError("dataset_fingerprint must be a lowercase SHA-256 digest")
        self._sequence_factory = sequence_factory
        self._resume_sequence_factory = resume_sequence_factory
        self.selected_episode_ids = tuple(selected_episode_ids)
        self.dataset_fingerprint = dataset_fingerprint

    @classmethod
    def from_layer2_cache(
        cls,
        cache: Layer2SequenceCache,
        *,
        sequence_decoder: Callable[[CachedSequenceEpisode], tuple[CompiledActorSequence, CompiledActorSequence]],
        selected_episode_ids: tuple[str, ...],
        episode_splits: Mapping[str, str],
        dataset_fingerprint: str,
        training_config_fingerprint: str,
        input_contract_fingerprint: str,
    ) -> "SequenceDataset":
        """Build the train view from verified, immutable Layer 2 cache episodes."""

        def sequence_factory_from(start_episode_index: int) -> Iterator[CompiledActorSequence]:
            for episode_id in selected_episode_ids[start_episode_index:]:
                episode = cache.read_episode(
                    episode_id,
                    expected_dataset_fingerprint=dataset_fingerprint,
                    expected_training_config_fingerprint=training_config_fingerprint,
                    expected_input_contract_fingerprint=input_contract_fingerprint,
                )
                if episode.split != "train":
                    raise ValueError("SequenceDataset accepts cached training episodes only")
                lanes = sequence_decoder(episode)
                if not isinstance(lanes, tuple) or len(lanes) != 2:
                    raise ValueError("sequence decoder must return actor 0 and actor 1 lanes")
                for actor, lane in enumerate(lanes):
                    if getattr(lane, "episode_id", None) != episode.episode_id:
                        raise ValueError("decoded sequence episode identity differs from cache manifest")
                    if getattr(lane, "actor", None) != actor:
                        raise ValueError("sequence decoder must return actor 0 then actor 1 lanes")
                    lane_profile = getattr(lane, "profile", None)
                    if getattr(lane_profile, "value", lane_profile) != episode.profile:
                        raise ValueError("decoded sequence profile differs from cache manifest")
                    identities = getattr(lane, "identities", None)
                    if not isinstance(identities, tuple):
                        raise ValueError("decoded sequence identities are missing")
                    if any(
                        getattr(identity, "episode_id", None) != episode.episode_id
                        or getattr(identity, "actor", None) != actor
                        for identity in identities
                    ):
                        raise ValueError("decoded sequence identity differs from cache manifest")
                    lane_ticks = tuple(getattr(identity, "tick", None) for identity in identities)
                    if lane_ticks != episode.ticks_by_actor[actor]:
                        raise ValueError("decoded sequence ticks differ from cache manifest")
                yield from lanes

        return cls(
            lambda: sequence_factory_from(0),
            selected_episode_ids=selected_episode_ids,
            episode_splits=episode_splits,
            dataset_fingerprint=dataset_fingerprint,
            resume_sequence_factory=sequence_factory_from,
        )

    def iter_episode_groups(
        self,
        *,
        start_after_episode_id: str | None = None,
    ) -> Iterator[tuple[CompiledActorSequence, CompiledActorSequence]]:
        """Yield complete actor pairs, optionally resuming after a whole episode."""

        if start_after_episode_id is not None and start_after_episode_id not in self.selected_episode_ids:
            raise ValueError("resume episode is not in the selected training dataset")
        skip_through = (
            self.selected_episode_ids.index(start_after_episode_id)
            if start_after_episode_id is not None
            else -1
        )
        start_episode_index = skip_through + 1
        if self._resume_sequence_factory is None:
            iterator = iter(self._sequence_factory())
            episodes_to_read = self.selected_episode_ids
        else:
            iterator = iter(self._resume_sequence_factory(start_episode_index))
            episodes_to_read = self.selected_episode_ids[start_episode_index:]
        profile: Any = None
        for episode_index, episode_id in enumerate(episodes_to_read):
            try:
                actor0 = next(iterator)
                actor1 = next(iterator)
            except StopIteration as error:
                raise ValueError(f"sequence dataset is missing actor lanes for {episode_id!r}") from error
            pair = (actor0, actor1)
            if any(
                sequence.episode_id != episode_id or sequence.actor != actor
                for actor, sequence in enumerate(pair)
            ):
                raise ValueError("sequence dataset order must match selected episodes and actors 0 then 1")
            if profile is None:
                profile = actor0.profile
            if actor0.profile != profile or actor1.profile != profile:
                raise ValueError("sequence dataset mixes observation profiles")
            if self._resume_sequence_factory is not None or episode_index > skip_through:
                yield pair
        try:
            next(iterator)
        except StopIteration:
            return
        raise ValueError("sequence dataset contains lanes outside the selected training episodes")

    def __iter__(self) -> Iterator[CompiledActorSequence]:
        for group in self.iter_episode_groups():
            yield from group

    def iter_batches(self, *, max_episodes: int) -> Iterator[SequenceBatch]:
        """Collate deterministic batches without splitting actor pairs."""

        if type(max_episodes) is not int or max_episodes <= 0:
            raise ValueError("max_episodes must be a positive integer")
        pending: list[tuple[CompiledActorSequence, CompiledActorSequence]] = []
        for group in self.iter_episode_groups():
            pending.append(group)
            if len(pending) == max_episodes:
                yield SequenceBatch.from_episode_groups(pending)
                pending = []
        if pending:
            yield SequenceBatch.from_episode_groups(pending)
