"""Read-only compatibility with sealed SemanticFrame full-episode caches."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation
from elixir_echo.domain.values import content_sha256
from .trajectory import TrajectoryEpisode, TrajectoryManifest, aggregate_record_hash


LEGACY_EPISODE_ENV = "ELIXIR_ECHO_LEGACY_EPISODE"


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
            # The compatibility view carries the verified source identities;
            # canonical in-memory step hashes are intentionally not substituted.
            record_hashes=tuple(hashes),
            aggregate_hash=str(manifest_payload["aggregate_hash"]),
            metadata_sha256=content_sha256(metadata),
            split=str(manifest_payload["split"]),
        )
        return TrajectoryEpisode(new_manifest, tuple(records), metadata)
