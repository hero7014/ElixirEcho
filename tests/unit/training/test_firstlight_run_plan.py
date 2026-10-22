from pathlib import Path

import json

from elixir_echo.datasets.legacy import LegacyTrainingTrajectoryStoreReader
from elixir_echo.datasets.trajectory import (
    TrainingTrajectoryReader,
    TrajectoryWriter,
    aggregate_record_hash,
)
from elixir_echo.domain.values import content_sha256
from elixir_echo.training.firstlight_run import (
    iter_verified_training_episodes,
    plan_training_dataset,
)

from tests.unit.test_cli import _step


def _write_legacy_store(root: Path, episode_id: str, split: str, outcome: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    assignments_path = root / "split_manifest.json"
    if assignments_path.exists():
        assignments = json.loads(assignments_path.read_text(encoding="utf-8"))["assignments"]
    else:
        assignments = {}
    assignments[episode_id] = split
    split_payload = {"version": "trajectory-split.v1", "assignments": assignments}
    assignments_path.write_text(json.dumps(split_payload, sort_keys=True), encoding="utf-8")
    split_hash = content_sha256(split_payload)
    for manifest_path in root.glob("episodes/*/*/episode_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("content_sha256", None)
        manifest["split_manifest_hash"] = split_hash
        manifest["content_sha256"] = content_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    context = {"episode_id": episode_id, "deck0": list(range(8)), "deck1": list(range(8, 16))}
    context_hash = content_sha256(context)
    context_path = root / "episodes" / "00" / episode_id
    context_path.mkdir(parents=True, exist_ok=True)
    (context_path / "episode_context.json").write_text(
        json.dumps({**context, "content_sha256": context_hash}, sort_keys=True), encoding="utf-8"
    )
    fixture = json.loads(Path("tests/fixtures/legacy_decision_step.json").read_text(encoding="utf-8"))
    records = []
    for actor in (0, 1):
        record = json.loads(json.dumps(fixture))
        record["identity"] = {"episode_id": episode_id, "tick": 90, "actor": actor}
        record["action_mask"]["identity"] = dict(record["identity"])
        record["episode_context_ref"] = context_hash
        record.pop("content_sha256", None)
        record["content_sha256"] = content_sha256(record)
        records.append(record)
    (context_path / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    hashes = [record["content_sha256"] for record in records]
    manifest = {
        "version": "full-episode-manifest.v1",
        "episode_id": episode_id,
        "replay_tag": f"replay-{episode_id}",
        "outcome": outcome,
        "split": split,
        "decision_interval_ticks": 5,
        "episode_context_ref": context_hash,
        "record_count": 2,
        "first_tick": 90,
        "last_tick": 90,
        "record_hashes": hashes,
        "aggregate_hash": aggregate_record_hash(episode_id, hashes),
        "split_manifest_hash": split_hash,
        "sequences": [
            {"actor": actor, "episode_start_ordinal": 0, "length": 1} for actor in (0, 1)
        ],
    }
    manifest["content_sha256"] = content_sha256(manifest)
    (context_path / "episode_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def _write_episode(root: Path, episode_id: str, *, split: str, deck: list[int]) -> None:
    writer = TrajectoryWriter(root).begin(
        {
            "episode_id": episode_id,
            "replay_tag": f"replay-{episode_id}",
            "outcome": "terminated",
            "split": split,
            "episode_context_ref": "private-context-hash",
            "episode_context": {"deck0": deck, "deck1": list(reversed(deck))},
        }
    )
    for actor in (0, 1):
        writer.append(_step(actor, episode_id))
    writer.commit()


def test_dataset_fingerprint_uses_safe_training_projection_not_episode_decks(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_episode(left, "safe-episode", split="train", deck=list(range(8)))
    _write_episode(right, "safe-episode", split="train", deck=list(range(20, 28)))

    left_plan = plan_training_dataset(TrainingTrajectoryReader(left), max_episodes=1)
    right_plan = plan_training_dataset(TrainingTrajectoryReader(right), max_episodes=1)

    assert left_plan.fingerprint == right_plan.fingerprint
    assert left_plan.selected_episode_ids == ("safe-episode",)
    assert left_plan.selected_decision_count == 2
    assert left_plan.selected_label_counts == {"wait": 2}
    assert tuple(iter_verified_training_episodes(TrainingTrajectoryReader(left), left_plan))[0].episode_id == "safe-episode"


def test_plan_excludes_validation_and_truncated_episodes(tmp_path: Path) -> None:
    _write_episode(tmp_path, "train-episode", split="train", deck=list(range(8)))
    writer = TrajectoryWriter(tmp_path).begin(
        {
            "episode_id": "short-episode",
            "replay_tag": "replay-short",
            "outcome": "truncated",
            "split": "train",
            "episode_context_ref": "private-context-hash",
            "episode_context": {"deck0": list(range(8))},
        }
    )
    for actor in (0, 1):
        writer.append(_step(actor, "short-episode"))
    writer.commit()

    plan = plan_training_dataset(TrainingTrajectoryReader(tmp_path), max_episodes=5)

    assert plan.selected_episode_count == 1
    assert plan.selected_episode_ids == ("train-episode",)
    assert plan.excluded_truncated_episode_count == 1


def test_legacy_store_catalog_uses_manifests_and_reader_hides_decks(tmp_path: Path) -> None:
    _write_legacy_store(tmp_path, "train-episode", "train", "terminated")
    _write_legacy_store(tmp_path, "validation-episode", "validation", "terminated")
    reader = LegacyTrainingTrajectoryStoreReader(tmp_path)

    plan = plan_training_dataset(reader, max_episodes=1)
    episode = reader.read_episode("train-episode")

    assert plan.source_episode_count == 2
    assert plan.source_split_counts == {"train": 1, "validation": 1}
    assert plan.selected_episode_ids == ("train-episode",)
    assert episode.episode_id == "train-episode"
    assert "deck0" not in json.dumps(episode.to_dict())
    assert tuple(iter_verified_training_episodes(reader, plan))[0].episode_id == "train-episode"
