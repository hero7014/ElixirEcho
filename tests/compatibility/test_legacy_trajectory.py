from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from elixir_echo.datasets.legacy import LegacyTrajectoryReader
from elixir_echo.domain.action import ACTION_MASK_SCHEMA


def test_opt_in_legacy_episode_is_verified_and_mapped_without_rewrite(monkeypatch):
    raw_path = os.environ.get("ELIXIR_ECHO_LEGACY_EPISODE")
    if not raw_path:
        pytest.skip("set ELIXIR_ECHO_LEGACY_EPISODE to a sealed local legacy episode")
    episode_path = Path(raw_path).resolve()
    if not episode_path.is_dir():
        pytest.skip("configured legacy episode is not present on this machine")
    before = {
        item.relative_to(episode_path).as_posix(): item.read_bytes()
        for item in episode_path.rglob("*")
        if item.is_file()
    }

    episode = LegacyTrajectoryReader(episode_path).read_episode()

    after = {
        item.relative_to(episode_path).as_posix(): item.read_bytes()
        for item in episode_path.rglob("*")
        if item.is_file()
    }
    assert before == after
    assert episode.records
    identities = [(item.identity.tick, item.identity.actor) for item in episode.records]
    assert identities == sorted(identities)
    assert len(identities) == episode.manifest.record_count
    assert episode.manifest.outcome in {"terminated", "truncated"}
    with (episode_path / "records.jsonl").open("r", encoding="utf-8") as stream:
        source_records = [json.loads(line) for line in stream if line.strip()]
    assert len(source_records) == len(episode.records)
    assert episode.manifest.record_hashes == tuple(item["content_sha256"] for item in source_records)
    for source, mapped in zip(source_records, episode.records, strict=True):
        assert source["identity"] == mapped.identity.to_dict()
        assert source["expert_action"] == mapped.expert_action.to_legacy()
        expected_mask = dict(source["action_mask"])
        expected_mask["schema"] = ACTION_MASK_SCHEMA
        assert expected_mask == mapped.action_mask.to_dict()
        canonical_frame = dict(source["frame"])
        canonical_frame.pop("version")
        canonical_frame["schema"] = "elixir_echo.observation@1"
        assert mapped.observation.to_dict() == canonical_frame
    source_manifest = json.loads((episode_path / "episode_manifest.json").read_text(encoding="utf-8"))
    assert tuple(item["actor"] for item in source_manifest["sequences"]) == (0, 1)
    assert all(item["episode_start_ordinal"] == 0 for item in source_manifest["sequences"])
