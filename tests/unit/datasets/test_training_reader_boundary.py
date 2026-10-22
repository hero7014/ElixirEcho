from __future__ import annotations

import json

from tests.unit.datasets.test_trajectory_store import _step, _writer
from elixir_echo.datasets.trajectory import TrainingTrajectoryReader


def test_training_projection_has_only_training_permitted_record_fields(tmp_path):
    writer = _writer(tmp_path)
    writer.append(_step(90, 0))
    writer.append(_step(90, 1))
    writer.commit()
    episode = TrainingTrajectoryReader(tmp_path).read_episode("episode-1")

    record = episode.steps[0]
    assert not hasattr(record, "episode_context_ref")
    assert not hasattr(record, "validation")
    assert not hasattr(record, "fingerprints")
    payload = json.dumps(record.to_dict(), sort_keys=True)
    assert "provider" not in payload
    assert "deck" not in payload
