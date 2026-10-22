from __future__ import annotations

import json
from pathlib import Path

from elixir_echo.cli import main
from elixir_echo.datasets.trajectory import TrajectoryWriter
from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation


def _step(actor: int) -> DecisionStep:
    identity = DecisionIdentity("cli-episode", 90, actor)
    return DecisionStep(
        identity=identity,
        episode_context_ref="private-context-hash",
        observation=Observation(
            match={"phase": "normal", "remaining_ms": 180_000},
            players=({"owner": actor, "relation": "self"},),
            towers=(),
            entities=(),
            terminal={"ended": False},
        ),
        action_mask=ActionMask(identity, {"wait": True}),
        expert_action=ExpertAction.wait(),
        event_delta=(),
        validation={"accepted": True, "private_diagnostic": "hidden"},
        fingerprints={"provider_path": "private/path"},
    )


def _write_episode(root: Path) -> None:
    writer = TrajectoryWriter(root).begin(
        {
            "episode_id": "cli-episode",
            "replay_tag": "replay-hash",
            "outcome": "terminated",
            "split": "test",
            "episode_context_ref": "private-context-hash",
            "episode_context": {"deck0": [1, 2, 3, 4, 5, 6, 7, 8]},
        }
    )
    writer.append(_step(0))
    writer.append(_step(1))
    writer.commit()


def test_verify_trajectory_reports_only_training_safe_summary(tmp_path: Path, capsys) -> None:
    _write_episode(tmp_path)

    result = main(["verify-trajectory", str(tmp_path)])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 0
    assert report["schema"] == "elixir_echo.trajectory_verification@1"
    assert report["episode_count"] == 1
    assert report["record_count"] == 2
    assert report["episodes"] == [
        {
            "episode_id": "cli-episode",
            "outcome": "terminated",
            "split": "test",
            "record_count": 2,
            "first_tick": 90,
            "last_tick": 90,
            "actors": [0, 1],
        }
    ]
    assert all(
        forbidden not in captured.out
        for forbidden in (
            "deck0",
            "private-context-hash",
            "private_diagnostic",
            "private/path",
        )
    )


def test_verify_trajectory_rejects_store_without_episode_directory(
    tmp_path: Path,
    capsys,
) -> None:
    result = main(["verify-trajectory", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 2
    assert "has no episodes directory" in captured.err


def test_verify_trajectory_rejects_empty_episode_store(tmp_path: Path, capsys) -> None:
    (tmp_path / "episodes").mkdir()

    result = main(["verify-trajectory", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 2
    assert "no complete trajectory episodes" in captured.err
