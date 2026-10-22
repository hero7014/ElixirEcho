from __future__ import annotations

import json
from pathlib import Path

from elixir_echo.cli import main
from elixir_echo.datasets.trajectory import TrajectoryWriter
from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation


def _step(actor: int, episode_id: str = "cli-episode") -> DecisionStep:
    identity = DecisionIdentity(episode_id, 90, actor)
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


def _write_episode(root: Path, *, split: str = "test", deck0: list[int] | None = None) -> None:
    writer = TrajectoryWriter(root).begin(
        {
            "episode_id": "cli-episode",
            "replay_tag": "replay-hash",
            "outcome": "terminated",
            "split": split,
            "episode_context_ref": "private-context-hash",
            "episode_context": {"deck0": deck0 or [1, 2, 3, 4, 5, 6, 7, 8]},
        }
    )
    writer.append(_step(0, "cli-episode"))
    writer.append(_step(1, "cli-episode"))
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


def test_train_firstlight_defaults_to_safe_dry_run_and_excludes_non_train(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    _write_episode(tmp_path)
    calls = []

    def forbidden_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("dry-run must not construct a model or train")

    monkeypatch.setattr("elixir_echo.cli._run_firstlight_training", forbidden_run, raising=False)

    result = main(
        [
            "train-firstlight",
            str(tmp_path),
            "--firstlight-root",
            str(tmp_path / "provider"),
            "--checkpoint-directory",
            str(tmp_path / "checkpoints"),
            "--max-episodes",
            "1",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 0
    assert calls == []
    assert report["mode"] == "dry_run"
    assert report["selected_episode_count"] == 0
    assert report["excluded_split_episode_count"] == 1
    assert "dataset_fingerprint" in report
    assert "deck0" not in captured.out
    assert not (tmp_path / "checkpoints").exists()


def test_train_firstlight_run_requires_explicit_confirmation(tmp_path: Path, capsys) -> None:
    _write_episode(tmp_path)
    result = main(
        [
            "train-firstlight",
            str(tmp_path),
            "--firstlight-root",
            str(tmp_path / "provider"),
            "--checkpoint-directory",
            str(tmp_path / "checkpoints"),
            "--max-episodes",
            "1",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert result == 0
    assert report["mode"] == "dry_run"


def test_train_firstlight_explicit_run_calls_capped_runner(tmp_path: Path, capsys, monkeypatch) -> None:
    _write_episode(tmp_path, split="train")
    seen = {}

    def fake_run(reader, plan, **kwargs):
        seen["plan"] = plan
        seen["kwargs"] = kwargs
        return {"mode": "training_run", "processed_episode_count": 1}

    monkeypatch.setattr("elixir_echo.cli.run_firstlight_training", fake_run)
    result = main(
        [
            "train-firstlight",
            str(tmp_path),
            "--firstlight-root",
            str(tmp_path / "provider"),
            "--checkpoint-directory",
            str(tmp_path / "checkpoints"),
            "--max-episodes",
            "1",
            "--run",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert seen["plan"].selected_episode_ids == ("cli-episode",)
    assert seen["kwargs"]["checkpoint_every"] == 10
    assert report["mode"] == "training_run"
    assert report["processed_episode_count"] == 1


def test_train_firstlight_rejects_missing_episode_store(tmp_path: Path, capsys) -> None:
    result = main(
        [
            "train-firstlight",
            str(tmp_path),
            "--firstlight-root",
            str(tmp_path / "provider"),
            "--checkpoint-directory",
            str(tmp_path / "checkpoints"),
            "--max-episodes",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "episodes directory" in captured.err
