from pathlib import Path

from elixir_echo.datasets.trajectory import TrainingTrajectoryReader, TrajectoryWriter
from elixir_echo.training.firstlight_run import (
    iter_verified_training_episodes,
    plan_training_dataset,
)

from tests.unit.test_cli import _step


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
