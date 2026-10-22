"""Read-only command-line checks for ElixirEcho trajectory data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Sequence

from elixir_echo import __version__
from elixir_echo.datasets.trajectory import TrainingTrajectoryReader


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elixir-echo",
        description="ElixirEcho replay-to-policy research tools.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ElixirEcho {__version__}",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser(
        "verify-trajectory",
        help="verify committed episode hashes and print a training-safe summary",
    )
    verify.add_argument("root", type=Path, help="trajectory cache root")
    return parser


def _verify_trajectory(root: Path) -> dict[str, object]:
    cache_root = root.expanduser().resolve()
    if not (cache_root / "episodes").is_dir():
        raise ValueError(f"trajectory store has no episodes directory: {cache_root}")

    episodes = []
    outcome_counts: Counter[str] = Counter()
    record_count = 0
    for episode in TrainingTrajectoryReader(cache_root).iter_episodes():
        if not episode.steps:
            raise ValueError(f"trajectory episode is empty: {episode.episode_id}")
        ticks = tuple(step.identity.tick for step in episode.steps)
        actors = sorted({step.identity.actor for step in episode.steps})
        episodes.append(
            {
                "episode_id": episode.episode_id,
                "outcome": episode.outcome,
                "split": episode.split,
                "record_count": len(episode.steps),
                "first_tick": min(ticks),
                "last_tick": max(ticks),
                "actors": actors,
            }
        )
        outcome_counts[episode.outcome] += 1
        record_count += len(episode.steps)

    if not episodes:
        raise ValueError(f"no complete trajectory episodes found: {cache_root}")

    return {
        "schema": "elixir_echo.trajectory_verification@1",
        "episode_count": len(episodes),
        "record_count": record_count,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "episodes": episodes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = _verify_trajectory(args.root)
    except (OSError, TypeError, ValueError) as error:
        print(f"elixir-echo: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
