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
from elixir_echo.training.firstlight_run import plan_training_dataset, run_firstlight_training


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
    train = commands.add_parser(
        "train-firstlight",
        help="plan a capped recurrent V4 training run (execution requires --run)",
    )
    train.add_argument("root", type=Path, help="full-episode trajectory cache root")
    train.add_argument("--firstlight-root", type=Path, required=True, help="read-only FirstLight checkout")
    train.add_argument("--checkpoint-directory", type=Path, required=True, help="new checkpoint output directory")
    train.add_argument("--max-episodes", type=int, required=True, help="hard cap on train-split terminated episodes")
    train.add_argument("--checkpoint-every", type=int, default=10, help="checkpoint after this many episodes")
    train.add_argument("--device", default="cpu", help="training device, for example cpu or cuda")
    train.add_argument("--resume-from", type=Path, help="resume from a compatible checkpoint directory")
    train.add_argument("--run", action="store_true", help="execute training; without this flag only dry-run planning occurs")
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


def _train_firstlight(args: argparse.Namespace) -> dict[str, object]:
    cache_root = args.root.expanduser().resolve()
    if not (cache_root / "episodes").is_dir():
        raise ValueError(f"trajectory store has no episodes directory: {cache_root}")
    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be a positive integer")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be a positive integer")

    reader = TrainingTrajectoryReader(cache_root)
    plan = plan_training_dataset(reader, max_episodes=args.max_episodes)
    if not args.run:
        return {
            "schema": "elixir_echo.training_run_plan@1",
            "mode": "dry_run",
            "cache_root": str(cache_root),
            "firstlight_root": str(args.firstlight_root.expanduser().resolve()),
            "checkpoint_directory": str(args.checkpoint_directory.expanduser().absolute()),
            "max_episodes": args.max_episodes,
            "checkpoint_every": args.checkpoint_every,
            "device": args.device,
            "resume_requested": args.resume_from is not None,
            **plan.to_dict(),
            "execution_note": "No FirstLight modules were imported and no checkpoint/model was created.",
        }
    return {
        "schema": "elixir_echo.training_run_result@1",
        "cache_root": str(cache_root),
        **run_firstlight_training(
            reader,
            plan,
            firstlight_root=args.firstlight_root,
            checkpoint_directory=args.checkpoint_directory,
            checkpoint_every=args.checkpoint_every,
            device=args.device,
            resume_from=args.resume_from,
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-trajectory":
            report = _verify_trajectory(args.root)
        else:
            report = _train_firstlight(args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"elixir-echo: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
