"""Portable command-line workflows for ElixirEcho."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

from elixir_echo import __version__
from elixir_echo.config import ConfigError, ProjectConfig, load_config
from elixir_echo.datasets.legacy import LegacyTrainingTrajectoryStoreReader
from elixir_echo.datasets.trajectory import (
    TrainingTrajectoryReader,
    TrajectoryReader,
    TrajectoryWriter,
)
from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation
from elixir_echo.domain.values import content_sha256
from elixir_echo.policies.protocol import PolicyInput
from elixir_echo.temporal.public_state import PublicStateTracker
from elixir_echo.training.firstlight_run import plan_training_dataset, run_firstlight_training


def _add_config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="TOML configuration path (default: ./configs/elixir_echo_baseline.toml if present)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elixir-echo",
        description="ElixirEcho replay-to-policy research tools.",
    )
    parser.add_argument("--version", action="version", version=f"ElixirEcho {__version__}")
    _add_config_option(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check configured data and Native integration prerequisites")
    _add_config_option(doctor)

    fixture = commands.add_parser("fixture", help="run a synthetic, dependency-free end-to-end fixture")
    _add_config_option(fixture)
    fixture.add_argument("--output", type=Path, help="new fixture output directory (default: <output_path>/fixture)")

    capture = commands.add_parser(
        "capture",
        help="check or explicitly capture one exact Native replay (--run required)",
    )
    _add_config_option(capture)
    capture.add_argument("--replay-dataset", type=Path, help="IL_Replay dataset root containing replays/*.parquet")
    capture.add_argument("--replay-tag", help="exact source replay identity; required with --run")
    capture.add_argument("--firstlight-root", type=Path, help="read-only FirstLight checkout")
    capture.add_argument("--output", type=Path, help="trajectory cache root")
    capture.add_argument("--host", default="127.0.0.1", help="already-running Native control host")
    capture.add_argument("--port", type=int, default=26789, help="already-running Native control port")
    capture.add_argument("--run", action="store_true", help="connect to Native and capture the selected replay")
    capture.add_argument("--debug-train", action="store_true", help="assign train only inside an isolated runs/real-replay-chain directory")

    compile_command = commands.add_parser("compile", help="compile bounded Layer 2 V4 sequences (explicit --run required)")
    _add_config_option(compile_command)
    compile_command.add_argument("root", nargs="?", type=Path, help="trajectory store (defaults to configured dataset)")
    compile_command.add_argument("--firstlight-root", type=Path, help="read-only FirstLight checkout")
    compile_command.add_argument("--output", type=Path, help="new Layer 2 cache directory")
    compile_command.add_argument("--max-episodes", type=int, help="maximum terminated train-split episodes to compile")
    compile_command.add_argument("--run", action="store_true", help="compile inputs; without this flag only verify the source store")

    verify = commands.add_parser(
        "verify-trajectory",
        help="verify committed episode hashes and print a training-safe summary",
    )
    _add_config_option(verify)
    verify.add_argument("root", nargs="?", type=Path, help="trajectory cache root")

    train = commands.add_parser(
        "train",
        aliases=["train-firstlight"],
        help="plan capped recurrent V4 training; execution requires --run",
    )
    _add_config_option(train)
    train.add_argument("root", nargs="?", type=Path, help="full-episode trajectory cache root")
    train.add_argument("--firstlight-root", type=Path, help="read-only FirstLight checkout")
    train.add_argument("--checkpoint-directory", type=Path, help="checkpoint output directory")
    train.add_argument("--max-episodes", type=int, help="hard cap on train-split terminated episodes")
    train.add_argument("--checkpoint-every", type=int, help="checkpoint after this many episodes")
    train.add_argument("--device", help="training device, for example cpu or cuda")
    train.add_argument("--resume-from", type=Path, help="resume from a compatible checkpoint directory")
    train.add_argument("--run", action="store_true", help="execute training; otherwise only print a dry-run plan")

    inspect = commands.add_parser("inspect-checkpoint", help="verify a checkpoint manifest and weights hash without loading it")
    _add_config_option(inspect)
    inspect.add_argument("path", type=Path, help="checkpoint directory")

    evaluate = commands.add_parser("evaluate", help="inspect evaluation readiness; metrics workflow is added in Task 8")
    _add_config_option(evaluate)
    evaluate.add_argument("checkpoint", type=Path, help="checkpoint directory to evaluate")
    evaluate.add_argument("--dataset", type=Path, help="held-out trajectory dataset")
    return parser


def _config(args: argparse.Namespace, **overrides: Any) -> ProjectConfig:
    return load_config(
        config_path=getattr(args, "config", None),
        overrides={key: value for key, value in overrides.items() if value is not None},
    )


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
    config = _config(
        args,
        dataset_path=args.root,
        firstlight_root=args.firstlight_root,
        output_path=args.checkpoint_directory,
        max_episodes=args.max_episodes,
        checkpoint_every=args.checkpoint_every,
        compute_device=args.device,
    )
    cache_root = config.dataset_path
    if not (cache_root / "episodes").is_dir():
        raise ValueError(f"trajectory store has no episodes directory: {cache_root}")
    if config.firstlight_root is None:
        raise ValueError("FirstLight is not configured; set --firstlight-root or ELIXIR_ECHO_FIRSTLIGHT_ROOT")
    if args.resume_from is not None and not args.resume_from.exists():
        raise ValueError(f"resume checkpoint does not exist: {args.resume_from.expanduser().resolve()}")

    if next((cache_root / "episodes").glob("*/*/trajectory_manifest.json"), None) is not None:
        reader = TrainingTrajectoryReader(cache_root)
        cache_format = "trajectory-store"
        verification_scope = "all episode records are verified while cataloging"
    elif (cache_root / "split_manifest.json").is_file() and next(
        (cache_root / "episodes").glob("*/*/episode_manifest.json"), None
    ) is not None:
        reader = LegacyTrainingTrajectoryStoreReader(cache_root)
        cache_format = "sealed-legacy-full-episode-store"
        verification_scope = (
            "all episode/split manifests are verified; selected episodes receive full record/context/hash validation"
        )
    else:
        raise ValueError(f"unsupported trajectory cache layout: {cache_root}")
    plan = plan_training_dataset(reader, max_episodes=config.max_episodes)
    if not args.run:
        return {
            "schema": "elixir_echo.training_run_plan@1",
            "mode": "dry_run",
            "cache_root": str(cache_root),
            "cache_format": cache_format,
            "verification_scope": verification_scope,
            "firstlight_root": str(config.firstlight_root),
            "checkpoint_directory": str(config.output_path),
            "max_episodes": config.max_episodes,
            "checkpoint_every": config.checkpoint_every,
            "device": config.compute_device,
            "resume_requested": args.resume_from is not None,
            **plan.to_dict(),
            "execution_note": "No FirstLight modules were imported and no checkpoint/model was created.",
        }
    return {
        "schema": "elixir_echo.training_run_result@1",
        "cache_root": str(cache_root),
        "cache_format": cache_format,
        **run_firstlight_training(
            reader,
            plan,
            firstlight_root=config.firstlight_root,
            checkpoint_directory=config.output_path,
            checkpoint_every=config.checkpoint_every,
            device=config.compute_device,
            resume_from=args.resume_from,
        ),
    }


def _file_check(path: Path | None, label: str, setting: str, *, directory: bool = False) -> dict[str, str]:
    if path is None:
        return {"name": label, "status": "missing", "action": f"Configure {setting} in TOML or environment."}
    candidate = path.expanduser().resolve()
    valid = candidate.is_dir() if directory else candidate.exists()
    if valid:
        return {"name": label, "status": "ready", "path": str(candidate)}
    return {
        "name": label,
        "status": "missing",
        "path": str(candidate),
        "action": f"Path is not available; configure {setting} with the correct local path.",
    }


def _doctor(args: argparse.Namespace) -> dict[str, object]:
    config = _config(args)
    firstlight = config.firstlight_root
    firstlight_ready = firstlight is not None and (firstlight / "native_runner" / "__init__.py").is_file()
    checks: list[dict[str, str]] = [
        _file_check(config.data_root, "data_root", "ELIXIR_ECHO_DATA_ROOT", directory=True),
        _file_check(config.dataset_path, "dataset_path", "ELIXIR_ECHO_DATASET_PATH", directory=True),
        {
            "name": "firstlight_root",
            "status": "ready" if firstlight_ready else "missing",
            **({"path": str(firstlight.resolve())} if firstlight is not None else {}),
            **({} if firstlight_ready else {"action": "Set ELIXIR_ECHO_FIRSTLIGHT_ROOT to a FirstLight checkout containing native_runner/__init__.py."}),
        },
        _file_check(config.game_assets_path, "game_assets_path", "ELIXIR_ECHO_GAME_ASSETS_PATH", directory=True),
        _file_check(config.mumu_manager_path, "mumu_manager_path", "ELIXIR_ECHO_MUMU_MANAGER_PATH"),
    ]
    adb_path = shutil.which(config.adb_executable)
    if adb_path is None and Path(config.adb_executable).is_file():
        adb_path = str(Path(config.adb_executable).resolve())
    checks.append(
        {"name": "adb_executable", "status": "ready", "path": adb_path}
        if adb_path
        else {"name": "adb_executable", "status": "missing", "action": "Install Android platform-tools or set ELIXIR_ECHO_ADB_EXECUTABLE."}
    )
    checks.append(
        {"name": "device_serial", "status": "ready" if config.device_serial else "missing", **({} if config.device_serial else {"action": "Set ELIXIR_ECHO_DEVICE_SERIAL to the intended emulator/device serial."})}
    )
    native_prerequisites = {"firstlight_root", "game_assets_path", "mumu_manager_path", "adb_executable", "device_serial"}
    native_ready = all(item["status"] == "ready" for item in checks if item["name"] in native_prerequisites)
    return {
        "schema": "elixir_echo.doctor@1",
        "fixture_ready": True,
        "native_ready": native_ready,
        "config_file": str(config.config_path) if config.config_path else None,
        "checks": checks,
        "note": "doctor is read-only; missing local integrations do not block the synthetic fixture.",
    }


def _capture_preflight(args: argparse.Namespace) -> dict[str, object]:
    if args.debug_train and not args.run:
        raise ValueError("capture --debug-train requires --run")
    config = _config(args, firstlight_root=args.firstlight_root)
    source = (args.replay_dataset or config.data_root / "raw" / "IL_Replay").expanduser().resolve()
    output = (args.output or config.data_root / "trajectory_cache").expanduser().resolve()
    provider = config.firstlight_root
    read_only_roots = [source]
    if source.name == "IL_Replay" and source.parent.name == "raw" and source.parent.parent.name == "data":
        inferred_source_project = source.parents[2]
        if (inferred_source_project / "src" / "semantic_frame" / "firstlight_full_episode.py").is_file():
            read_only_roots.append(inferred_source_project)
    checks = [
        {"name": "replay_dataset", "ready": (source / "replays").is_dir(), "path": str(source)},
        {"name": "firstlight_root", "ready": provider is not None and (provider / "native_runner" / "__init__.py").is_file(), "path": str(provider) if provider else None},
        {"name": "replay_tag", "ready": bool(args.replay_tag), "value": args.replay_tag},
        {"name": "native_endpoint", "ready": bool(args.host) and 1 <= args.port <= 65535, "host": args.host, "port": args.port},
    ]
    if any(output == root or output.is_relative_to(root) for root in read_only_roots) or (
        provider is not None and (output == provider or output.is_relative_to(provider))
    ):
        raise ValueError("capture output must be outside read-only inputs")
    report: dict[str, object] = {
        "schema": "elixir_echo.capture_preflight@1",
        "mode": "preflight_only",
        "ready": all(item["ready"] for item in checks),
        "checks": checks,
        "output": str(output),
        "native_reachability": "not_checked",
        "note": "Configuration only: Native reachability was not checked and no replay/output was created.",
    }
    if not args.run:
        return report
    missing = [item["name"] for item in checks if not item["ready"]]
    if missing:
        raise ValueError("capture --run requires: " + ", ".join(missing))

    from elixir_echo.integrations.firstlight.capture import (
        CaptureStageError,
        ReplaySourceError,
        capture_replay,
        load_replay_payload,
    )
    from elixir_echo.integrations.firstlight.replay import ReplayCaptureError

    try:
        payload = load_replay_payload(source, args.replay_tag)
        result = capture_replay(
            payload, firstlight_root=provider, output_root=output,
            host=args.host, port=args.port, read_only_roots=read_only_roots,
            debug_train=args.debug_train,
        )
    except (ReplaySourceError, ReplayCaptureError) as error:
        layer = "source" if isinstance(error, ReplaySourceError) else error.layer
        return {
            "schema": "elixir_echo.capture_failure@1",
            "mode": "failed",
            "replay_tag": args.replay_tag,
            "layer": layer,
            "stage": error.stage if isinstance(error, CaptureStageError) else "source_selection" if isinstance(error, ReplaySourceError) else "frame_or_cache",
            "reason": str(error),
        }
    return {"mode": "captured", **result.to_dict()}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="")


class _FixturePolicy:
    """Tiny dependency-free policy used only by the no-assets fixture."""

    def __init__(self) -> None:
        self.wait_bias = 0.0
        self.act_bias = 0.0

    def logits(self, policy_input: PolicyInput) -> tuple[float, float]:
        if "expert_action" in policy_input.tensors:
            raise ValueError("fixture policy input must not contain its target label")
        elixir = float(policy_input.tensors["elixir"])
        return self.wait_bias + 0.01 * elixir, self.act_bias

    def loss(self, policy_input: PolicyInput, target: str) -> tuple[float, tuple[float, float]]:
        if target not in {"wait", "act"}:
            raise ValueError("fixture target must be wait or act")
        logits = self.logits(policy_input)
        maximum = max(logits)
        log_normalizer = maximum + math.log(sum(math.exp(item - maximum) for item in logits))
        target_index = 0 if target == "wait" else 1
        return log_normalizer - logits[target_index], logits

    def sgd_step(self, policy_input: PolicyInput, target: str, *, learning_rate: float) -> float:
        loss, logits = self.loss(policy_input, target)
        maximum = max(logits)
        exponentials = tuple(math.exp(item - maximum) for item in logits)
        denominator = sum(exponentials)
        probabilities = tuple(item / denominator for item in exponentials)
        target_index = 0 if target == "wait" else 1
        gradients = [probabilities[0], probabilities[1]]
        gradients[target_index] -= 1.0
        self.wait_bias -= learning_rate * gradients[0]
        self.act_bias -= learning_rate * gradients[1]
        return loss


def _fixture(args: argparse.Namespace) -> dict[str, object]:
    config = _config(args)
    output = (args.output if args.output is not None else config.output_path / "fixture").expanduser().resolve()
    if output.exists():
        raise ValueError(f"fixture output already exists; choose a new path: {output}")
    episode_id = "synthetic-fixture-episode"
    context = {"source": "synthetic_fixture"}
    context_ref = content_sha256(context)
    metadata = {
        "episode_id": episode_id,
        "replay_tag": "synthetic-fixture",
        "outcome": "terminated",
        "split": "train",
        "episode_context_ref": context_ref,
        "episode_context": context,
    }
    writer = TrajectoryWriter(output / "trajectory").begin(metadata)
    for actor in (0, 1):
        identity = DecisionIdentity(episode_id, 90, actor)
        step = DecisionStep(
            identity=identity,
            episode_context_ref=context_ref,
            observation=Observation(
                match={"phase": "normal", "remaining_ms": 180_000},
                players=({"owner": actor, "relation": "self", "elixir": 5},),
                towers=(),
                entities=(),
                terminal={"ended": False},
            ),
            action_mask=ActionMask(identity, {"wait": True, "play_card": False}),
            expert_action=ExpertAction.wait(),
            event_delta=(),
            validation={"accepted": True},
            fingerprints={"source": "synthetic_fixture"},
        )
        writer.append(step)
    trajectory_manifest = writer.commit()
    training_episode = TrainingTrajectoryReader(output / "trajectory").read_episode(episode_id)
    training_step = training_episode.steps[0]
    tracker = PublicStateTracker()
    temporal = tracker.before_step(training_step)
    observation = training_step.observation
    self_player = next(player for player in observation.players if player.get("relation") == "self")
    policy_input = PolicyInput(
        identity=training_step.identity,
        profile="fixture-min",
        tensors={
            "remaining_ms": observation.match["remaining_ms"],
            "elixir": self_player["elixir"],
            "entity_count": len(observation.entities),
        },
        action_mask=training_step.action_mask.constraints,
        temporal_context=temporal,
    )

    # A tiny pure-Python policy computes a supervised cross-entropy gradient
    # and applies one SGD update. The label is passed only as the target.
    target = "wait" if training_step.expert_action.is_wait else "act"
    fake_policy = _FixturePolicy()
    before_loss, _ = fake_policy.loss(policy_input, target)
    fake_policy.sgd_step(policy_input, target, learning_rate=0.25)
    after_loss, _ = fake_policy.loss(policy_input, target)
    tracker.after_action(training_step.expert_action)

    fixture_manifest = {
        "schema": "elixir_echo.fixture-run@1",
        "source": "synthetic",
        "episode_id": episode_id,
        "profile": policy_input.profile,
        "supervised_steps": 1,
        "optimizer_updates": 1,
        "initial_loss": before_loss,
        "loss_after_one_update": after_loss,
        "fake_policy_parameters": {"wait_bias": fake_policy.wait_bias, "act_bias": fake_policy.act_bias},
        "trajectory_manifest": trajectory_manifest.to_dict(),
        "note": "This fixture proves the data/control path only; it is not a trained FirstLight checkpoint.",
    }
    _write_json(output / "fixture_manifest.json", fixture_manifest)
    # The checkpoint manifest is intentionally metadata-only and identifies a
    # synthetic fake policy, never a loadable model checkpoint.
    _write_json(
        output / "checkpoint_manifest.json",
        {
            "schema": "elixir_echo.fixture-checkpoint-manifest@1",
            "source": "synthetic_fake_policy",
            "loadable": False,
            "optimizer_updates": 1,
            "fixture_manifest_sha256": content_sha256(fixture_manifest),
        },
    )
    return {
        "schema": "elixir_echo.fixture-report@1",
        "mode": "fixture",
        "output": str(output),
        "episode_id": episode_id,
        "record_count": trajectory_manifest.record_count,
        "supervised_steps": 1,
        "optimizer_updates": 1,
        "initial_loss": before_loss,
        "loss_after_one_update": after_loss,
        "checkpoint_manifest": str(output / "checkpoint_manifest.json"),
        "dependencies": "standard library only; no external game runtime, game assets, or Torch",
    }


def _compile_preflight(args: argparse.Namespace) -> dict[str, object]:
    config = _config(args, dataset_path=args.root, firstlight_root=args.firstlight_root, max_episodes=args.max_episodes)
    summary = _verify_trajectory(config.dataset_path)
    if not args.run:
        return {
            "schema": "elixir_echo.compile_preflight@1",
            "mode": "verified_inputs_only",
            "dataset_path": str(config.dataset_path),
            "max_episodes": config.max_episodes,
            **{key: summary[key] for key in ("episode_count", "record_count", "outcome_counts")},
            "note": "No provider modules were loaded and no Layer 2 cache was written. Pass --run to compile the configured terminated train-split prefix.",
        }

    if config.firstlight_root is None:
        raise ValueError("Layer 2 compilation needs FirstLight; set --firstlight-root or ELIXIR_ECHO_FIRSTLIGHT_ROOT")
    cache_root = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (config.data_root / "layer2_sequence_cache").resolve()
    )
    if cache_root.exists():
        raise ValueError(f"Layer 2 output already exists; choose a new directory: {cache_root}")

    reader = TrainingTrajectoryReader(config.dataset_path)
    plan = plan_training_dataset(reader, max_episodes=config.max_episodes)
    if not plan.selected_episode_ids:
        raise ValueError("no terminated train-split episodes are eligible for Layer 2 compilation")

    # Provider dependencies are imported only after the explicit --run gate.
    from elixir_echo.datasets.sequence import Layer2SequenceCache, SequenceDatasetCompiler
    from elixir_echo.datasets.trajectory import TrajectoryReader
    from elixir_echo.integrations.firstlight.il import compile_firstlight_actor_sequence
    from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
    from elixir_echo.integrations.firstlight.sequence_cache import encode_actor_sequence
    from elixir_echo.integrations.firstlight.tensorizer import PolicyInputBuilder, load_v4_components
    from elixir_echo.profiles import ObservationProfile
    from elixir_echo.training.config import TrainingConfig

    runtime = FirstLightRuntime(config.firstlight_root)
    components = load_v4_components(runtime)
    training_config = TrainingConfig()
    input_builder = PolicyInputBuilder(runtime, components=components)
    input_contract_fingerprint = content_sha256(
        {
            "input_schema": training_config.to_dict()["input_schema"],
            "input_mode": training_config.input_mode,
            "profile": training_config.profile.value,
        }
    )

    def compile_actor(episode: Any, actor: int) -> Any:
        return compile_firstlight_actor_sequence(
            episode,
            actor=actor,
            input_builder=input_builder,
            components=components,
            config=training_config,
        )

    compiler = SequenceDatasetCompiler(
        cache=Layer2SequenceCache(cache_root),
        dataset_fingerprint=plan.fingerprint,
        training_config_fingerprint=training_config.fingerprint(),
        input_contract_fingerprint=input_contract_fingerprint,
        profile=ObservationProfile.RICH,
        actor_sequence_compiler=compile_actor,
        actor_sequence_encoder=lambda sequence: encode_actor_sequence(
            sequence, training_config=training_config
        ),
    )
    source_reader = TrajectoryReader(config.dataset_path)
    manifests = [
        compiler.compile_episode(source_reader.read_episode(episode_id))
        for episode_id in plan.selected_episode_ids
    ]
    return {
        "schema": "elixir_echo.layer2_compile@1",
        "mode": "compiled",
        "dataset_path": str(config.dataset_path),
        "output_path": str(cache_root),
        "dataset_fingerprint": plan.fingerprint,
        "training_config_fingerprint": training_config.fingerprint(),
        "profile": training_config.profile.value,
        "compiled_episode_count": len(manifests),
        "compiled_episode_ids": [item["episode_id"] for item in manifests],
        "note": "Only the configured terminated train-split prefix was compiled; no model was trained.",
    }


def _inspect_checkpoint(path: Path) -> dict[str, object]:
    directory = path.expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not directory.is_dir() or not manifest_path.is_file():
        raise ValueError(f"checkpoint directory or manifest.json is missing: {directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint manifest is unreadable: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest root must be an object")
    weights_name = manifest.get("weights_file")
    if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
        raise ValueError("checkpoint manifest has an unsafe weights_file")
    weights_path = directory / weights_name
    if not weights_path.is_file():
        raise ValueError(f"checkpoint weights file is missing: {weights_path.name}")
    digest_state = hashlib.sha256()
    with weights_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_state.update(chunk)
    digest = digest_state.hexdigest()
    expected = manifest.get("checkpoint_sha256")
    if not isinstance(expected, str) or digest != expected:
        raise ValueError("checkpoint weights hash does not match manifest")
    return {
        "schema": "elixir_echo.checkpoint_inspection@1",
        "status": "verified",
        "checkpoint_id": manifest.get("checkpoint_id"),
        "update_step": manifest.get("update_step"),
        "input_mode": manifest.get("input_mode"),
        "training_config_fingerprint": manifest.get("training_config_fingerprint"),
        "weights_file": weights_name,
        "checkpoint_sha256": digest,
        "loaded_into_model": False,
    }


def _evaluate_preflight(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = _inspect_checkpoint(args.checkpoint)
    dataset = args.dataset
    if dataset is None:
        dataset = _config(args).dataset_path
    if not (dataset.expanduser().resolve() / "episodes").is_dir():
        raise ValueError(f"held-out trajectory dataset has no episodes directory: {dataset.expanduser().resolve()}")
    return {
        "schema": "elixir_echo.evaluation_preflight@1",
        "mode": "preflight_only",
        "checkpoint": checkpoint,
        "dataset_path": str(dataset.expanduser().resolve()),
        "note": "Offline metrics and held-out split assertions are implemented in Task 8; no evaluation was run.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = _doctor(args)
        elif args.command == "fixture":
            report = _fixture(args)
        elif args.command == "capture":
            report = _capture_preflight(args)
        elif args.command == "compile":
            report = _compile_preflight(args)
        elif args.command == "verify-trajectory":
            config = _config(args, dataset_path=args.root)
            report = _verify_trajectory(config.dataset_path)
        elif args.command in {"train", "train-firstlight"}:
            report = _train_firstlight(args)
        elif args.command == "inspect-checkpoint":
            report = _inspect_checkpoint(args.path)
        elif args.command == "evaluate":
            report = _evaluate_preflight(args)
        else:  # pragma: no cover - argparse enforces known commands
            raise ValueError(f"unsupported command: {args.command}")
    except (ConfigError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"elixir-echo: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if args.command == "capture" and report.get("mode") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
