"""Port the validated replay reconstruction sequence into ElixirEcho.

The source selection and Native call order come from the earlier local
``semantic_frame.firstlight_full_episode`` implementation. Current-state
translation and episode storage use ElixirEcho's existing contracts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping

from elixir_echo.domain.values import content_sha256
from elixir_echo.integrations.firstlight.observation import FirstLightObservationAdapter
from elixir_echo.integrations.firstlight.replay import (
    DECISION_INTERVAL_TICKS,
    FIRST_DECISION_TICK,
    FirstLightReplayProvider,
    ReplayCaptureError,
    group_timed_expert_actions,
)
from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime


_DEBUG_RUN_ROOT = Path(__file__).resolve().parents[4] / "runs" / "real-replay-chain"


class ReplaySourceError(ValueError):
    """The requested source replay is absent, ambiguous, or unsuitable."""


class CaptureStageError(ReplayCaptureError):
    """A failed Native stage with a stable failure layer for reporting."""

    def __init__(self, layer: str, stage: str, reason: str) -> None:
        self.layer = layer
        self.stage = stage
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class CaptureResult:
    replay_tag: str
    episode_id: str
    output_path: Path
    outcome: str
    decision_tick_count: int
    record_count: int
    label_leakage_checks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "elixir_echo.capture_result@1",
            "replay_tag": self.replay_tag,
            "episode_id": self.episode_id,
            "output_path": str(self.output_path),
            "outcome": self.outcome,
            "decision_tick_count": self.decision_tick_count,
            "record_count": self.record_count,
            "label_leakage_checks": self.label_leakage_checks,
        }


def replay_tag(payload: Mapping[str, Any]) -> str:
    source = payload.get("source")
    source_tag = source.get("replay_tag") if isinstance(source, Mapping) else None
    return str(payload.get("replay_tag") or payload.get("replayTag") or source_tag or "").strip()


def preflight_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the earlier project's source-only decision-window acceptance rule."""

    windows: dict[tuple[str, int], int] = {}
    data_i_values: set[int] = set()
    action_count = early = 0
    events = payload.get("events")
    if not isinstance(events, list):
        events = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("kind") not in {"play_card", "activate_ability"}:
            continue
        tick = event.get("replay_tick_20hz")
        side = event.get("side")
        if type(tick) is not int or side not in {"team", "opponent"}:
            continue
        action_count += 1
        source_fields = event.get("source_fields")
        if isinstance(source_fields, Mapping) and type(source_fields.get("data_i")) is int:
            data_i_values.add(source_fields["data_i"])
        if tick < FIRST_DECISION_TICK:
            early += 1
            continue
        decision_tick = FIRST_DECISION_TICK + ((tick - FIRST_DECISION_TICK) // DECISION_INTERVAL_TICKS) * DECISION_INTERVAL_TICKS
        key = (str(side), decision_tick)
        windows[key] = windows.get(key, 0) + 1
    data_i = next(iter(data_i_values)) if len(data_i_values) == 1 else None
    overflow = sum(count > 2 for count in windows.values())
    return {
        "replay_tag": replay_tag(payload),
        "action_count": action_count,
        "early_action_count": early,
        "overflow_window_count": overflow,
        "data_i": data_i,
        "accepted": bool(replay_tag(payload)) and action_count > 0 and early == 0 and overflow == 0 and data_i in (0, 1),
    }


def iter_replay_payloads(dataset_root: Path) -> Iterator[Mapping[str, Any]]:
    """Read only the source payload column; pyarrow is needed only to capture."""

    try:
        import pyarrow.dataset as arrow_dataset
    except ImportError as error:
        raise ReplaySourceError("replay capture needs pyarrow; install the firstlight-capture extra") from error
    replay_root = Path(dataset_root).resolve() / "replays"
    if not replay_root.is_dir():
        raise ReplaySourceError(f"missing source replay directory: {replay_root}")
    scanner = arrow_dataset.dataset(replay_root, format="parquet").scanner(
        columns=["payload_json"], batch_size=64, use_threads=False
    )
    for batch in scanner.to_batches():
        for raw in batch.column("payload_json").to_pylist():
            payload = json.loads(str(raw))
            if isinstance(payload, Mapping):
                yield payload


def select_replay_payload(payloads: Iterable[Mapping[str, Any]], requested_tag: str) -> Mapping[str, Any]:
    if not requested_tag or not requested_tag.strip():
        raise ReplaySourceError("an exact replay tag is required")
    selected: Mapping[str, Any] | None = None
    for payload in payloads:
        if replay_tag(payload) != requested_tag:
            continue
        if selected is not None:
            raise ReplaySourceError(f"duplicate source replay identity: {requested_tag}")
        selected = payload
    if selected is None:
        raise ReplaySourceError(f"requested replay is absent: {requested_tag}")
    report = preflight_payload(selected)
    if not report["accepted"]:
        raise ReplaySourceError(f"source replay failed preflight: {report}")
    return selected


def load_replay_payload(dataset_root: Path, requested_tag: str) -> Mapping[str, Any]:
    """Push exact identity selection into Parquet before decoding payload JSON."""

    if not requested_tag or not requested_tag.strip():
        raise ReplaySourceError("an exact replay tag is required")
    try:
        import pyarrow.dataset as arrow_dataset
    except ImportError as error:
        raise ReplaySourceError("replay capture needs pyarrow; install the firstlight-capture extra") from error
    replay_root = Path(dataset_root).resolve() / "replays"
    if not replay_root.is_dir():
        raise ReplaySourceError(f"missing source replay directory: {replay_root}")
    dataset = arrow_dataset.dataset(replay_root, format="parquet")
    if "replay_tag" not in dataset.schema.names:
        raise ReplaySourceError("replay dataset has no replay_tag column")
    scanner = dataset.scanner(
        columns=["payload_json"],
        filter=arrow_dataset.field("replay_tag") == requested_tag,
        batch_size=64,
        use_threads=False,
    )

    def matching_payloads():
        for batch in scanner.to_batches():
            for raw in batch.column("payload_json").to_pylist():
                payload = json.loads(str(raw))
                if not isinstance(payload, Mapping):
                    raise ReplaySourceError("matching replay payload is not a JSON object")
                yield payload

    return select_replay_payload(matching_payloads(), requested_tag)


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    base = root.resolve()
    return {
        file.relative_to(base).as_posix(): (file.stat().st_size, file.stat().st_mtime_ns)
        for file in base.rglob("*") if file.is_file()
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _provider_imports(runtime: FirstLightRuntime):
    """Keep the configured checkout on the import path during lazy Native use."""

    old_path = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime.root))
    try:
        yield
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


class _HeadlessCalibrationProbe:
    def __init__(self, native: Any) -> None:
        self.native = native

    def create_native_match(self, config: Any, *, wait_timeout: float = 30.0) -> Mapping[str, Any]:
        del wait_timeout
        return self.native.create_match(config)

    @staticmethod
    def pause() -> Mapping[str, Any]:
        return {"ok": True, "paused": True, "mode": "headless"}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)


def _wait_group(action_type: Any, ticks: int) -> dict[int, tuple[Any, ...]]:
    return {owner: (action_type.wait(owner, ticks=ticks),) for owner in (0, 1)}


def _guard_output(output_root: Path, *read_only_roots: Path) -> None:
    output = output_root.resolve()
    for root in read_only_roots:
        source = root.resolve()
        if output == source or output.is_relative_to(source):
            raise ValueError(f"capture output must be outside read-only input root: {source}")


def capture_replay(
    payload: Mapping[str, Any],
    *,
    firstlight_root: Path,
    output_root: Path,
    host: str = "127.0.0.1",
    port: int = 26789,
    calibration_attempts: int = 10,
    read_only_roots: Iterable[Path] = (),
    debug_train: bool = False,
) -> CaptureResult:
    """Reconstruct one fixed source replay and atomically publish its trajectory."""

    source_tag = replay_tag(payload)
    report = preflight_payload(payload)
    if not report["accepted"]:
        raise ReplaySourceError(f"source replay failed preflight: {report}")
    if not 1 <= port <= 65535 or not 1 <= calibration_attempts <= 12:
        raise ValueError("invalid Native port or calibration attempt count")
    runtime = FirstLightRuntime(firstlight_root)
    _guard_output(output_root, runtime.root, *read_only_roots)
    if debug_train:
        debug_root = _DEBUG_RUN_ROOT.resolve()
        output = output_root.resolve()
        if output == debug_root or not output.is_relative_to(debug_root):
            raise ValueError("debug capture output must be inside runs/real-replay-chain/<run-id>")
    before = _tree_snapshot(runtime.root)
    native = None
    failure: BaseException | None = None
    captured = context = fingerprints = None
    stage = "provider_import"
    try:
        with _provider_imports(runtime):
            battle_module = runtime.import_module("native_runner.battle_env")
            contracts = runtime.import_module("native_runner.contracts")
            native_module = runtime.import_module("native_runner.cr_native_env")
            forms_module = runtime.import_module("native_runner.normal_form_evidence")
            replay_module = runtime.import_module("native_runner.royaleapi_replay")
            viewer_module = runtime.import_module("native_runner.training.replay_viewer")

            stage = "native_connect"
            native = native_module.NativeClashEnv(host, port, timeout=60.0)
            native.status()
            native.stop_resident_mode()
            stage = "prepare_calibrate"
            prepared = replay_module.prepare_collected_replay(payload)
            calibrated = replay_module.calibrate_collected_replay_deal(
                prepared, _HeadlessCalibrationProbe(native), max_attempts=calibration_attempts
            )
            replay = calibrated.replay
            allowed_ids = {source_tag, f"royaleapi-{source_tag}"}
            if str(replay.episode_id) not in allowed_ids:
                raise ReplaySourceError("calibrated Provider episode identity differs from the requested replay")
            stage = "battle_reset"
            source = replay.episode_config.environment
            execution_environment = replace(source, max_battle_ticks=replay.end_native_tick)
            ruleset_id = contracts.content_hash({
                "source_ruleset_id": replay.ruleset_id,
                "environment": execution_environment.to_dict(),
            })
            environment = battle_module.BattleEnvV1(
                native=native, ruleset_id=ruleset_id, warmup_ticks=source.warmup_ticks,
                max_battle_ticks=replay.end_native_tick, entity_limit=source.entity_limit,
                event_limit=source.event_limit, event_window_ticks=source.event_window_ticks,
                shaping_beta=source.shaping_beta, shaping_tau_ticks=source.shaping_tau_ticks,
                include_native_digest=source.include_native_digest,
            )
            execution_episode = replace(replay.episode_config, environment=execution_environment, ruleset_id=ruleset_id)
            observations, _ = environment.reset(execution_episode, options={
                "render_mode": "headless", "defer_headless_warmup": True,
                "headless_initial_step_ticks": 1,
                "allow_initial_remaining_runtime_rejections": True,
                "allow_verified_standard_layout_alias": True,
                "decision_ticks": DECISION_INTERVAL_TICKS, "event_driven_decisions": False,
            })
            current_tick = int(observations[0].tick)
            if current_tick < FIRST_DECISION_TICK:
                observations, _, terminated, truncated, _ = environment.step(
                    _wait_group(contracts.ActionV1, FIRST_DECISION_TICK - current_tick), record_trace=False
                )
                if terminated[0] or truncated[0]:
                    raise ReplayCaptureError("battle ended during opening warmup")
                current_tick = int(observations[0].tick)
            if current_tick != FIRST_DECISION_TICK:
                raise ReplayCaptureError(f"expected first decision tick {FIRST_DECISION_TICK}, got {current_tick}")
            commands = viewer_module._direct_render_commands(replay)
            if any(command.target_tick <= current_tick for command in commands):
                raise ReplayCaptureError("replay command precedes first decision input")
            viewer_module._schedule_direct_render_commands(native, commands)

            stage = "frame_capture"
            bindings = forms_module.NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID
            adapter = FirstLightObservationAdapter(runtime_hero_forms={
                int(binding.runtime_form_card_id): (int(binding.source_card_id), str(binding.hero_form_id))
                for binding in bindings.values()
            })
            context = {
                "episode_id": source_tag,
                "deck0": list(replay.episode_config.deck0),
                "deck1": list(replay.episode_config.deck1),
                "reproduction": {
                    "ruleset_id": ruleset_id, "seed": replay.episode_config.seed,
                    "replay_tag": calibrated.replay_tag,
                    "provider_episode_id": replay.episode_id,
                },
            }
            context_ref = content_sha256(context)
            fingerprints = {
                "firstlight_contracts": _file_hash(runtime.root / "native_runner" / "contracts.py"),
                "firstlight_battle_env": _file_hash(runtime.root / "native_runner" / "battle_env.py"),
                "adapter_source": _file_hash(Path(__file__).with_name("observation.py")),
                "capture_source": _file_hash(Path(__file__)),
                "fairness_profile": "actor_fair", "ruleset_id": ruleset_id,
                "decision_ticks": DECISION_INTERVAL_TICKS,
            }

            def step_wait(ticks: int):
                return environment.step(_wait_group(contracts.ActionV1, ticks), record_trace=False)

            bound = max(1, ((int(replay.end_native_tick) - FIRST_DECISION_TICK) // DECISION_INTERVAL_TICKS) + 2)
            provider = FirstLightReplayProvider(adapter)
            captured = provider.capture_timeline(
                episode_id=source_tag, episode_context_ref=context_ref,
                initial_observations=observations, expert_actions=group_timed_expert_actions(replay),
                step_wait=step_wait, fingerprints=fingerprints, max_decision_steps=bound,
            )
    except BaseException as error:
        failure = error
    finally:
        if native is not None:
            try:
                native.pause()
            except Exception:
                pass
    after = _tree_snapshot(runtime.root)
    if before != after:
        raise CaptureStageError("infrastructure", "provider_read_only_guard", "FirstLight checkout changed during replay capture") from failure
    if failure is not None:
        if isinstance(failure, (ReplaySourceError, ReplayCaptureError, KeyboardInterrupt, SystemExit)):
            raise failure
        layer = "infrastructure" if stage in {"provider_import", "native_connect"} else "upstream_reconstruction"
        raise CaptureStageError(layer, stage, str(failure)) from failure
    assert captured is not None and context is not None and fingerprints is not None
    manifest = FirstLightReplayProvider.write_trajectory(
        captured, root=str(output_root), replay_tag=source_tag,
        episode_context=context, fingerprints=fingerprints,
        split="train" if debug_train else None,
    )
    from elixir_echo.datasets.trajectory import TrajectoryReader

    return CaptureResult(
        replay_tag=source_tag, episode_id=source_tag,
        output_path=TrajectoryReader(output_root).episode_path(source_tag),
        outcome=captured.outcome, decision_tick_count=captured.decision_tick_count,
        record_count=manifest.record_count, label_leakage_checks=captured.label_leakage_checks,
    )
