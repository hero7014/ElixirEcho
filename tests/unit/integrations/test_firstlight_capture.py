from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from elixir_echo.datasets.trajectory import TrajectoryReader
from elixir_echo.domain import Observation
from elixir_echo.integrations.firstlight import capture


def _payload(tag: str = "r1") -> dict:
    return {
        "replay_tag": tag,
        "events": [{
            "kind": "play_card", "side": "team", "replay_tick_20hz": 92,
            "source_fields": {"data_i": 0},
        }],
    }


def test_exact_replay_selection_never_substitutes_or_accepts_duplicates():
    assert capture.select_replay_payload((_payload("other"), _payload()), "r1")["replay_tag"] == "r1"
    with pytest.raises(capture.ReplaySourceError, match="absent"):
        capture.select_replay_payload((_payload("other"),), "r1")
    with pytest.raises(capture.ReplaySourceError, match="duplicate"):
        capture.select_replay_payload((_payload(), _payload()), "r1")
    invalid = _payload()
    invalid["events"][0]["replay_tick_20hz"] = 80
    with pytest.raises(capture.ReplaySourceError, match="preflight"):
        capture.select_replay_payload((invalid,), "r1")


@dataclass(frozen=True)
class _EnvironmentSpec:
    max_battle_ticks: int = 100
    warmup_ticks: int = 0
    entity_limit: int = 64
    event_limit: int = 64
    event_window_ticks: int = 5
    shaping_beta: float = 0.0
    shaping_tau_ticks: int = 1
    include_native_digest: bool = False

    def to_dict(self):
        return {"max_battle_ticks": self.max_battle_ticks}


@dataclass(frozen=True)
class _EpisodeConfig:
    environment: _EnvironmentSpec
    deck0: tuple[int, ...] = (1, 2)
    deck1: tuple[int, ...] = (3, 4)
    seed: int = 7
    ruleset_id: str = "rules"


class _Native:
    calls: list[str] = []

    def __init__(self, *_args, **_kwargs):
        self.calls = []

    def status(self):
        self.calls.append("status")

    def stop_resident_mode(self):
        self.calls.append("stop_resident")

    def pause(self):
        self.calls.append("pause")


def _observation(actor: int, tick: int):
    return SimpleNamespace(
        owner=actor, tick=tick, events=(),
        action_mask={"identity": {"tick": tick, "actor": actor}, "kinds": {"wait": True, "play_card": True}},
    )


class _BattleEnv:
    def __init__(self, **_kwargs):
        self.tick = 90

    def reset(self, _episode, *, options):
        assert options["decision_ticks"] == 5
        return {actor: _observation(actor, 90) for actor in (0, 1)}, {}

    def step(self, _actions, *, record_trace):
        assert record_trace is False
        self.tick += 5
        return (
            {actor: _observation(actor, self.tick) for actor in (0, 1)},
            {}, {0: True, 1: True}, {0: False, 1: False},
            {actor: {"terminal_reason": "fixture"} for actor in (0, 1)},
        )


class _Adapter:
    def adapt(self, source, *, actor):
        return Observation(
            match={"phase": "normal", "remaining_ms": 120_000 - source.tick * 100},
            players=({"owner": actor, "relation": "self"},),
            towers=(), entities=(), terminal={"ended": False},
        )


class _Runtime:
    def __init__(self, root: Path, modules: dict):
        self.root = root
        self.modules = modules

    def import_module(self, name: str):
        return self.modules[name]


def _runtime_fixture(tmp_path: Path, *, provider_episode_id="r1"):
    root = tmp_path / "provider"
    native_root = root / "native_runner"
    native_root.mkdir(parents=True)
    for filename in ("__init__.py", "contracts.py", "battle_env.py"):
        (native_root / filename).write_text("# fake provider\n", encoding="utf-8")
    action = SimpleNamespace(
        kind="play_card", owner=0, hand_slot=1, card_id=26_000_001,
        source_entity=None, ability_id=None, target_kind="grid", target_grid=(4, 8),
        target_entity=None, subcell_offset=None, action_id="expert-1",
        metadata={"source_command_tick": 92, "source_index": 0, "native_observable_tick": 93},
    )
    replay = SimpleNamespace(
        episode_id=provider_episode_id, end_native_tick=100, ruleset_id="rules",
        episode_config=_EpisodeConfig(_EnvironmentSpec()),
        operations=(SimpleNamespace(actions=(action,)),),
    )
    modules = {
        "native_runner.battle_env": SimpleNamespace(BattleEnvV1=_BattleEnv),
        "native_runner.contracts": SimpleNamespace(ActionV1=SimpleNamespace(wait=lambda owner, ticks: (owner, ticks)), content_hash=lambda obj: "rules-hash"),
        "native_runner.cr_native_env": SimpleNamespace(NativeClashEnv=_Native),
        "native_runner.normal_form_evidence": SimpleNamespace(NORMAL_MODE_HERO_FORM_BINDINGS_BY_CARD_ID={}),
        "native_runner.royaleapi_replay": SimpleNamespace(
            prepare_collected_replay=lambda payload: payload,
            calibrate_collected_replay_deal=lambda _prepared, _probe, max_attempts: SimpleNamespace(replay=replay, replay_tag="r1"),
        ),
        "native_runner.training.replay_viewer": SimpleNamespace(
            _direct_render_commands=lambda _replay: (SimpleNamespace(target_tick=92),),
            _schedule_direct_render_commands=lambda _native, _commands: None,
        ),
    }
    return _Runtime(root, modules)


def test_old_native_call_order_feeds_existing_atomic_trajectory_writer(tmp_path, monkeypatch):
    runtime = _runtime_fixture(tmp_path)
    monkeypatch.setattr(capture, "FirstLightRuntime", lambda _root: runtime)
    monkeypatch.setattr(capture, "FirstLightObservationAdapter", lambda **_kwargs: _Adapter())
    output = tmp_path / "trajectories"

    result = capture.capture_replay(_payload(), firstlight_root=runtime.root, output_root=output)

    assert result.episode_id == "r1"
    assert result.outcome == "terminated"
    assert result.decision_tick_count == 1
    assert result.record_count == result.label_leakage_checks == 2
    episode = TrajectoryReader(output).read_episode("r1")
    assert [(step.identity.tick, step.identity.actor) for step in episode.records] == [(90, 0), (90, 1)]
    assert episode.records[0].expert_action.actions[0]["kind"] == "play_card"
    assert "expert-1" not in str(episode.records[0].observation.to_dict())


def test_provider_identity_mismatch_fails_before_any_episode_is_written(tmp_path, monkeypatch):
    runtime = _runtime_fixture(tmp_path, provider_episode_id="wrong")
    monkeypatch.setattr(capture, "FirstLightRuntime", lambda _root: runtime)
    output = tmp_path / "trajectories"
    with pytest.raises(capture.ReplaySourceError, match="identity"):
        capture.capture_replay(_payload(), firstlight_root=runtime.root, output_root=output)
    assert not output.exists()


def test_calibration_failure_is_reported_as_upstream_and_leaves_no_episode(tmp_path, monkeypatch):
    runtime = _runtime_fixture(tmp_path)
    runtime.modules["native_runner.royaleapi_replay"].calibrate_collected_replay_deal = (
        lambda _prepared, _probe, max_attempts: (_ for _ in ()).throw(RuntimeError("calibration failed"))
    )
    monkeypatch.setattr(capture, "FirstLightRuntime", lambda _root: runtime)
    output = tmp_path / "trajectories"
    with pytest.raises(capture.CaptureStageError) as failure:
        capture.capture_replay(_payload(), firstlight_root=runtime.root, output_root=output)
    assert failure.value.layer == "upstream_reconstruction"
    assert failure.value.stage == "prepare_calibrate"
    assert not output.exists()


def test_capture_output_cannot_write_into_provider_or_source_tree(tmp_path, monkeypatch):
    runtime = _runtime_fixture(tmp_path)
    monkeypatch.setattr(capture, "FirstLightRuntime", lambda _root: runtime)
    with pytest.raises(ValueError, match="read-only"):
        capture.capture_replay(_payload(), firstlight_root=runtime.root, output_root=runtime.root / "output")
