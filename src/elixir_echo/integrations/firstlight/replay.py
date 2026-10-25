"""Continuous, actor-paired replay capture behind the FirstLight boundary.

Native construction and replay calibration are injected by the caller.  This
module owns the deterministic decision clock, causal label boundary, and
conversion to the project-owned trajectory store.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from elixir_echo.datasets.trajectory import TrajectoryManifest, TrajectoryWriter
from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation
from elixir_echo.domain.values import canonical_json


FIRST_DECISION_TICK = 90
DECISION_INTERVAL_TICKS = 5


class ReplayCaptureError(RuntimeError):
    """Base class for errors classified at the replay integration boundary."""

    layer = "upstream_reconstruction"


class ReplayReconstructionError(ReplayCaptureError):
    """The upstream replay clock or Native step did not reconstruct reliably."""


class ReplayAdapterError(ReplayCaptureError):
    """A FirstLight observation could not satisfy the frozen domain contract."""

    layer = "adapter_semantic"


class ReplayLabelLeakageError(ReplayCaptureError):
    """A current label or its future effect entered the current input."""

    layer = "adapter_semantic"


class ReplayCacheError(ReplayCaptureError):
    """A validated capture could not be committed to the trajectory cache."""

    layer = "cache_failure"


@dataclass(frozen=True, slots=True)
class TimedExpertAction:
    decision_tick: int
    source_tick: int
    observable_tick: int
    actor: int
    action_id: str
    action: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CapturedTrajectory:
    episode_id: str
    records: tuple[DecisionStep, ...]
    outcome: str
    outcome_detail: Mapping[str, Any]
    decision_tick_count: int
    label_leakage_checks: int


def _plain_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise ReplayAdapterError(f"expected a mapping-compatible value, got {type(value).__name__}")


def _mask_for(source: Any, identity: DecisionIdentity) -> ActionMask:
    raw = _plain_mapping(getattr(source, "action_mask", None))
    source_identity = raw.get("identity")
    if source_identity is not None:
        expected_owner = getattr(source, "owner", identity.actor)
        expected_tick = getattr(source, "tick", identity.tick)
        if source_identity.get("tick") not in (None, expected_tick) or source_identity.get("actor") not in (None, expected_owner):
            raise ReplayAdapterError("FirstLight action mask identity does not match its observation")
    constraints = {key: value for key, value in raw.items() if key not in {"schema", "identity"}}
    return ActionMask(identity, constraints)


def _event_mapping(event: Any, actor: int) -> Mapping[str, Any]:
    def get_value(item: Any, name: str, default: Any = None) -> Any:
        return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)

    def value_of(item: Any) -> Any:
        return getattr(item, "value", item)

    result: dict[str, Any] = {}
    for name in ("tick", "owner", "entity_id", "card_id", "position"):
        item = get_value(event, name)
        if item is not None:
            result[name] = item
    event_type = get_value(event, "event_type", get_value(event, "kind"))
    if event_type is not None:
        result["event_type"] = str(value_of(event_type))

    combat = get_value(event, "combat")
    if combat is not None:
        provenance = get_value(combat, "provenance")
        evidence = get_value(provenance, "field_evidence", {})
        allowed = (
            "kind", "source_entity", "source_card_id", "target_entity", "target_card_id",
            "position", "amount", "effect_id", "projectile_id", "ability_id",
            "evolution_form_id", "visible_by_owner",
        )
        public: dict[str, Any] = {}
        for name in allowed:
            state = value_of(evidence.get(name, "unknown")) if isinstance(evidence, Mapping) else "unknown"
            if state in {"unknown", "not_applicable"}:
                continue
            item = get_value(combat, name)
            if name == "visible_by_owner" and isinstance(item, Mapping):
                item = item.get(str(actor))
            if item is not None:
                public[name] = value_of(item)
        if public:
            result["combat"] = public
    return result


def _event_tick(event: Any) -> int:
    value = event.get("tick", -1) if isinstance(event, Mapping) else getattr(event, "tick", -1)
    return int(value)


def _sanitized_action(action: Any, *, decision_tick: int, source_tick: int) -> dict[str, Any]:
    raw = {
        "owner": getattr(action, "owner", None),
        "kind": getattr(getattr(action, "kind", None), "value", getattr(action, "kind", None)),
        "hand_slot": getattr(action, "hand_slot", None),
        "card_id": getattr(action, "card_id", None),
        "source_entity": getattr(action, "source_entity", None),
        "ability_id": getattr(action, "ability_id", None),
        "target_kind": getattr(getattr(action, "target_kind", None), "value", getattr(action, "target_kind", None)),
        "target_grid": getattr(action, "target_grid", None),
        "target_entity": getattr(action, "target_entity", None),
        "subcell_offset": getattr(action, "subcell_offset", None),
        "execute_offset_ticks": source_tick - decision_tick,
    }
    return {key: value for key, value in raw.items() if value is not None}


def group_timed_expert_actions(replay: Any) -> dict[tuple[int, int], tuple[TimedExpertAction, ...]]:
    """Translate replay actions to their causal five-tick decision windows."""

    grouped: dict[tuple[int, int], list[tuple[int, int, TimedExpertAction]]] = {}
    for operation in replay.operations:
        for action in operation.actions:
            kind = str(getattr(getattr(action, "kind", None), "value", getattr(action, "kind", "")))
            if kind not in {"play_card", "activate_ability"}:
                continue
            metadata = getattr(action, "metadata", {})
            source_tick = metadata.get("source_command_tick")
            source_index = metadata.get("source_index", 0)
            observable_tick = metadata.get("native_observable_tick", source_tick + 1 if type(source_tick) is int else None)
            if type(source_tick) is not int or source_tick < FIRST_DECISION_TICK:
                raise ValueError("expert action has an invalid source command tick")
            if type(source_index) is not int or source_index < 0 or type(observable_tick) is not int:
                raise ValueError("expert action timing metadata is invalid")
            actor = int(action.owner)
            if actor not in (0, 1):
                raise ValueError("expert action owner must be 0 or 1")
            decision_tick = FIRST_DECISION_TICK + ((source_tick - FIRST_DECISION_TICK) // DECISION_INTERVAL_TICKS) * DECISION_INTERVAL_TICKS
            item = TimedExpertAction(
                decision_tick=decision_tick,
                source_tick=source_tick,
                observable_tick=observable_tick,
                actor=actor,
                action_id=str(getattr(action, "action_id", "") or ""),
                action=_sanitized_action(action, decision_tick=decision_tick, source_tick=source_tick),
            )
            grouped.setdefault((decision_tick, actor), []).append((source_tick, source_index, item))
    result = {}
    for key, values in grouped.items():
        if len(values) > 2:
            raise ValueError(f"expert decision window {key} contains more than two actions")
        result[key] = tuple(item for _tick, _index, item in sorted(values, key=lambda value: (value[0], value[1])))
    return result


def _nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _nested_strings(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nested_strings(item)


def assert_no_label_leakage(
    previous_tick: int,
    tick: int,
    actor: int,
    observation: Observation,
    action_mask: ActionMask,
    event_delta: Sequence[Mapping[str, Any]],
    labels: Sequence[TimedExpertAction],
) -> None:
    """Assert input data is causally prior to the current action label."""

    for label in labels:
        if label.decision_tick != tick or label.actor != actor:
            raise ReplayLabelLeakageError("expert label is bound to a different decision identity")
        if not tick <= label.source_tick < tick + DECISION_INTERVAL_TICKS:
            raise ReplayLabelLeakageError("expert action source tick is outside its decision window")
        if label.observable_tick <= tick:
            raise ReplayLabelLeakageError("current expert action is observable at/before its input snapshot")
    for event in event_delta:
        event_tick = event.get("tick")
        if type(event_tick) is not int or not previous_tick < event_tick <= tick:
            raise ReplayLabelLeakageError("event delta is outside the prior-open/current-closed interval")
    input_payload = {
        "observation": observation.to_dict(),
        "action_mask": action_mask.to_dict(),
        "event_delta": event_delta,
    }
    input_text = canonical_json(input_payload).casefold()
    if "expert_action" in input_text or "expert action" in input_text:
        raise ReplayLabelLeakageError("expert action field leaked into current policy input")
    identifiers = {label.action_id for label in labels if label.action_id}
    evidence = set(_nested_strings(event_delta))
    if identifiers & evidence:
        raise ReplayLabelLeakageError("current expert action identity leaked into current event delta")
    if any(action_id.casefold() in input_text for action_id in identifiers):
        raise ReplayLabelLeakageError("current expert action identity leaked into current input")


class FirstLightReplayProvider:
    """Capture both actor views every five ticks from a prepared Native replay."""

    def __init__(self, adapter: Any, *, event_sanitizer: Callable[[Any, int], Mapping[str, Any]] = _event_mapping) -> None:
        self.adapter = adapter
        self.event_sanitizer = event_sanitizer

    def capture_timeline(
        self,
        *,
        episode_id: str,
        episode_context_ref: str,
        initial_observations: Mapping[int, Any],
        expert_actions: Mapping[tuple[int, int], Sequence[TimedExpertAction]],
        step_wait: Callable[[int], tuple[Mapping[int, Any], Any, Mapping[int, bool], Mapping[int, bool], Mapping[int, Mapping[str, Any]]]],
        fingerprints: Mapping[str, Any],
        max_decision_steps: int | None = None,
    ) -> CapturedTrajectory:
        observations = dict(initial_observations)
        if set(observations) != {0, 1}:
            raise ReplayReconstructionError("capture requires observations for exactly actors 0 and 1")
        if any(int(getattr(item, "tick", -1)) != FIRST_DECISION_TICK for item in observations.values()):
            raise ReplayReconstructionError(f"full episode must start at tick {FIRST_DECISION_TICK}")
        if max_decision_steps is not None and max_decision_steps < 1:
            raise ValueError("max_decision_steps must be positive or None")

        records: list[DecisionStep] = []
        previous_by_actor: dict[int, int] = {0: FIRST_DECISION_TICK - DECISION_INTERVAL_TICKS, 1: FIRST_DECISION_TICK - DECISION_INTERVAL_TICKS}
        decision_ticks = 0
        leakage_checks = 0
        while True:
            if max_decision_steps is not None and decision_ticks >= max_decision_steps:
                raise ReplayReconstructionError("reconstruction exceeded trusted decision-step bound")
            tick_values = {int(getattr(item, "tick", -1)) for item in observations.values()}
            if len(tick_values) != 1:
                raise ReplayReconstructionError("actor observations disagree on current tick")
            tick = tick_values.pop()
            if tick != FIRST_DECISION_TICK + decision_ticks * DECISION_INTERVAL_TICKS:
                raise ReplayReconstructionError("decision clock is not a continuous five-tick sequence")

            for actor in (0, 1):
                source = observations[actor]
                if int(getattr(source, "owner", -1)) != actor:
                    raise ReplayReconstructionError("FAIR observation owner differs from actor")
                identity = DecisionIdentity(episode_id, tick, actor)
                try:
                    current_observation = self.adapter.adapt(source, actor=actor)
                    if not isinstance(current_observation, Observation):
                        raise TypeError("adapter did not return canonical Observation")
                    mask = _mask_for(source, identity)
                except Exception as error:
                    raise ReplayAdapterError(f"adapter/mask rejected tick={tick}, actor={actor}: {error}") from error
                prior_tick = previous_by_actor[actor]
                raw_events = tuple(
                    event for event in getattr(source, "events", ())
                    if prior_tick < _event_tick(event) <= tick
                )
                try:
                    event_delta = tuple(self.event_sanitizer(event, actor) for event in raw_events)
                except Exception as error:
                    raise ReplayAdapterError(f"public event adaptation failed at tick={tick}, actor={actor}: {error}") from error
                labels = tuple(expert_actions.get((tick, actor), ()))
                assert_no_label_leakage(prior_tick, tick, actor, current_observation, mask, event_delta, labels)
                records.append(DecisionStep(
                    identity=identity,
                    episode_context_ref=episode_context_ref,
                    observation=current_observation,
                    action_mask=mask,
                    expert_action=ExpertAction(tuple(label.action for label in labels)),
                    event_delta=event_delta,
                    validation={
                        "accepted": True,
                        "atomic_tick": tick,
                        "event_interval": {"after_tick": prior_tick, "through_tick": tick},
                        "label_leakage_asserted": True,
                    },
                    fingerprints=fingerprints,
                ))
                previous_by_actor[actor] = tick
                leakage_checks += 1

            decision_ticks += 1
            try:
                next_observations, _rewards, terminated, truncated, infos = step_wait(DECISION_INTERVAL_TICKS)
            except Exception as error:
                raise ReplayReconstructionError(f"FirstLight replay step failed after tick {tick}: {error}") from error
            term_values = {bool(terminated.get(actor, False)) for actor in (0, 1)}
            trunc_values = {bool(truncated.get(actor, False)) for actor in (0, 1)}
            if len(term_values) != 1 or len(trunc_values) != 1:
                raise ReplayReconstructionError("actors disagree on terminated/truncated outcome")
            terminated_value = term_values.pop()
            truncated_value = trunc_values.pop()
            if terminated_value and truncated_value:
                raise ReplayReconstructionError("episode cannot be both terminated and truncated")
            if terminated_value or truncated_value:
                final_ticks = {int(getattr(item, "tick", -1)) for item in next_observations.values()}
                if len(final_ticks) != 1:
                    raise ReplayReconstructionError("terminal observations disagree on final tick")
                reasons = sorted({
                    str(info["terminal_reason"])
                    for info in infos.values()
                    if isinstance(info, Mapping) and info.get("terminal_reason") is not None
                })
                return CapturedTrajectory(
                    episode_id=episode_id,
                    records=tuple(records),
                    outcome="terminated" if terminated_value else "truncated",
                    outcome_detail={"final_observed_tick": next(iter(final_ticks)), "terminal_reasons": reasons},
                    decision_tick_count=decision_ticks,
                    label_leakage_checks=leakage_checks,
                )
            observations = dict(next_observations)
            if set(observations) != {0, 1}:
                raise ReplayReconstructionError("FirstLight step omitted an actor observation")
            if any(int(getattr(item, "tick", -1)) != tick + DECISION_INTERVAL_TICKS for item in observations.values()):
                raise ReplayReconstructionError("FirstLight did not advance by exactly five ticks")

    @staticmethod
    def write_trajectory(
        captured: CapturedTrajectory,
        *,
        root: str,
        replay_tag: str,
        episode_context: Mapping[str, Any],
        fingerprints: Mapping[str, Any],
        split: str | None = None,
    ) -> TrajectoryManifest:
        """Commit a captured timeline; storage failures remain separately classifiable."""

        writer = TrajectoryWriter(root)
        try:
            writer.begin({
                "episode_id": captured.episode_id,
                "replay_tag": replay_tag,
                "outcome": captured.outcome,
                "outcome_detail": dict(captured.outcome_detail),
                "episode_context_ref": captured.records[0].episode_context_ref,
                "episode_context": dict(episode_context),
                "fingerprints": dict(fingerprints),
                "decision_tick_count": captured.decision_tick_count,
                "label_leakage_checks": captured.label_leakage_checks,
                "split": split,
            })
            for record in captured.records:
                writer.append(record)
            return writer.commit()
        except Exception as error:
            writer.abort()
            raise ReplayCacheError(f"trajectory cache commit failed: {error}") from error
