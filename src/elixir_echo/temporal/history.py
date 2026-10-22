"""Actor-local public history with an explicit current-label commit boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from elixir_echo.domain import DecisionIdentity, ExpertAction, Observation
from elixir_echo.domain.values import canonical_json, frozen_map


class HistorySource(Protocol):
    """The training-safe fields needed to advance public temporal state."""

    identity: DecisionIdentity
    observation: Observation
    expert_action: ExpertAction
    event_delta: tuple[Mapping[str, Any], ...]


class PreviousActionStatus(str, Enum):
    START = "start"
    KNOWN = "known"
    UNKNOWN_MASKED = "unknown_masked"


@dataclass(frozen=True, slots=True)
class PreviousAction:
    """Prior action state; unsupported real actions remain explicitly masked."""

    status: PreviousActionStatus
    actions: tuple[Mapping[str, Any], ...]
    source_tick: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(frozen_map(item) for item in self.actions))
        if self.status is PreviousActionStatus.START:
            if self.actions or self.source_tick is not None:
                raise ValueError("episode-start action state cannot carry an action or source tick")
        elif self.source_tick is None or self.source_tick < 0:
            raise ValueError("a completed prior decision requires a source tick")
        if self.status is PreviousActionStatus.UNKNOWN_MASKED and self.actions:
            raise ValueError("unknown/masked action state cannot expose the unsupported action")

    @classmethod
    def start(cls) -> "PreviousAction":
        return cls(PreviousActionStatus.START, (), None)

    @classmethod
    def known(cls, actions: tuple[Mapping[str, Any], ...], source_tick: int) -> "PreviousAction":
        return cls(PreviousActionStatus.KNOWN, actions, source_tick)

    @classmethod
    def unknown_masked(cls, source_tick: int) -> "PreviousAction":
        return cls(PreviousActionStatus.UNKNOWN_MASKED, (), source_tick)


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    identity: DecisionIdentity
    observation: Observation
    events: tuple[Mapping[str, Any], ...]
    action: PreviousAction

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(frozen_map(item) for item in self.events))


@dataclass(frozen=True, slots=True)
class HistoryView:
    identity: DecisionIdentity
    previous_tick: int | None
    prior_entries: tuple[HistoryEntry, ...]
    current_public_events: tuple[Mapping[str, Any], ...]
    previous_action: PreviousAction

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_public_events",
            tuple(frozen_map(item) for item in self.current_public_events),
        )


def _known_wait(source_tick: int) -> PreviousAction:
    return PreviousAction.known((), source_tick)


def _representable_play(action: Mapping[str, Any], actor: int) -> bool:
    slot = action.get("hand_slot")
    target = action.get("target_grid")
    return (
        action.get("kind") == "play_card"
        and action.get("owner") == actor
        and type(slot) is int
        and 0 <= slot < 4
        and action.get("target_kind") == "grid"
        and isinstance(target, (tuple, list))
        and len(target) == 2
        and all(type(value) in (int, float) for value in target)
    )


def _classify_action(action: ExpertAction, actor: int, tick: int) -> PreviousAction:
    actions = action.actions
    if not actions:
        return _known_wait(tick)
    if len(actions) != 1:
        return PreviousAction.unknown_masked(tick)
    item = actions[0]
    if item.get("kind") == "wait":
        return _known_wait(tick)
    if not _representable_play(item, actor):
        return PreviousAction.unknown_masked(tick)
    return PreviousAction.known((item,), tick)


class History:
    """Ordered actor-local history, advanced only after the current action commits."""

    def __init__(self) -> None:
        self._episode_id: str | None = None
        self._actor: int | None = None
        self._start_tick: int | None = None
        self._last_tick: int | None = None
        self._entries: list[HistoryEntry] = []
        self._previous_action = PreviousAction.start()
        self._pending_step: HistorySource | None = None

    def reset(self, identity: DecisionIdentity) -> None:
        self._episode_id = identity.episode_id
        self._actor = identity.actor
        self._start_tick = identity.tick
        self._last_tick = None
        self._entries.clear()
        self._previous_action = PreviousAction.start()
        self._pending_step = None

    def _validate_step(self, step: HistorySource) -> None:
        if self._episode_id is None or self._actor is None or self._start_tick is None:
            raise ValueError("history must be reset before use")
        identity = step.identity
        if identity.episode_id != self._episode_id or identity.actor != self._actor:
            raise ValueError("history record identity changed within an actor sequence")
        expected = self._start_tick if self._last_tick is None else self._last_tick + 5
        if identity.tick != expected:
            raise ValueError(f"history expected tick {expected}, got {identity.tick}")
        for event in step.event_delta:
            event_tick = event.get("tick")
            if type(event_tick) is not int or event_tick > identity.tick:
                raise ValueError("current public event is not causally available at this tick")

    def before_step(self, step: HistorySource) -> HistoryView:
        if self._pending_step is not None:
            raise ValueError("the previous input view has not been committed")
        self._validate_step(step)
        self._pending_step = step
        return HistoryView(
            identity=step.identity,
            previous_tick=self._last_tick,
            prior_entries=tuple(self._entries),
            current_public_events=step.event_delta,
            previous_action=self._previous_action,
        )

    def after_action(self, action: ExpertAction) -> None:
        step = self._pending_step
        if step is None:
            raise ValueError("before_step must open the current decision before action commit")
        if canonical_json(action.to_legacy()) != canonical_json(step.expert_action.to_legacy()):
            raise ValueError("committed action does not match the opened decision label")
        completed = _classify_action(action, step.identity.actor, step.identity.tick)
        self._entries.append(
            HistoryEntry(step.identity, step.observation, step.event_delta, completed)
        )
        self._previous_action = completed
        self._last_tick = step.identity.tick
        self._pending_step = None
