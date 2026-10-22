"""Deterministic tracker for facts proven by actor-fair public event history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from elixir_echo.domain import DecisionIdentity, DecisionStep, ExpertAction
from elixir_echo.domain.values import canonical_json
from .history import History, HistoryView, PreviousAction
from .policy_input import TemporalContext


_PUBLIC_PLAY_EVENTS = frozenset({"action_executed", "card_play", "play_card"})
_IDENTITY_FIELDS = ("tick", "event_type", "owner", "card_id", "position", "combat")
_COMBAT_FIELDS = ("kind", "amount", "position", "target_card_id")


@dataclass(frozen=True, slots=True)
class PublicPlay:
    tick: int
    owner: int
    card_id: int


@dataclass(frozen=True, slots=True)
class PublicPlayDistance:
    card_id: int
    count: int
    capped: bool


@dataclass(frozen=True, slots=True)
class PublicState:
    episode_id: str
    actor: int
    tick: int
    revealed_cards: tuple[int, ...]
    public_play_history: tuple[PublicPlay, ...]
    public_plays_since_last_use: tuple[PublicPlayDistance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "actor": self.actor,
            "tick": self.tick,
            "revealed_cards": list(self.revealed_cards),
            "public_play_history": [
                {"tick": item.tick, "owner": item.owner, "card_id": item.card_id}
                for item in self.public_play_history
            ],
            "public_plays_since_last_use": [
                {"card_id": item.card_id, "count": item.count, "capped": item.capped}
                for item in self.public_plays_since_last_use
            ],
        }


def _deployable_card_id(value: Any) -> int | None:
    if type(value) is not int or not 26_000_000 <= value < 29_000_000:
        return None
    return value


def _semantic_event_identity(event: Mapping[str, Any]) -> str:
    stable: dict[str, Any] = {}
    for key in _IDENTITY_FIELDS:
        if key not in event:
            continue
        value = event[key]
        if key == "combat" and isinstance(value, Mapping):
            value = {name: value[name] for name in _COMBAT_FIELDS if name in value}
        stable[key] = value
    return canonical_json(stable)


class PublicStateTracker:
    """Causal state tracker; no deck, private state, or provider handles are accepted."""

    def __init__(self, *, max_public_plays: int = 64, max_play_distance: int = 15) -> None:
        if max_public_plays <= 0 or max_play_distance <= 0:
            raise ValueError("tracker bounds must be positive")
        self._max_public_plays = int(max_public_plays)
        self._max_play_distance = int(max_play_distance)
        self._history = History()
        self._episode_id: str | None = None
        self._actor: int | None = None
        self._tick: int | None = None
        self._seen_events: set[str] = set()
        self._revealed_cards: list[int] = []
        self._revealed_set: set[int] = set()
        self._public_plays: list[PublicPlay] = []
        self._play_distances: dict[int, tuple[int, bool]] = {}
        self._pending_step: DecisionStep | None = None

    def reset(self, identity: DecisionIdentity) -> None:
        self._episode_id = identity.episode_id
        self._actor = identity.actor
        self._tick = None
        self._seen_events.clear()
        self._revealed_cards.clear()
        self._revealed_set.clear()
        self._public_plays.clear()
        self._play_distances.clear()
        self._pending_step = None
        self._history.reset(identity)

    @property
    def history(self) -> History:
        return self._history

    def _ensure_identity(self, identity: DecisionIdentity) -> None:
        if self._episode_id is None:
            self.reset(identity)
        if identity.episode_id != self._episode_id or identity.actor != self._actor:
            raise ValueError("tracker identity changed within an actor sequence; reset is required")

    def _apply_events(self, tick: int, events: Sequence[Mapping[str, Any]]) -> None:
        current_tick = self._tick
        if current_tick is not None and tick < current_tick:
            raise ValueError("tracker cannot move backward in time")
        for event in events:
            event_tick = event.get("tick")
            if type(event_tick) is not int or event_tick > tick:
                raise ValueError("event is not publicly available at the requested tick")
            identity = _semantic_event_identity(event)
            if identity in self._seen_events:
                continue
            if current_tick is not None and event_tick < current_tick:
                raise ValueError("new public event arrived behind the committed tracker time")
            self._seen_events.add(identity)
            card_id = _deployable_card_id(event.get("card_id"))
            owner = event.get("owner")
            if owner == 1 - self._actor and card_id is not None and card_id not in self._revealed_set:
                self._revealed_set.add(card_id)
                self._revealed_cards.append(card_id)
            if (
                event.get("event_type") in _PUBLIC_PLAY_EVENTS
                and owner in (0, 1)
                and card_id is not None
            ):
                self._record_public_play(event_tick, int(owner), card_id)
        self._tick = tick

    def _record_public_play(self, tick: int, owner: int, card_id: int) -> None:
        self._public_plays.append(PublicPlay(tick, owner, card_id))
        if len(self._public_plays) > self._max_public_plays:
            del self._public_plays[: len(self._public_plays) - self._max_public_plays]
        if owner != 1 - self._actor:
            return
        for revealed in self._revealed_cards:
            if revealed == card_id:
                continue
            count, capped = self._play_distances.get(revealed, (0, False))
            next_count = count + 1
            self._play_distances[revealed] = (
                min(next_count, self._max_play_distance),
                capped or next_count >= self._max_play_distance,
            )
        self._play_distances[card_id] = (0, False)

    def snapshot(self) -> PublicState:
        if self._episode_id is None or self._actor is None or self._tick is None:
            raise ValueError("tracker has no current public state")
        distances = tuple(
            PublicPlayDistance(card_id, *self._play_distances[card_id])
            for card_id in self._revealed_cards
            if card_id in self._play_distances
        )
        return PublicState(
            self._episode_id,
            self._actor,
            self._tick,
            tuple(self._revealed_cards),
            tuple(self._public_plays),
            distances,
        )

    def before_step(self, step: DecisionStep) -> TemporalContext:
        if self._pending_step is not None:
            raise ValueError("the previous step action has not been committed")
        self._ensure_identity(step.identity)
        # Validate temporal continuity before mutating either history or tracker state.
        expected_tick = step.identity.tick if self._tick is None else self._tick + 5
        if step.identity.tick != expected_tick:
            raise ValueError(f"tracker expected tick {expected_tick}, got {step.identity.tick}")
        for event in step.event_delta:
            event_tick = event.get("tick")
            if type(event_tick) is not int or event_tick > step.identity.tick:
                raise ValueError("event is not publicly available at the requested tick")
        history_view = self._history.before_step(step)
        self._apply_events(step.identity.tick, step.event_delta)
        self._pending_step = step
        return TemporalContext(step.identity, history_view, self.snapshot())

    def after_action(self, action: ExpertAction) -> None:
        if self._pending_step is None:
            raise ValueError("before_step must open the current decision before action commit")
        self._history.after_action(action)
        self._pending_step = None
