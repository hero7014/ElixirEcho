"""Causal temporal context supplied alongside the current Observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from elixir_echo.domain import DecisionIdentity
from .history import HistoryView

if TYPE_CHECKING:
    from .public_state import PublicState


@dataclass(frozen=True, slots=True)
class TemporalContext:
    """History-derived public context for one actor decision."""

    identity: DecisionIdentity
    history: HistoryView
    public_state: PublicState

    def __post_init__(self) -> None:
        if self.history.identity != self.identity:
            raise ValueError("temporal history identity does not match current decision")
        if (
            self.public_state.episode_id != self.identity.episode_id
            or self.public_state.actor != self.identity.actor
            or self.public_state.tick != self.identity.tick
        ):
            raise ValueError("public state identity does not match current decision")

    @property
    def previous_action(self):
        return self.history.previous_action
