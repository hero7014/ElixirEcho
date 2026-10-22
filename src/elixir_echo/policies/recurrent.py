"""Episode-boundary helpers for opaque recurrent policy states."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .protocol import RecurrentState


class StatefulBackend(Protocol):
    def initial_state(self, batch_size: int = 1) -> RecurrentState: ...


def reset_for_episode(backend: StatefulBackend, state: RecurrentState) -> RecurrentState:
    """Create a fresh zero state while preserving the carried batch size."""

    return backend.initial_state(batch_size=state.batch_size)
