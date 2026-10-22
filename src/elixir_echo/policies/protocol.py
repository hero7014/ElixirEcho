"""Provider-neutral policy and recurrent-state interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from elixir_echo.domain import DecisionIdentity
from .recurrent import reset_for_episode


@dataclass(frozen=True, slots=True)
class PolicyInput:
    """One actor's selected observation, temporal context, and legal support."""

    identity: DecisionIdentity
    profile: str
    tensors: Any
    action_mask: Mapping[str, Any] | None = None
    temporal_context: Any | None = None


@dataclass(frozen=True, slots=True)
class RecurrentState:
    """Opaque backend-owned recurrent values plus the episode they belong to."""

    payload: Any
    batch_size: int = 1
    episode_id: str | None = None
    last_identity: DecisionIdentity | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("recurrent-state batch_size must be positive")
        if self.last_identity is not None:
            if self.episode_id not in (None, self.last_identity.episode_id):
                raise ValueError("recurrent-state episode differs from its last identity")
            object.__setattr__(self, "episode_id", self.last_identity.episode_id)


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """Backend-neutral action-head logits and the state after this decision."""

    action_logits: Mapping[str, Any]
    next_state: RecurrentState
    value: Any | None = None


@runtime_checkable
class PolicyBackend(Protocol):
    """The minimal recurrent policy contract required by a training loop."""

    def initial_state(self, batch_size: int = 1) -> RecurrentState: ...

    def step(self, policy_input: PolicyInput, state: RecurrentState) -> PolicyOutput: ...


def run_policy_step(
    backend: PolicyBackend,
    policy_input: PolicyInput,
    state: RecurrentState,
    *,
    episode_start: bool = False,
) -> PolicyOutput:
    """Run one decision, clearing carried state exactly at episode boundaries."""

    if episode_start:
        state = reset_for_episode(backend, state)
    elif state.episode_id is not None and state.episode_id != policy_input.identity.episode_id:
        raise ValueError("recurrent state crossed an episode boundary without reset")
    elif state.last_identity is not None and state.last_identity.actor != policy_input.identity.actor:
        raise ValueError("recurrent state crossed an actor boundary")
    elif state.last_identity is not None and policy_input.identity.tick <= state.last_identity.tick:
        raise ValueError("recurrent-state tick must advance within an actor episode")

    output = backend.step(policy_input, state)
    next_state = replace(
        output.next_state,
        episode_id=policy_input.identity.episode_id,
        last_identity=policy_input.identity,
    )
    return replace(output, next_state=next_state)
