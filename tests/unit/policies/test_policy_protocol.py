from __future__ import annotations

from dataclasses import dataclass

import pytest

from elixir_echo.domain import DecisionIdentity
from elixir_echo.policies.protocol import (
    PolicyInput,
    PolicyOutput,
    RecurrentState,
    run_policy_step,
)


@dataclass
class FakeBackend:
    calls: list[tuple[PolicyInput, object]]

    def initial_state(self, batch_size: int = 1) -> RecurrentState:
        return RecurrentState(payload={"counter": 0}, batch_size=batch_size)

    def step(self, policy_input: PolicyInput, state: RecurrentState) -> PolicyOutput:
        self.calls.append((policy_input, state.payload))
        return PolicyOutput(
            action_logits={"gate": (0.2, 0.8)},
            next_state=RecurrentState(payload={"counter": state.payload["counter"] + 1}),
        )


def _input(episode: str, tick: int) -> PolicyInput:
    return PolicyInput(
        identity=DecisionIdentity(episode, tick, 0),
        profile="core",
        tensors=object(),
    )


def test_policy_backend_protocol_is_provider_neutral_and_recurrent_state_carries() -> None:
    backend = FakeBackend(calls=[])
    state = backend.initial_state()

    first = run_policy_step(backend, _input("episode-a", 90), state, episode_start=True)
    second = run_policy_step(backend, _input("episode-a", 95), first.next_state)

    assert [
        (item.identity.to_dict(), payload)
        for item, payload in backend.calls
    ] == [
        ({"episode_id": "episode-a", "tick": 90, "actor": 0}, {"counter": 0}),
        ({"episode_id": "episode-a", "tick": 95, "actor": 0}, {"counter": 1}),
    ]
    assert second.action_logits["gate"] == (0.2, 0.8)
    assert second.next_state.payload == {"counter": 2}


def test_policy_step_clears_state_on_episode_start() -> None:
    backend = FakeBackend(calls=[])
    output = run_policy_step(
        backend,
        _input("episode-b", 90),
        RecurrentState(payload={"counter": 99}, episode_id="episode-a"),
        episode_start=True,
    )

    assert backend.calls[0][0].identity == DecisionIdentity("episode-b", 90, 0)
    assert backend.calls[0][1] == {"counter": 0}
    assert output.next_state.episode_id == "episode-b"


def test_policy_step_rejects_cross_episode_state_without_reset() -> None:
    backend = FakeBackend(calls=[])

    with pytest.raises(ValueError, match="episode boundary"):
        run_policy_step(
            backend,
            _input("episode-b", 90),
            RecurrentState(payload={"counter": 1}, episode_id="episode-a"),
        )


def test_policy_step_rejects_cross_actor_state_within_episode() -> None:
    backend = FakeBackend(calls=[])
    previous = DecisionIdentity("episode-a", 90, 0)

    with pytest.raises(ValueError, match="actor boundary"):
        run_policy_step(
            backend,
            PolicyInput(
                identity=DecisionIdentity("episode-a", 95, 1),
                profile="core",
                tensors=object(),
            ),
            RecurrentState(payload={"counter": 1}, last_identity=previous),
        )


@pytest.mark.parametrize("tick", [90, 94])
def test_policy_step_rejects_duplicate_or_backwards_tick(tick: int) -> None:
    backend = FakeBackend(calls=[])
    previous = DecisionIdentity("episode-a", 95, 0)
    current = PolicyInput(
        identity=DecisionIdentity("episode-a", tick, 0),
        profile="core",
        tensors=object(),
    )

    with pytest.raises(ValueError, match="tick must advance"):
        run_policy_step(
            backend,
            current,
            RecurrentState(payload={"counter": 1}, last_identity=previous),
        )
