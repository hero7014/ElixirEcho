"""Compile one full terminated replay into actor-local recurrent inputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from elixir_echo.datasets.trajectory import TrainingEpisode, TrainingStep
from elixir_echo.domain import DecisionIdentity
from elixir_echo.policies.protocol import PolicyInput
from elixir_echo.profiles import ObservationProfile
from elixir_echo.temporal import PublicStateTracker
from .config import TrainingConfig
from .labels import CompiledLabel


LabelCompiler = Callable[
    [TrainingStep, PolicyInput, TrainingConfig, ObservationProfile], CompiledLabel
]


@dataclass(frozen=True, slots=True)
class CompiledActorSequence:
    """Parallel model inputs and labels for one actor in one replay."""

    episode_id: str
    actor: int
    profile: ObservationProfile
    identities: tuple[DecisionIdentity, ...]
    inputs: tuple[PolicyInput, ...]
    labels: tuple[CompiledLabel, ...]

    def __post_init__(self) -> None:
        if self.actor not in (0, 1) or not self.identities:
            raise ValueError("compiled actor sequence needs an actor and at least one decision")
        if not (len(self.identities) == len(self.inputs) == len(self.labels)):
            raise ValueError("compiled input, label, and identity counts differ")
        if any(
            identity.episode_id != self.episode_id or identity.actor != self.actor
            for identity in self.identities
        ):
            raise ValueError("compiled sequence contains a foreign episode or actor")
        if any(item.identity != identity for item, identity in zip(self.inputs, self.identities, strict=True)):
            raise ValueError("compiled PolicyInput identity differs from its record")
        if any(right.tick - left.tick != 5 for left, right in zip(self.identities, self.identities[1:])):
            raise ValueError("compiled actor sequence is not continuous at five-tick intervals")

    @property
    def label_counts(self) -> dict[str, int]:
        counts = Counter(item.category.value for item in self.labels)
        return dict(counts)

    @property
    def trainable_count(self) -> int:
        return sum(item.trainable for item in self.labels)


def _validate_episode(episode: TrainingEpisode) -> None:
    if episode.outcome != "terminated":
        raise ValueError("first-checkpoint training compiles terminated episodes only")
    if not episode.steps:
        raise ValueError("training episode has no decision steps")
    identities = tuple(item.identity for item in episode.steps)
    if any(item.episode_id != episode.episode_id for item in identities):
        raise ValueError("training step episode identity differs from its container")
    keys = tuple((item.tick, item.actor) for item in identities)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise ValueError("training episode records are not in strict (tick, actor) order")
    grouped: dict[int, list[int]] = {}
    for tick, actor in keys:
        grouped.setdefault(tick, []).append(actor)
    ticks = tuple(grouped)
    if ticks[0] != 90 or any(grouped[tick] != [0, 1] for tick in ticks):
        raise ValueError("training episode must contain both actors from decision tick 90")
    if any(right - left != 5 for left, right in zip(ticks, ticks[1:])):
        raise ValueError("training episode ticks are not continuous at five-tick intervals")


def compile_actor_sequence(
    episode: TrainingEpisode,
    *,
    actor: int,
    input_builder: Any,
    label_compiler: LabelCompiler,
    config: TrainingConfig,
) -> CompiledActorSequence:
    """Build each input before committing its label into next-step history.

    Provider-specific label compilation is injected by the integration layer.
    """

    if actor not in (0, 1):
        raise ValueError("actor must be 0 or 1")
    _validate_episode(episode)
    profile = ObservationProfile(config.profile)
    tracker = PublicStateTracker()
    actor_steps = tuple(step for step in episode.steps if step.identity.actor == actor)
    identities: list[DecisionIdentity] = []
    inputs: list[PolicyInput] = []
    labels: list[CompiledLabel] = []
    for step in actor_steps:
        temporal = tracker.before_step(step)
        policy_input = input_builder.build(
            step.observation,
            step.action_mask,
            temporal,
            profile,
        )
        if policy_input.identity != step.identity or policy_input.profile != profile.value:
            raise ValueError("input builder changed the current identity or selected profile")
        label = label_compiler(step, policy_input, config, profile)
        # The current expert action becomes history only after this input and
        # its separately stored label have both been created.
        tracker.after_action(step.expert_action)
        identities.append(step.identity)
        inputs.append(policy_input)
        labels.append(label)
    return CompiledActorSequence(
        episode_id=episode.episode_id,
        actor=actor,
        profile=profile,
        identities=tuple(identities),
        inputs=tuple(inputs),
        labels=tuple(labels),
    )
