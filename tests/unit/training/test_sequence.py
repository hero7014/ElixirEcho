from __future__ import annotations

import pytest

from elixir_echo.datasets.trajectory import TrainingEpisode, TrainingStep
from elixir_echo.domain import ActionMask, DecisionIdentity, ExpertAction, Observation
from elixir_echo.policies.protocol import PolicyInput
from elixir_echo.profiles import ObservationProfile
from elixir_echo.temporal import PreviousActionStatus
from elixir_echo.training.config import TrainingConfig, supervision_for
from elixir_echo.training.labels import CompiledLabel, classify_expert_action
from elixir_echo.training.sequence import compile_actor_sequence


class FakeInputBuilder:
    def build(self, observation, action_mask, temporal_context, profile):
        return PolicyInput(
            identity=temporal_context.identity,
            profile=profile.value,
            tensors=observation,
            action_mask=action_mask.constraints,
            temporal_context=temporal_context,
        )


def _step(tick: int, actor: int, action, *, events=()) -> TrainingStep:
    identity = DecisionIdentity("episode-time", tick, actor)
    return TrainingStep(
        identity=identity,
        observation=Observation({}, (), (), (), {"ended": False}),
        action_mask=ActionMask(identity, {}),
        expert_action=ExpertAction.from_legacy(action),
        event_delta=events,
    )


def _compile_generic_label(step, policy_input, config, profile):
    category = classify_expert_action(step.expert_action)
    return CompiledLabel(
        category=category,
        action=step.expert_action,
        supervision=supervision_for(category, config),
        trainable=config.supports(category),
    )


def test_actor_sequence_builds_current_input_before_committing_current_label():
    first_play = {
        "kind": "play_card",
        "owner": 0,
        "card_id": 26_000_001,
        "hand_slot": 1,
        "target_kind": "grid",
        "target_grid": [8, 10],
        "execute_offset_ticks": 0,
    }
    episode = TrainingEpisode(
        episode_id="episode-time",
        outcome="terminated",
        split="train",
        steps=(
            _step(90, 0, first_play),
            _step(90, 1, None),
            _step(
                95,
                0,
                None,
                events=(
                    {"tick": 95, "event_type": "card_play", "owner": 1, "card_id": 26_000_002},
                ),
            ),
            _step(95, 1, None),
        ),
    )

    compiled = compile_actor_sequence(
        episode,
        actor=0,
        input_builder=FakeInputBuilder(),
        label_compiler=_compile_generic_label,
        config=TrainingConfig(),
    )

    assert [item.tick for item in compiled.identities] == [90, 95]
    assert compiled.inputs[0].temporal_context.previous_action.status is PreviousActionStatus.START
    assert compiled.inputs[0].temporal_context.public_state.revealed_cards == ()
    assert compiled.inputs[1].temporal_context.previous_action.status is PreviousActionStatus.KNOWN
    assert compiled.inputs[1].temporal_context.previous_action.actions[0]["kind"] == "play_card"
    assert compiled.inputs[1].temporal_context.public_state.revealed_cards == (26_000_002,)
    assert compiled.labels[0].action == ExpertAction.from_legacy(first_play)
    assert compiled.labels[1].action == ExpertAction.from_legacy(None)
    assert not hasattr(compiled.inputs[0], "expert_action")


def test_truncated_episode_is_not_compilable_for_first_checkpoint():
    episode = TrainingEpisode(
        episode_id="episode-time",
        outcome="truncated",
        split="train",
        steps=(_step(90, 0, None), _step(90, 1, None)),
    )

    with pytest.raises(ValueError, match="terminated"):
        compile_actor_sequence(
            episode,
            actor=0,
            input_builder=FakeInputBuilder(),
            label_compiler=_compile_generic_label,
            config=TrainingConfig(),
        )
