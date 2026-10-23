"""Strict expert-label compilation and IL sequence adaptation for FirstLight V4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elixir_echo.datasets.trajectory import TrainingStep
from elixir_echo.profiles import ObservationProfile
from elixir_echo.training.config import TrainingConfig, supervision_for
from elixir_echo.training.labels import CompiledLabel, LabelCategory, classify_expert_action
from elixir_echo.training.sequence import CompiledActorSequence, compile_actor_sequence

if TYPE_CHECKING:
    from elixir_echo.integrations.firstlight.tensorizer import TensorizedDecision


class _UnrepresentableLabel(ValueError):
    pass


def _empty_action(components: Any, tensorized: TensorizedDecision) -> Any:
    source = tensorized.batch.candidates.native_visible_card_id
    width = components.config.max_micro_actions
    action = components.types.ActionSequenceV4(
        gate=source.new_tensor([components.types.GATE_WAIT], dtype=source.dtype),
        micro_action_count=source.new_zeros((1,), dtype=source.dtype),
        candidate_index=source.new_full((1, width), -1),
        candidate_uid=source.new_full((1, width), -1),
        target_cell=source.new_full((1, width), -1),
        delay_offset_bin=source.new_full((1, width), -1),
    )
    action.validate(components.config, candidate_count=components.config.max_action_candidates)
    return action


def _compile_play_card(
    step: TrainingStep,
    tensorized: TensorizedDecision,
    components: Any,
    placeholder: Any,
) -> Any:
    expert = step.expert_action.actions[0]
    actor = step.identity.actor
    if type(expert.get("owner")) is not int or expert["owner"] != actor:
        raise _UnrepresentableLabel("expert action owner does not match decision actor")

    slot = expert.get("hand_slot")
    if type(slot) is not int or slot not in tensorized.candidate_index_by_slot:
        raise _UnrepresentableLabel("expert hand slot has no legal V4 candidate")
    candidate_row = int(tensorized.candidate_index_by_slot[slot])
    candidates = tensorized.batch.candidates
    candidate_count = int(candidates.native_visible_card_id.shape[1])
    if not 0 <= candidate_row < candidate_count or not bool(candidates.mask[0, candidate_row]):
        raise _UnrepresentableLabel("expert hand slot resolves to an inactive candidate")
    expected_card = int(candidates.native_visible_card_id[0, candidate_row].item())
    if type(expert.get("card_id")) is not int or expert["card_id"] != expected_card:
        raise _UnrepresentableLabel("expert card does not match its ActionMask candidate")
    if slot not in tensorized.candidate_uid_by_slot:
        raise _UnrepresentableLabel("expert hand slot has no stable candidate identity")

    target = expert.get("target_grid")
    if (
        expert.get("target_kind") != "grid"
        or not isinstance(target, (tuple, list))
        or len(target) != 2
        or any(type(value) is not int for value in target)
    ):
        raise _UnrepresentableLabel("play_card expert action requires one exact grid target")
    x, y = target
    if actor == 1:
        x = components.config.board_width - 1 - x
        y = components.config.board_height - 1 - y
    if not (0 <= x < components.config.board_width and 0 <= y < components.config.board_height):
        raise _UnrepresentableLabel("expert target is outside the V4 grid")
    if not bool(candidates.placement[0, candidate_row, y, x]):
        raise _UnrepresentableLabel("expert target is illegal under the paired ActionMask")

    delay_ticks = expert.get("execute_offset_ticks")
    if type(delay_ticks) is not int:
        raise _UnrepresentableLabel("expert action delay is missing or non-integral")
    delay_ms = delay_ticks * 50
    try:
        delay_bin = components.config.delay_offset_ms.index(delay_ms)
    except ValueError as error:
        raise _UnrepresentableLabel("expert action delay is outside the V4 vocabulary") from error

    placeholder.gate[0] = components.types.GATE_ACT
    placeholder.micro_action_count[0] = 1
    placeholder.candidate_index[0, 0] = candidate_row
    placeholder.candidate_uid[0, 0] = int(tensorized.candidate_uid_by_slot[slot])
    placeholder.target_cell[0, 0] = y * components.config.board_width + x
    placeholder.delay_offset_bin[0, 0] = delay_bin
    placeholder.validate(components.config, candidate_count=components.config.max_action_candidates)
    return placeholder


def compile_expert_label(
    step: TrainingStep,
    tensorized: TensorizedDecision,
    components: Any,
    config: TrainingConfig,
    profile: ObservationProfile,
) -> CompiledLabel:
    """Compile WAIT/single-card labels; preserve all other classes without loss."""

    selected_profile = ObservationProfile(profile)
    if selected_profile is not config.profile:
        raise ValueError("training profile differs from the frozen TrainingConfig")
    if step.action_mask.identity != step.identity:
        raise ValueError("ActionMask identity differs from expert label identity")
    if tensorized.actor != step.identity.actor:
        raise ValueError("tensorized actor differs from expert label identity")

    category = classify_expert_action(step.expert_action)
    placeholder = _empty_action(components, tensorized)
    if not config.supports(category):
        return CompiledLabel(
            category=category,
            action=placeholder,
            supervision=supervision_for(category, config),
            trainable=False,
            report_reason=f"{category.value} is outside the first-pass label policy",
        )
    if category is LabelCategory.WAIT:
        return CompiledLabel(
            category=category,
            action=placeholder,
            supervision=supervision_for(category, config),
            trainable=True,
        )

    try:
        action = _compile_play_card(step, tensorized, components, placeholder)
    except _UnrepresentableLabel as error:
        return CompiledLabel(
            category=LabelCategory.MALFORMED,
            action=_empty_action(components, tensorized),
            supervision=supervision_for(LabelCategory.MALFORMED, config),
            trainable=False,
            report_reason=str(error),
        )
    return CompiledLabel(
        category=category,
        action=action,
        supervision=supervision_for(category, config),
        trainable=True,
    )


def compile_firstlight_actor_sequence(
    episode: Any,
    *,
    actor: int,
    input_builder: Any,
    components: Any,
    config: TrainingConfig,
) -> CompiledActorSequence:
    """Bind the provider-neutral sequence compiler to FirstLight V4 labels."""

    input_mode = getattr(getattr(input_builder, "tensorizer", None), "input_mode", None)
    if input_mode != config.input_mode:
        raise ValueError("input builder mode differs from the fingerprinted TrainingConfig")

    def label_compiler(step, policy_input, selected_config, profile):
        return compile_expert_label(
            step,
            policy_input.tensors,
            components,
            selected_config,
            profile,
        )

    return compile_actor_sequence(
        episode,
        actor=actor,
        input_builder=input_builder,
        label_compiler=label_compiler,
        config=config,
    )
