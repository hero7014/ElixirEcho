"""Label categories supported by the first imitation-learning pass."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from elixir_echo.domain import ExpertAction

if TYPE_CHECKING:
    from .config import HeadSupervision


class LabelCategory(str, Enum):
    """Describe the recorded expert label without changing or guessing it."""

    WAIT = "wait"
    SINGLE_PLAY_CARD = "single_play_card"
    SINGLE_ABILITY = "single_ability"
    MULTI_ACTION = "multi_action"
    MALFORMED = "malformed_or_unrepresentable"


@dataclass(frozen=True, slots=True)
class CompiledLabel:
    """Provider-neutral label metadata plus an opaque backend action target."""

    category: LabelCategory
    action: Any
    supervision: HeadSupervision
    trainable: bool
    report_reason: str | None = None


def classify_expert_action(action: ExpertAction) -> LabelCategory:
    """Classify an expert action; only WAIT and one card play are V1-trainable."""

    if not isinstance(action, ExpertAction):
        raise TypeError("action must be an ExpertAction")
    if not action.actions:
        return LabelCategory.WAIT
    if len(action.actions) > 1:
        return LabelCategory.MULTI_ACTION
    kind = action.actions[0].get("kind")
    if kind == "play_card":
        return LabelCategory.SINGLE_PLAY_CARD
    if kind == "activate_ability":
        return LabelCategory.SINGLE_ABILITY
    return LabelCategory.MALFORMED
