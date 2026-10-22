"""Frozen learning choices for the first supervised training pass."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from elixir_echo.domain.values import content_sha256
from elixir_echo.profiles import ObservationProfile
from .labels import LabelCategory


_FIRST_PASS_LABELS = (LabelCategory.WAIT, LabelCategory.SINGLE_PLAY_CARD)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Identify the model-facing profile, supported labels, and IL objective."""

    profile: ObservationProfile = ObservationProfile.RICH
    supported_label_categories: tuple[LabelCategory, ...] = _FIRST_PASS_LABELS
    gate_loss_weight: float = 1.0
    candidate_loss_weight: float = 1.0
    target_loss_weight: float = 1.0
    continue_action_loss_weight: float = 1.0
    delay_loss_weight: float = 0.0
    value_loss_weight: float = 0.0
    gate_act_weight: float = 8.0
    delay_neighbor_weight: float = 0.2
    learning_rate: float = 3e-5
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    tbptt_length: int = 32
    optimizer_name: str = "adamw"

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", ObservationProfile(self.profile))
        categories = tuple(LabelCategory(item) for item in self.supported_label_categories)
        object.__setattr__(self, "supported_label_categories", categories)
        if len(set(categories)) != len(categories):
            raise ValueError("supported label categories contain duplicates")
        if not set(categories).issubset(_FIRST_PASS_LABELS):
            raise ValueError("the first training pass supports only WAIT and single play_card")
        for name in (
            "gate_loss_weight",
            "candidate_loss_weight",
            "target_loss_weight",
            "continue_action_loss_weight",
            "delay_loss_weight",
            "value_loss_weight",
            "gate_act_weight",
            "delay_neighbor_weight",
            "learning_rate",
            "weight_decay",
            "max_grad_norm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.gate_act_weight <= 0 or self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("gate_act_weight, learning_rate, and max_grad_norm must be positive")
        if type(self.tbptt_length) is not int or self.tbptt_length <= 0:
            raise ValueError("tbptt_length must be a positive integer")
        if self.delay_loss_weight != 0.0 or self.value_loss_weight != 0.0:
            raise ValueError("delay and value losses are disabled in the first training pass")
        if self.delay_neighbor_weight > 1.0 / 3.0:
            raise ValueError("delay_neighbor_weight must not exceed one third")
        if self.optimizer_name != "adamw":
            raise ValueError("the first training pass uses AdamW")

    def to_dict(self) -> dict[str, Any]:
        supervision_masks = {
            category.value: supervision_for(category, self).to_dict()
            for category in LabelCategory
        }
        return {
            "profile": self.profile.value,
            "supported_label_categories": [item.value for item in self.supported_label_categories],
            "supervision_masks": supervision_masks,
            "loss_weights": {
                "gate": float(self.gate_loss_weight),
                "candidate": float(self.candidate_loss_weight),
                "target": float(self.target_loss_weight),
                "continue_action": float(self.continue_action_loss_weight),
                "delay": float(self.delay_loss_weight),
                "value": float(self.value_loss_weight),
            },
            "gate_act_weight": float(self.gate_act_weight),
            "delay_neighbor_weight": float(self.delay_neighbor_weight),
            "optimizer": {
                "name": self.optimizer_name,
                "learning_rate": float(self.learning_rate),
                "weight_decay": float(self.weight_decay),
            },
            "max_grad_norm": float(self.max_grad_norm),
            "tbptt_length": self.tbptt_length,
        }

    def fingerprint(self) -> str:
        return content_sha256(self.to_dict())

    def supports(self, category: LabelCategory) -> bool:
        return LabelCategory(category) in self.supported_label_categories


@dataclass(frozen=True, slots=True)
class HeadSupervision:
    """Explicit per-label head mask; unsupported categories never get loss."""

    gate: bool = False
    candidate: bool = False
    target: bool = False
    continue_action: bool = False
    delay: bool = False
    value: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "gate": self.gate,
            "candidate": self.candidate,
            "target": self.target,
            "continue_action": self.continue_action,
            "delay": self.delay,
            "value": self.value,
        }


def supervision_for(category: LabelCategory, config: TrainingConfig) -> HeadSupervision:
    category = LabelCategory(category)
    if not config.supports(category):
        return HeadSupervision()
    if category is LabelCategory.WAIT:
        return HeadSupervision(gate=True)
    if category is LabelCategory.SINGLE_PLAY_CARD:
        return HeadSupervision(gate=True, candidate=True, target=True, continue_action=True)
    return HeadSupervision()
