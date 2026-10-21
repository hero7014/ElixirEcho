"""Atomic decision identity and trajectory-step contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .action import ActionMask, ExpertAction
from .observation import Observation
from .values import content_sha256, frozen_map, jsonable


DECISION_STEP_SCHEMA = "elixir_echo.decision-step@1"


@dataclass(frozen=True, slots=True)
class DecisionIdentity:
    """Episode, time, and actor coordinates of one decision step."""

    episode_id: str
    tick: int
    actor: int

    def __post_init__(self) -> None:
        if not self.episode_id or self.tick < 0 or self.actor not in (0, 1):
            raise ValueError("invalid decision identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "tick": self.tick,
            "actor": self.actor,
        }


@dataclass(frozen=True, slots=True)
class DecisionStep:
    """Atomic current input, public delta, and expert label for one actor."""

    identity: DecisionIdentity
    episode_context_ref: str
    observation: Observation
    action_mask: ActionMask
    expert_action: ExpertAction
    event_delta: tuple[Mapping[str, Any], ...]
    validation: Mapping[str, Any]
    fingerprints: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.episode_context_ref:
            raise ValueError("episode context reference is required")
        if self.action_mask.identity != self.identity:
            raise ValueError("action mask identity does not match decision step identity")
        object.__setattr__(
            self,
            "event_delta",
            tuple(frozen_map(item) for item in self.event_delta),
        )
        object.__setattr__(self, "validation", frozen_map(self.validation))
        object.__setattr__(self, "fingerprints", frozen_map(self.fingerprints))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DECISION_STEP_SCHEMA,
            "identity": self.identity.to_dict(),
            "episode_context_ref": self.episode_context_ref,
            "observation": self.observation.to_dict(),
            "action_mask": self.action_mask.to_dict(),
            "expert_action": self.expert_action.to_legacy(),
            "event_delta": jsonable(self.event_delta),
            "validation": jsonable(self.validation),
            "fingerprints": jsonable(self.fingerprints),
        }

    def content_sha256(self) -> str:
        return content_sha256(self.to_dict())
