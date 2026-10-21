"""Legal-action support and expert supervision contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .values import DomainContractError, frozen_map, jsonable

if TYPE_CHECKING:
    from .trajectory import DecisionIdentity


ACTION_MASK_SCHEMA = "elixir_echo.action-mask@1"


@dataclass(frozen=True, slots=True)
class ActionMask:
    """Actions that are legal for the actor at one decision time."""

    identity: DecisionIdentity
    constraints: Mapping[str, Any]

    def __post_init__(self) -> None:
        if "identity" in self.constraints or "schema" in self.constraints:
            raise DomainContractError(
                "action mask constraints cannot redefine identity or schema"
            )
        object.__setattr__(self, "constraints", frozen_map(self.constraints))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ACTION_MASK_SCHEMA,
            "identity": self.identity.to_dict(),
            **jsonable(self.constraints),
        }


@dataclass(frozen=True, slots=True)
class ExpertAction:
    """Expert behavior aligned to one decision step; no actions means WAIT."""

    actions: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "actions",
            tuple(frozen_map(action) for action in self.actions),
        )

    @classmethod
    def wait(cls) -> "ExpertAction":
        return cls(())

    @classmethod
    def from_legacy(
        cls,
        value: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    ) -> "ExpertAction":
        if value is None:
            return cls.wait()
        if isinstance(value, Mapping):
            return cls((value,))
        return cls(tuple(value))

    @property
    def is_wait(self) -> bool:
        return not self.actions

    def to_legacy(self) -> list[dict[str, Any]] | None:
        if self.is_wait:
            return None
        return jsonable(self.actions)
