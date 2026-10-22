"""Training-time compilation and learning interfaces."""

from .config import HeadSupervision, TrainingConfig, supervision_for
from .labels import CompiledLabel, LabelCategory, classify_expert_action
from .sequence import CompiledActorSequence, compile_actor_sequence

__all__ = [
    "HeadSupervision",
    "CompiledLabel",
    "LabelCategory",
    "CompiledActorSequence",
    "TrainingConfig",
    "classify_expert_action",
    "compile_actor_sequence",
    "supervision_for",
]
