"""In-memory recurrent IL smoke using FirstLight V4's model and loss."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from elixir_echo.integrations.firstlight.policy import FirstLightPolicyBackend
from elixir_echo.training.config import TrainingConfig
from elixir_echo.training.sequence import CompiledActorSequence


@dataclass(frozen=True, slots=True)
class TinyOverfitResult:
    episode_id: str
    actor: int
    profile: str
    device: str
    sequence_shape: tuple[int, int]
    input_shapes: dict[str, tuple[int, ...]]
    trainable_parameters: int
    optimizer_steps: int
    loss_history: tuple[float, ...]
    head_losses: dict[str, float]
    weighted_head_losses: dict[str, float]
    head_counts: dict[str, int]
    last_gradient_norm: float


@dataclass(frozen=True, slots=True)
class GuardedILLoss:
    """FirstLight IL losses computed under explicit per-head supervision masks."""

    total: Any
    gate: Any
    candidate: Any
    target: Any
    delay: Any
    continue_action: Any
    value: Any
    gate_count: int
    candidate_count: int
    target_count: int
    delay_count: int
    continue_count: int
    value_count: int


class FirstLightILTrainer:
    """Teacher-force one actor sequence and run only explicit in-memory steps."""

    def __init__(self, backend: FirstLightPolicyBackend, config: TrainingConfig) -> None:
        self.backend = backend
        self.config = config
        self._imitation = backend.runtime.import_module("native_runner.training.v4.imitation")
        self._learning = backend.runtime.import_module("native_runner.training.v4.learning")

    def _sequence(self, compiled: CompiledActorSequence) -> Any:
        import torch

        if compiled.profile is not self.config.profile:
            raise ValueError("compiled sequence profile differs from TrainingConfig")
        if not compiled.trainable_count:
            raise ValueError("training sequence contains no supported WAIT/play_card labels")
        device = self.backend.device
        observations = tuple(
            item.tensors.batch.to_model_input(device) for item in compiled.inputs
        )
        actions = tuple(item.action.to(device) for item in compiled.labels)
        time_steps = len(observations)
        episode_start = torch.zeros((time_steps, 1), dtype=torch.bool, device=device)
        episode_start[0, 0] = True
        valid_mask = torch.ones((time_steps, 1), dtype=torch.bool, device=device)
        gate_loss_mask = torch.tensor(
            [[item.supervision.gate] for item in compiled.labels],
            dtype=torch.bool,
            device=device,
        )
        sequence = self._imitation.ILSequenceV4(
            observations=observations,
            actions=actions,
            episode_start=episode_start,
            valid_mask=valid_mask,
            returns=torch.zeros((time_steps, 1), dtype=torch.float32, device=device),
            gate_loss_mask=gate_loss_mask,
            value_loss_mask=torch.zeros((time_steps, 1), dtype=torch.bool, device=device),
            sequence_id=f"{compiled.episode_id}:actor-{compiled.actor}",
            natural_time=True,
        )
        sequence.validate()
        return sequence

    def _loss(self, sequence: Any, labels: tuple[Any, ...]) -> GuardedILLoss:
        import torch

        if len(labels) != sequence.valid_mask.shape[0]:
            raise ValueError("label and recurrent sequence lengths differ")
        weights = self._imitation.ILLossWeightsV4(
            gate=self.config.gate_loss_weight,
            candidate=self.config.candidate_loss_weight,
            target=self.config.target_loss_weight,
            delay=self.config.delay_loss_weight,
            continue_action=self.config.continue_action_loss_weight,
            value=self.config.value_loss_weight,
        )
        evaluation = self._learning.evaluate_recurrent_sequence(
            self.backend.model,
            sequence.observations,
            sequence.actions,
            sequence.episode_start,
            validate=True,
        )
        heads = ("gate", "candidate", "target", "delay", "continue_action", "value")
        losses: dict[str, Any] = {}
        counts: dict[str, int] = {}
        total = evaluation.value.sum() * 0.0
        for head in heads:
            mask = torch.tensor(
                [[bool(getattr(item.supervision, head))] for item in labels],
                dtype=torch.bool,
                device=sequence.valid_mask.device,
            )
            mask = mask & sequence.valid_mask
            head_sequence = replace(
                sequence,
                valid_mask=mask,
                gate_loss_mask=mask,
                value_loss_mask=mask,
            )
            head_weights = self._imitation.ILLossWeightsV4(
                **{name: float(getattr(weights, name)) if name == head else 0.0 for name in heads}
            )
            result = self._imitation.imitation_loss(
                head_sequence,
                evaluation,
                weights=head_weights,
                gate_act_weight=self.config.gate_act_weight,
                delay_neighbor_weight=self.config.delay_neighbor_weight,
            )
            loss_name = "continue_action" if head == "continue_action" else head
            count_name = "continue_count" if head == "continue_action" else f"{head}_count"
            losses[head] = getattr(result, loss_name)
            counts[head] = int(getattr(result, count_name).detach().cpu())
            total = total + result.total
        return GuardedILLoss(
            total=total,
            gate=losses["gate"],
            candidate=losses["candidate"],
            target=losses["target"],
            delay=losses["delay"],
            continue_action=losses["continue_action"],
            value=losses["value"],
            gate_count=counts["gate"],
            candidate_count=counts["candidate"],
            target_count=counts["target"],
            delay_count=counts["delay"],
            continue_count=counts["continue_action"],
            value_count=counts["value"],
        )

    def tiny_overfit(
        self,
        compiled: CompiledActorSequence,
        *,
        optimizer_steps: int = 3,
    ) -> TinyOverfitResult:
        """Verify recurrent forward/loss/backprop without saving a checkpoint."""

        import torch

        if type(optimizer_steps) is not int or optimizer_steps <= 0:
            raise ValueError("optimizer_steps must be a positive integer")
        sequence = self._sequence(compiled)
        model = self.backend.model
        was_training = model.training
        model.train(True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_history: list[float] = []
        last_gradient_norm = 0.0
        final_loss = None
        try:
            result = self._loss(sequence, compiled.labels)
            loss_history.append(float(result.total.detach().cpu()))
            for _ in range(optimizer_steps):
                optimizer.zero_grad(set_to_none=True)
                result.total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                last_gradient_norm = float(gradient_norm.detach().cpu())
                if not math.isfinite(last_gradient_norm):
                    raise FloatingPointError("non-finite FirstLight V4 gradient norm")
                optimizer.step()
                result = self._loss(sequence, compiled.labels)
                loss_value = float(result.total.detach().cpu())
                if not math.isfinite(loss_value):
                    raise FloatingPointError("non-finite FirstLight V4 IL loss")
                loss_history.append(loss_value)
                final_loss = result
        finally:
            model.train(was_training)

        if final_loss is None:
            raise RuntimeError("tiny overfit did not perform an optimizer step")
        first_batch = sequence.observations[0]
        input_shapes = {
            "match_scalars": tuple(first_batch.match_scalars.shape),
            "own_cards": tuple(first_batch.own_cards.mask.shape),
            "towers": tuple(first_batch.towers.mask.shape),
            "groups": tuple(first_batch.groups.mask.shape),
            "spatial_planes": tuple(first_batch.explicit_spatial_planes.shape),
            "candidates": tuple(first_batch.candidates.mask.shape),
        }
        head_losses = {
            name: float(getattr(final_loss, name).detach().cpu())
            for name in ("gate", "candidate", "target", "delay", "continue_action", "value")
        }
        head_weights = {
            "gate": self.config.gate_loss_weight,
            "candidate": self.config.candidate_loss_weight,
            "target": self.config.target_loss_weight,
            "delay": self.config.delay_loss_weight,
            "continue_action": self.config.continue_action_loss_weight,
            "value": self.config.value_loss_weight,
        }
        weighted_head_losses = {
            name: head_losses[name] * weight for name, weight in head_weights.items()
        }
        head_counts = {
            name: (
                int(getattr(final_loss, f"{name if name != 'continue_action' else 'continue'}_count"))
                if head_weights[name] > 0.0
                else 0
            )
            for name in head_weights
        }
        parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        return TinyOverfitResult(
            episode_id=compiled.episode_id,
            actor=compiled.actor,
            profile=compiled.profile.value,
            device=str(self.backend.device),
            sequence_shape=(len(sequence.observations), sequence.batch_size),
            input_shapes=input_shapes,
            trainable_parameters=parameter_count,
            optimizer_steps=optimizer_steps,
            loss_history=tuple(loss_history),
            head_losses=head_losses,
            weighted_head_losses=weighted_head_losses,
            head_counts=head_counts,
            last_gradient_norm=last_gradient_norm,
        )
