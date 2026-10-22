from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from elixir_echo.integrations.firstlight.trainer import FirstLightILTrainer
from elixir_echo.training.config import HeadSupervision, TrainingConfig
from elixir_echo.training.labels import CompiledLabel, LabelCategory


@dataclass(frozen=True)
class FakeSequence:
    observations: tuple[object, ...]
    actions: tuple[object, ...]
    episode_start: object
    valid_mask: object
    returns: object
    gate_loss_mask: object
    value_loss_mask: object
    sequence_id: str

    def validate(self):
        return None


def _label(category: LabelCategory, supervision: HeadSupervision) -> CompiledLabel:
    return CompiledLabel(
        category=category,
        action=object(),
        supervision=supervision,
        trainable=any(supervision.to_dict().values()),
    )


def test_firstlight_loss_applies_each_explicit_head_mask_before_provider_loss():
    labels = (
        _label(LabelCategory.WAIT, HeadSupervision(gate=True)),
        _label(
            LabelCategory.SINGLE_PLAY_CARD,
            HeadSupervision(gate=True, candidate=True, target=True, continue_action=True),
        ),
        _label(LabelCategory.SINGLE_ABILITY, HeadSupervision()),
    )
    sequence = FakeSequence(
        observations=(object(),) * 3,
        actions=(object(),) * 3,
        episode_start=torch.tensor([[True], [False], [False]]),
        valid_mask=torch.ones((3, 1), dtype=torch.bool),
        returns=torch.zeros((3, 1)),
        gate_loss_mask=torch.ones((3, 1), dtype=torch.bool),
        value_loss_mask=torch.zeros((3, 1), dtype=torch.bool),
        sequence_id="episode:actor-0",
    )
    observed = []

    def fake_loss(masked_sequence, _evaluation, *, weights, **_kwargs):
        observed.append((masked_sequence.valid_mask.clone(), weights))
        count = masked_sequence.valid_mask.sum().to(dtype=torch.float32)
        values = {name: count for name in ("gate", "candidate", "target", "delay", "continue_action", "value")}
        counts = {name: int(count.item()) for name in values}
        return SimpleNamespace(
            total=sum(getattr(weights, name) * values[name] for name in values),
            **values,
            gate_count=torch.tensor(counts["gate"]),
            candidate_count=torch.tensor(counts["candidate"]),
            target_count=torch.tensor(counts["target"]),
            delay_count=torch.tensor(counts["delay"]),
            continue_count=torch.tensor(counts["continue_action"]),
            value_count=torch.tensor(counts["value"]),
        )

    trainer = object.__new__(FirstLightILTrainer)
    trainer.config = TrainingConfig()
    trainer.backend = SimpleNamespace(model=object())
    trainer._learning = SimpleNamespace(
        evaluate_recurrent_sequence=lambda *_args, **_kwargs: SimpleNamespace(
            value=torch.zeros((3, 1)), log_prob=torch.zeros((3, 1))
        )
    )
    trainer._imitation = SimpleNamespace(
        ILLossWeightsV4=lambda **kwargs: SimpleNamespace(**kwargs),
        imitation_loss=fake_loss,
    )

    result = trainer._loss(sequence, labels)

    expected = (
        torch.tensor([[True], [True], [False]]),
        torch.tensor([[False], [True], [False]]),
        torch.tensor([[False], [True], [False]]),
        torch.tensor([[False], [False], [False]]),
        torch.tensor([[False], [True], [False]]),
        torch.tensor([[False], [False], [False]]),
    )
    assert len(observed) == 6
    assert all(torch.equal(item[0], mask) for item, mask in zip(observed, expected, strict=True))
    configured_weights = {
        "gate": trainer.config.gate_loss_weight,
        "candidate": trainer.config.candidate_loss_weight,
        "target": trainer.config.target_loss_weight,
        "delay": trainer.config.delay_loss_weight,
        "continue_action": trainer.config.continue_action_loss_weight,
        "value": trainer.config.value_loss_weight,
    }
    for selected_head, (_, weights) in zip(configured_weights, observed, strict=True):
        assert {
            name: getattr(weights, name) for name in configured_weights
        } == {
            name: configured_weights[name] if name == selected_head else 0.0
            for name in configured_weights
        }
    assert result.gate_count == 2
    assert result.candidate_count == 1
    assert result.target_count == 1
    assert result.delay_count == 0
    assert result.continue_count == 1
    assert result.value_count == 0
