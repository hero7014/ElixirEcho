from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from elixir_echo.integrations.firstlight.trainer import FirstLightILTrainer
from elixir_echo.datasets.trajectory import TrainingEpisode
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
            value=torch.zeros((3, 1)), log_prob=torch.zeros((3, 1)), final_state=None
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


def test_tbptt_carries_only_detached_state_across_ordered_chunks_and_starts_empty():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    observations = tuple(f"frame-{index}" for index in range(5))
    sequence = FakeSequence(
        observations=observations,
        actions=tuple(f"action-{index}" for index in range(5)),
        episode_start=torch.tensor([[True], [False], [False], [False], [False]]),
        valid_mask=torch.ones((5, 1), dtype=torch.bool),
        returns=torch.zeros((5, 1)),
        gate_loss_mask=torch.ones((5, 1), dtype=torch.bool),
        value_loss_mask=torch.zeros((5, 1), dtype=torch.bool),
        sequence_id="episode:actor-0",
    )
    labels = tuple(_label(LabelCategory.WAIT, HeadSupervision(gate=True)) for _ in observations)
    trainer = object.__new__(FirstLightILTrainer)
    trainer.config = TrainingConfig(tbptt_length=2)
    trainer.backend = SimpleNamespace(model=Model())
    trainer._sequence = lambda _compiled: sequence
    seen = []

    def fake_loss(chunk, chunk_labels, *, initial_state=None):
        seen.append((chunk.observations, chunk.actions, chunk_labels, chunk.episode_start.clone(), initial_state))
        if initial_state is None:
            final_state = trainer.backend.model.weight.reshape(1)
        else:
            final_state = initial_state + 1
        return SimpleNamespace(total=trainer.backend.model.weight.square(), final_state=final_state)

    trainer._loss = fake_loss
    compiled = SimpleNamespace(episode_id="episode", actor=0, labels=labels)

    result = trainer.train_sequence(compiled)

    assert result.chunks == result.optimizer_steps == 3
    assert result.tbptt_length == 2
    assert [item[0] for item in seen] == [
        ("frame-0", "frame-1"),
        ("frame-2", "frame-3"),
        ("frame-4",),
    ]
    assert [item[1] for item in seen] == [
        ("action-0", "action-1"),
        ("action-2", "action-3"),
        ("action-4",),
    ]
    assert seen[0][2] == labels[:2]
    assert seen[1][2] == labels[2:4]
    assert seen[2][2] == labels[4:]
    assert seen[0][3].tolist() == [[True], [False]]
    assert seen[1][3].tolist() == [[False], [False]]
    assert seen[2][3].tolist() == [[False]]
    assert seen[0][4] is None  # each actor episode resets
    assert seen[1][4] is not None and not seen[1][4].requires_grad
    assert seen[2][4] is not None and not seen[2][4].requires_grad
    assert len(result.loss_history) == len(result.gradient_norms) == 3


def test_training_pass_uses_only_train_terminated_episodes_and_resets_per_actor(monkeypatch):
    from elixir_echo.integrations.firstlight import trainer as firstlight_trainer

    labels = (
        _label(LabelCategory.WAIT, HeadSupervision(gate=True)),
        _label(LabelCategory.SINGLE_PLAY_CARD, HeadSupervision(gate=True)),
    )
    episodes = (
        TrainingEpisode("train-a", "terminated", "train", (object(),)),
        TrainingEpisode("valid-a", "terminated", "validation", (object(),)),
        TrainingEpisode("truncated-a", "truncated", "train", (object(),)),
    )
    compiler_calls = []

    def compile_episode(episode, *, actor, **_kwargs):
        compiler_calls.append((episode.episode_id, actor))
        return SimpleNamespace(
            episode_id=episode.episode_id,
            actor=actor,
            labels=labels,
            trainable_count=2,
        )

    monkeypatch.setattr(firstlight_trainer, "compile_firstlight_actor_sequence", compile_episode)
    trainer = object.__new__(FirstLightILTrainer)
    trainer.config = TrainingConfig(tbptt_length=2)
    trainer.backend = SimpleNamespace(model=object(), runtime=object(), components=object())
    training_calls = []

    def train_sequence(compiled, *, optimizer=None):
        training_calls.append((compiled.episode_id, compiled.actor, optimizer))
        return SimpleNamespace(optimizer_steps=1, loss_history=(0.5,))

    trainer.train_sequence = train_sequence
    shared_optimizer = object()

    result = trainer.train_episodes(
        episodes,
        input_builder=object(),
        optimizer=shared_optimizer,
    )

    assert compiler_calls == [("train-a", 0), ("train-a", 1)]
    assert training_calls == [
        ("train-a", 0, shared_optimizer),
        ("train-a", 1, shared_optimizer),
    ]
    assert result.episode_count == 1
    assert result.actor_sequence_count == 2
    assert result.decision_count == 4
    assert result.trainable_decision_count == 4
    assert result.excluded_split_episode_count == 1
    assert result.excluded_truncated_episode_count == 1
    assert result.excluded_untrainable_actor_sequence_count == 0
    assert result.optimizer_steps == 2
    assert result.loss_history == (0.5, 0.5)


def test_training_pass_stops_consuming_episodes_at_requested_limit(monkeypatch):
    from elixir_echo.integrations.firstlight import trainer as firstlight_trainer

    labels = (_label(LabelCategory.WAIT, HeadSupervision(gate=True)),)
    episode = TrainingEpisode("train-a", "terminated", "train", (object(),))
    monkeypatch.setattr(
        firstlight_trainer,
        "compile_firstlight_actor_sequence",
        lambda episode, *, actor, **_kwargs: SimpleNamespace(
            episode_id=episode.episode_id,
            actor=actor,
            labels=labels,
            trainable_count=1,
        ),
    )
    trainer = object.__new__(FirstLightILTrainer)
    trainer.config = TrainingConfig()
    trainer.backend = SimpleNamespace(model=object(), components=object())
    trainer.train_sequence = lambda *_args, **_kwargs: SimpleNamespace(
        optimizer_steps=1, loss_history=(0.25,)
    )

    def episodes():
        yield episode
        raise AssertionError("training pass read beyond max_episodes")

    result = trainer.train_episodes(
        episodes(),
        input_builder=object(),
        optimizer=object(),
        max_episodes=1,
    )

    assert result.episode_count == 1


def test_training_pass_reports_but_does_not_train_actor_sequences_without_supported_labels(monkeypatch):
    from elixir_echo.integrations.firstlight import trainer as firstlight_trainer

    unsupported = (_label(LabelCategory.SINGLE_ABILITY, HeadSupervision()),)
    episode = TrainingEpisode("train-empty", "terminated", "train", (object(),))
    monkeypatch.setattr(
        firstlight_trainer,
        "compile_firstlight_actor_sequence",
        lambda episode, *, actor, **_kwargs: SimpleNamespace(
            episode_id=episode.episode_id,
            actor=actor,
            labels=unsupported,
            trainable_count=0,
        ),
    )
    trainer = object.__new__(FirstLightILTrainer)
    trainer.config = TrainingConfig()
    trainer.backend = SimpleNamespace(model=object(), components=object())
    training_calls = []
    trainer.train_sequence = lambda *args, **kwargs: training_calls.append((args, kwargs))

    result = trainer.train_episodes(
        (episode,),
        input_builder=object(),
        optimizer=object(),
    )

    assert training_calls == []
    assert result.episode_count == 1
    assert result.actor_sequence_count == 2
    assert result.decision_count == 2
    assert result.trainable_decision_count == 0
    assert result.excluded_untrainable_actor_sequence_count == 2
    assert result.optimizer_steps == 0
