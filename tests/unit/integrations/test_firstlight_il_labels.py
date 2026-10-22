from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from elixir_echo.datasets.trajectory import TrainingStep
from elixir_echo.domain import ActionMask, DecisionIdentity, ExpertAction, Observation
from elixir_echo.integrations.firstlight.il import compile_expert_label
from elixir_echo.profiles import ObservationProfile
from elixir_echo.training.config import TrainingConfig
from elixir_echo.training.labels import LabelCategory


class FakeActionSequence:
    def __init__(self, **values):
        self.__dict__.update(values)

    def validate(self, config, *, candidate_count):
        assert candidate_count == config.max_action_candidates


@pytest.fixture
def components():
    config = SimpleNamespace(
        max_micro_actions=2,
        board_width=18,
        board_height=32,
        delay_offset_ms=(0, 50, 100),
        max_action_candidates=128,
    )
    types = SimpleNamespace(ActionSequenceV4=FakeActionSequence, GATE_WAIT=0, GATE_ACT=1)
    return SimpleNamespace(config=config, types=types)


def _training_step(action, *, actor=0):
    identity = DecisionIdentity("episode-test", 90, actor)
    observation = Observation({}, (), (), (), {"ended": False})
    return TrainingStep(
        identity=identity,
        observation=observation,
        action_mask=ActionMask(identity, {}),
        expert_action=ExpertAction.from_legacy(action),
        event_delta=(),
    )


def _tensorized(*, actor=0, card_id=26_000_001, target=(8, 10)):
    placements = torch.zeros((1, 2, 32, 18), dtype=torch.bool)
    x, y = target
    placements[0, 0, y, x] = True
    candidates = SimpleNamespace(
        native_visible_card_id=torch.tensor([[card_id, 26_000_003]], dtype=torch.long),
        mask=torch.tensor([[True, True]], dtype=torch.bool),
        placement=placements,
    )
    batch = SimpleNamespace(candidates=candidates)
    return SimpleNamespace(
        batch=batch,
        actor=actor,
        candidate_index_by_slot={1: 0},
        candidate_uid_by_slot={1: 431},
    )


def test_wait_label_has_only_gate_supervision(components):
    step = _training_step(None)

    compiled = compile_expert_label(
        step, _tensorized(), components, TrainingConfig(), ObservationProfile.RICH
    )

    assert compiled.category is LabelCategory.WAIT
    assert compiled.trainable is True
    assert compiled.action.gate.tolist() == [0]
    assert compiled.action.micro_action_count.tolist() == [0]
    assert compiled.supervision.gate is True
    assert compiled.supervision.candidate is False
    assert compiled.supervision.target is False
    assert compiled.supervision.continue_action is False
    assert compiled.supervision.delay is False
    assert compiled.supervision.value is False


def test_single_play_label_uses_the_paired_legal_candidate_and_grid(components):
    step = _training_step(
        {
            "kind": "play_card",
            "owner": 0,
            "card_id": 26_000_001,
            "hand_slot": 1,
            "target_kind": "grid",
            "target_grid": [8, 10],
            "execute_offset_ticks": 1,
        }
    )

    compiled = compile_expert_label(
        step, _tensorized(), components, TrainingConfig(), ObservationProfile.RICH
    )

    assert compiled.category is LabelCategory.SINGLE_PLAY_CARD
    assert compiled.trainable is True
    assert compiled.action.gate.tolist() == [1]
    assert compiled.action.micro_action_count.tolist() == [1]
    assert compiled.action.candidate_index.tolist() == [[0, -1]]
    assert compiled.action.candidate_uid.tolist() == [[431, -1]]
    assert compiled.action.target_cell.tolist() == [[188, -1]]
    assert compiled.action.delay_offset_bin.tolist() == [[1, -1]]
    assert compiled.supervision.gate is True
    assert compiled.supervision.candidate is True
    assert compiled.supervision.target is True
    assert compiled.supervision.continue_action is True
    assert compiled.supervision.delay is False
    assert compiled.supervision.value is False


def test_unsupported_ability_is_preserved_as_category_but_not_supervised(components):
    step = _training_step(
        {"kind": "activate_ability", "owner": 0, "entity_id": 44}
    )

    compiled = compile_expert_label(
        step, _tensorized(), components, TrainingConfig(), ObservationProfile.RICH
    )

    assert compiled.category is LabelCategory.SINGLE_ABILITY
    assert compiled.trainable is False
    assert compiled.action.gate.tolist() == [0]
    assert not any(compiled.supervision.to_dict().values())


def test_card_label_must_match_the_mask_candidate(components):
    step = _training_step(
        {
            "kind": "play_card",
            "owner": 0,
            "card_id": 26_000_002,
            "hand_slot": 1,
            "target_kind": "grid",
            "target_grid": [8, 10],
            "execute_offset_ticks": 1,
        }
    )

    compiled = compile_expert_label(
        step, _tensorized(), components, TrainingConfig(), ObservationProfile.RICH
    )

    assert compiled.category is LabelCategory.MALFORMED
    assert compiled.trainable is False
    assert "does not match" in compiled.report_reason
    assert compiled.action.gate.tolist() == [0]


def test_actor_one_grid_label_is_mirrored_into_firstlight_view(components):
    step = _training_step(
        {
            "kind": "play_card",
            "owner": 1,
            "card_id": 26_000_001,
            "hand_slot": 1,
            "target_kind": "grid",
            "target_grid": [8, 10],
            "execute_offset_ticks": 0,
        },
        actor=1,
    )

    compiled = compile_expert_label(
        step, _tensorized(actor=1, target=(9, 21)), components, TrainingConfig(), ObservationProfile.RICH
    )

    assert compiled.category is LabelCategory.SINGLE_PLAY_CARD
    assert compiled.action.target_cell.tolist() == [[387, -1]]
