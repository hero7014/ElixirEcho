from __future__ import annotations

from dataclasses import replace

import pytest

from elixir_echo.profiles import ObservationProfile
from elixir_echo.training.config import TrainingConfig, supervision_for
from elixir_echo.training.labels import LabelCategory


def test_first_pass_freezes_full_profile_wait_and_single_card_labels():
    config = TrainingConfig()

    assert config.profile is ObservationProfile.RICH
    assert config.supported_label_categories == (
        LabelCategory.WAIT,
        LabelCategory.SINGLE_PLAY_CARD,
    )
    assert config.delay_loss_weight == 0.0
    assert config.value_loss_weight == 0.0
    assert supervision_for(LabelCategory.WAIT, config).to_dict() == {
        "gate": True,
        "candidate": False,
        "target": False,
        "continue_action": False,
        "delay": False,
        "value": False,
    }


def test_training_fingerprint_covers_gate_act_weight_and_profile():
    config = TrainingConfig()

    assert config.fingerprint() != replace(config, gate_act_weight=4.0).fingerprint()
    assert config.fingerprint() != replace(config, profile=ObservationProfile.MIN).fingerprint()
    assert config.fingerprint() != replace(config, tbptt_length=16).fingerprint()
    assert config.to_dict()["tbptt_length"] == 32


def test_training_config_serializes_each_label_head_mask_for_checkpoint_identity():
    config = TrainingConfig()
    payload = config.to_dict()

    assert payload["supervision_masks"] == {
        "wait": {
            "gate": True,
            "candidate": False,
            "target": False,
            "continue_action": False,
            "delay": False,
            "value": False,
        },
        "single_play_card": {
            "gate": True,
            "candidate": True,
            "target": True,
            "continue_action": True,
            "delay": False,
            "value": False,
        },
        "single_ability": {
            "gate": False,
            "candidate": False,
            "target": False,
            "continue_action": False,
            "delay": False,
            "value": False,
        },
        "multi_action": {
            "gate": False,
            "candidate": False,
            "target": False,
            "continue_action": False,
            "delay": False,
            "value": False,
        },
        "malformed_or_unrepresentable": {
            "gate": False,
            "candidate": False,
            "target": False,
            "continue_action": False,
            "delay": False,
            "value": False,
        },
    }


def test_training_fingerprint_covers_head_supervision_policy():
    default = TrainingConfig()
    gate_only = TrainingConfig(
        supported_label_categories=(LabelCategory.WAIT,),
    )

    assert default.to_dict()["supervision_masks"] != gate_only.to_dict()["supervision_masks"]
    assert default.fingerprint() != gate_only.fingerprint()


def test_first_pass_cannot_enable_ability_supervision():
    with pytest.raises(ValueError, match="only WAIT and single play_card"):
        TrainingConfig(
            supported_label_categories=(
                LabelCategory.WAIT,
                LabelCategory.SINGLE_ABILITY,
            )
        )


@pytest.mark.parametrize(
    "values",
    [
        {"delay_loss_weight": 0.1},
        {"value_loss_weight": 0.1},
        {"learning_rate": 0.0},
        {"gate_act_weight": 0.0},
        {"tbptt_length": 0},
        {"tbptt_length": True},
        {"tbptt_length": 1.5},
    ],
)
def test_first_pass_rejects_out_of_scope_or_invalid_training_config(values):
    with pytest.raises(ValueError):
        TrainingConfig(**values)
