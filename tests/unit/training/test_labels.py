from __future__ import annotations

import pytest

from elixir_echo.domain import ExpertAction
from elixir_echo.training.labels import LabelCategory, classify_expert_action


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (ExpertAction.wait(), LabelCategory.WAIT),
        (
            ExpertAction.from_legacy(
                {"kind": "play_card", "owner": 0, "card_id": 26_000_001}
            ),
            LabelCategory.SINGLE_PLAY_CARD,
        ),
        (
            ExpertAction.from_legacy(
                {"kind": "activate_ability", "owner": 0, "entity_id": 44}
            ),
            LabelCategory.SINGLE_ABILITY,
        ),
        (
            ExpertAction.from_legacy(
                [
                    {"kind": "play_card", "owner": 0},
                    {"kind": "activate_ability", "owner": 0},
                ]
            ),
            LabelCategory.MULTI_ACTION,
        ),
    ],
)
def test_expert_action_is_classified_without_inferring_unsupported_labels(action, expected):
    assert classify_expert_action(action) is expected


def test_malformed_expert_action_has_its_own_category():
    action = ExpertAction.from_legacy({"not": "an action list"})

    assert classify_expert_action(action) is LabelCategory.MALFORMED
