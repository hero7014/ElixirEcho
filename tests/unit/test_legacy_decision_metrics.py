from __future__ import annotations

import pytest

from tools.evaluate_legacy_decisions import DecisionCounts, decision_metrics, episode_bootstrap


def test_decision_metrics_keep_rare_act_denominators_separate() -> None:
    counts = DecisionCounts(
        frames=100, wait=90, play=10,
        true_wait=80, false_act=10, false_wait=8, true_act=2,
        candidate_trials=10, candidate_hits=6, target_trials=10, target_hits=4,
        card_and_target_hits=3, full_decision_hits=1,
        random_candidate_expected_hits=2.5, random_target_expected_hits=0.5,
    )
    metrics = decision_metrics(counts)
    assert metrics["gate_accuracy"] == pytest.approx(0.82)
    assert metrics["always_wait_gate_accuracy"] == pytest.approx(0.90)
    assert metrics["act_recall"] == pytest.approx(0.20)
    assert metrics["candidate_accuracy_given_expert_act"] == pytest.approx(0.60)
    assert metrics["target_accuracy_given_expert_card"] == pytest.approx(0.40)
    assert metrics["card_and_target_accuracy_given_expert_act"] == pytest.approx(0.30)
    assert metrics["full_decision_accuracy_given_expert_act"] == pytest.approx(0.10)


def test_episode_bootstrap_uses_episode_units_and_is_repeatable() -> None:
    episodes = {
        str(index): DecisionCounts(wait=10, play=2, true_wait=9, false_act=1,
                                   true_act=index % 2, false_wait=2 - index % 2,
                                   candidate_trials=2, candidate_hits=index % 2,
                                   target_trials=2, card_and_target_hits=index % 2)
        for index in range(10)
    }
    first = episode_bootstrap(episodes, draws=100, seed=7)
    assert first == episode_bootstrap(episodes, draws=100, seed=7)
    assert 0 <= first["act_recall"][0] <= first["act_recall"][1] <= 1
    assert 0 <= first["candidate_accuracy_given_expert_act"][0] <= 1
