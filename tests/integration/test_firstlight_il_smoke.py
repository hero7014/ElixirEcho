from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")

from elixir_echo.datasets.trajectory import TrainingEpisode, TrainingStep
from elixir_echo.domain import ActionMask, DecisionIdentity, ExpertAction, Observation
from elixir_echo.integrations.firstlight.il import compile_firstlight_actor_sequence
from elixir_echo.integrations.firstlight.policy import FirstLightPolicyBackend
from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
from elixir_echo.integrations.firstlight.tensorizer import PolicyInputBuilder
from elixir_echo.integrations.firstlight.trainer import FirstLightILTrainer
from elixir_echo.profiles import ObservationProfile
from elixir_echo.training.config import TrainingConfig


def _observation(actor: int) -> Observation:
    self_player = {
        "owner": actor,
        "relation": "self",
        "crowns": 0,
        "tower_ids": [1],
        "self_private": {
            "elixir": 5.0,
            "hand": [26_000_000, 26_000_001, 26_000_002, 26_000_003],
            "next_card": 26_000_004,
        },
        "current_evolution": [],
    }
    other_player = {
        "owner": 1 - actor,
        "relation": "opponent",
        "crowns": 0,
        "tower_ids": [2],
    }
    towers = (
        {
            "entity_id": 1,
            "owner": actor,
            "tower_kind": "king",
            "position": [9_000.0, 30_000.0 if actor == 0 else 2_000.0],
            "hitpoints": 4_000.0,
            "max_hitpoints": 4_000.0,
            "tower_troop_id": 159_000_000,
            "active": True,
            "shield": {"hitpoints": 0.0},
        },
    )
    entity = {
        "entity_id": 44,
        "owner": actor,
        "card_id": 26_000_000,
        "current_form": {"status": "known", "value": {"variant": "base", "form_id": "base"}},
        "entity_kind": "troop",
        "position": [9_000.0, 12_000.0 if actor == 0 else 20_000.0],
        "hitpoints": 400.0,
        "max_hitpoints": 500.0,
        "shield": {"hitpoints": 0.0},
        "visibility_basic": "visible",
        "velocity": [0.0, 0.0],
        "age_ms": 100,
        "public_target_id": None,
        "source_entity_id": None,
        "causal_relation": {"kind": "deployment", "source_card_id": 26_000_000},
        "attack": {"phase": "unknown"},
        "effects": [],
        "projectile": None,
        "visibility": {},
        "abilities": [],
        "evolution": None,
    }
    return Observation(
        match={"phase": "normal", "remaining_ms": 180_000, "elixir_multiplier": 1.0},
        players=(self_player, other_player),
        towers=towers,
        entities=(entity,),
        terminal={"ended": False},
    )


def _mask(identity: DecisionIdentity) -> ActionMask:
    placement = [[True] * 18 for _ in range(32)]
    return ActionMask(
        identity,
        {
            "hand_slots": [False, True, False, False],
            "placement_masks": {
                "1": {
                    "visible_card_id": 26_000_001,
                    "effective_card_id": 26_000_001,
                    "effective_cost": 3.0,
                    "form_code": 0,
                    "row_major": placement,
                }
            },
            "reasons": {"effective_elixir": 5.0},
        },
    )


def _step(tick: int, actor: int, action) -> TrainingStep:
    identity = DecisionIdentity("il-smoke-episode", tick, actor)
    return TrainingStep(
        identity=identity,
        observation=_observation(actor),
        action_mask=_mask(identity),
        expert_action=ExpertAction.from_legacy(action),
        event_delta=(),
    )


@pytest.mark.firstlight
@pytest.mark.skipif(
    os.environ.get("ELIXIR_ECHO_RUN_TRAINING_SMOKE") != "1",
    reason="set ELIXIR_ECHO_RUN_TRAINING_SMOKE=1 to opt in to one optimizer step",
)
def test_full_profile_sequence_runs_firstlight_v4_il_and_checkpoint_roundtrip(tmp_path):
    root = os.environ.get("ELIXIR_ECHO_FIRSTLIGHT_ROOT")
    if not root:
        pytest.skip("set ELIXIR_ECHO_FIRSTLIGHT_ROOT to run the FirstLight IL smoke")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime = FirstLightRuntime(root)
    backend = FirstLightPolicyBackend(runtime, device=device)
    config = TrainingConfig(profile=ObservationProfile.RICH, tbptt_length=1)
    play = {
        "kind": "play_card",
        "owner": 0,
        "card_id": 26_000_001,
        "hand_slot": 1,
        "target_kind": "grid",
        "target_grid": [8, 10],
        "execute_offset_ticks": 0,
    }
    episode = TrainingEpisode(
        episode_id="il-smoke-episode",
        outcome="terminated",
        split="train",
        steps=(
            _step(90, 0, play),
            _step(90, 1, None),
            _step(95, 0, None),
            _step(95, 1, None),
        ),
    )
    compiled = compile_firstlight_actor_sequence(
        episode,
        actor=0,
        input_builder=PolicyInputBuilder(runtime, components=backend.components),
        components=backend.components,
        config=config,
    )

    trainer = FirstLightILTrainer(backend, config)
    result = trainer.tiny_overfit(compiled, optimizer_steps=1)
    optimizer = torch.optim.AdamW(
        backend.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    recurrent_result = trainer.train_sequence(compiled, optimizer=optimizer)

    assert result.sequence_shape == (2, 1)
    assert result.profile == "full"
    assert result.trainable_parameters > 0
    assert result.optimizer_steps == 1
    assert len(result.loss_history) == 2
    assert all(math.isfinite(item) for item in result.loss_history)
    assert math.isfinite(result.last_gradient_norm)
    assert result.head_counts["gate"] == 2
    assert result.head_counts["candidate"] == 1
    assert result.head_counts["target"] == 1
    assert result.head_counts["continue_action"] == 1
    assert result.head_counts["delay"] == 0
    assert result.head_counts["value"] == 0
    assert recurrent_result.chunks == recurrent_result.optimizer_steps == 2
    assert recurrent_result.tbptt_length == 1
    assert all(math.isfinite(item) for item in recurrent_result.loss_history)
    assert all(math.isfinite(item) for item in recurrent_result.gradient_norms)
    expected_parameters = torch.nn.utils.parameters_to_vector(backend.model.parameters()).detach().clone()
    optimizer_state_count = len(optimizer.state)
    checkpoint_path = tmp_path / "synthetic-checkpoint"
    saved = trainer.save_checkpoint(
        checkpoint_path,
        optimizer=optimizer,
        update_step=recurrent_result.optimizer_steps,
    )
    with torch.no_grad():
        next(backend.model.parameters()).add_(1.0)
    optimizer.state.clear()

    restored = trainer.restore_checkpoint(checkpoint_path, optimizer=optimizer)

    actual_parameters = torch.nn.utils.parameters_to_vector(backend.model.parameters()).detach()
    assert torch.equal(actual_parameters, expected_parameters)
    assert len(optimizer.state) == optimizer_state_count
    assert saved.checkpoint_id == restored.checkpoint_id
    assert restored.update_step == recurrent_result.optimizer_steps
