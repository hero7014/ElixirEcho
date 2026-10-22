from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
from elixir_echo.integrations.firstlight.tensorizer import ObservationTensorizer, TensorizationError
from elixir_echo.integrations.firstlight.policy import FirstLightPolicyBackend
from elixir_echo.domain import DecisionIdentity
from elixir_echo.domain import ActionMask, Observation
from elixir_echo.integrations.firstlight.tensorizer import PolicyInputBuilder
from elixir_echo.policies.protocol import PolicyInput, RecurrentState
from elixir_echo.profiles import ObservationProfile
from elixir_echo.temporal import HistoryView, PreviousAction, PublicState, TemporalContext
from tests.fixtures.firstlight_inputs import action_mask_payload as _mask
from tests.fixtures.firstlight_inputs import observation_payload as _profile


@pytest.fixture(scope="module")
def runtime() -> FirstLightRuntime:
    root = os.environ.get("ELIXIR_ECHO_FIRSTLIGHT_ROOT")
    if not root:
        pytest.skip("set ELIXIR_ECHO_FIRSTLIGHT_ROOT to run provider compatibility tests")
    return FirstLightRuntime(root)


def test_core_and_full_produce_v4_shapes_and_preserve_legal_candidate_order(runtime) -> None:
    tensorizer = ObservationTensorizer(runtime)
    core = tensorizer.tensorize(_profile(rich=False), actor=0, action_mask=_mask())
    full = tensorizer.tensorize(_profile(rich=True), actor=0, action_mask=_mask())

    assert core.batch.match_scalars.shape == full.batch.match_scalars.shape
    assert core.batch.towers.mask.shape == (1, tensorizer.config.max_towers)
    assert core.batch.groups.child_mask.shape == (1, tensorizer.config.max_children_total)
    assert core.batch.explicit_spatial_planes.shape == (
        1,
        tensorizer.config.explicit_spatial_channels,
        tensorizer.config.board_height,
        tensorizer.config.board_width,
    )
    assert core.batch.candidates.native_hand_slot[0, core.batch.candidates.mask[0]].tolist() == [1, 3]
    assert core.candidate_index_by_slot.keys() == {1, 3}
    assert core.batch.candidates.placement[0, 0, 10, 7]
    assert core.batch.candidates.is_building[0, 0]
    assert core.batch.candidates.building_half_width[0, 0].item() == 1.0
    assert core.batch.candidates.building_half_height[0, 0].item() == 1.5
    assert core.batch.candidates.building_offset_x[0, 0].item() == 0.5
    assert core.batch.candidates.building_offset_y[0, 0].item() == -0.5
    assert core.batch.groups.child_features[0, 0, 9].item() == 0.0
    assert full.batch.groups.child_features[0, 0, 9].item() != 0.0
    assert full.batch.active_effects.mask.sum().item() == 1
    assert set(full.batch.relation_edges.relation_type[0, full.batch.relation_edges.mask[0]].tolist()) >= {1, 2}
    assert core.batch.events.mask.shape[1] == 0
    assert not core.batch.previous_action.action_mask.any()


def test_actor_one_towers_are_mirrored_and_evolved_entities_keep_their_form(runtime) -> None:
    tensorizer = ObservationTensorizer(runtime)
    profile = _profile(rich=False)
    profile["towers"] = [
        {
            "entity_id": 2,
            "owner": 1,
            "tower_kind": "princess_left",
            "position": [5_000.0, 20_000.0],
            "hitpoints": 2_000.0,
            "max_hitpoints": 2_000.0,
            "active": True,
            "shield": {"hitpoints": 0.0},
        }
    ]
    profile["entities"] = [
        {
            **profile["entities"][0],
            "current_form": {"status": "known", "value": {"variant": "evolved", "form_id": "evolved"}},
        }
    ]

    decision = tensorizer.tensorize(profile, actor=1, action_mask=_mask())

    assert decision.batch.towers.tower_type[0, 0].item() == tensorizer.TOWER_TYPES["princess_right"]
    assert decision.batch.groups.runtime_form[0, 0].item() == tensorizer.FORM_EVOLUTION


def test_rich_hero_active_form_and_attack_charge_fields_reach_v4(runtime) -> None:
    tensorizer = ObservationTensorizer(runtime)
    profile = _profile(rich=True)
    profile["entities"] = [
        {
            **profile["entities"][0],
            "entity_kind": "hero",
            "abilities": ({"phase": "active"},),
            "attack": {"phase": "charging", "charge_stage": 4, "charge_elapsed_ms": 2_500},
        }
    ]

    decision = tensorizer.tensorize(profile, actor=0, action_mask=_mask())

    assert decision.batch.groups.runtime_form[0, 0].item() == tensorizer.FORM_HERO_ACTIVE
    assert decision.batch.groups.child_features[0, 0, 22].item() == pytest.approx(0.4)
    assert decision.batch.groups.child_features[0, 0, 23].item() == pytest.approx(0.5)


def test_numeric_shield_hitpoints_reach_tower_entity_and_spatial_features(runtime) -> None:
    tensorizer = ObservationTensorizer(runtime)
    profile = _profile(rich=False)
    profile["towers"][0]["shield"] = {"hitpoints": 384.0}
    profile["entities"][0]["shield"] = {"hitpoints": 384.0}

    decision = tensorizer.tensorize(profile, actor=0, action_mask=_mask())

    assert decision.batch.towers.features[0, 0, 1].item() == pytest.approx(384 / 4_000)
    assert decision.batch.groups.features[0, 0, 3].item() == pytest.approx(384 / 10_000)
    assert decision.batch.groups.child_features[0, 0, 3].item() == pytest.approx(384 / 10_000)
    assert decision.batch.explicit_spatial_planes[0, 5].sum().item() == pytest.approx(384 / 10_000)


@pytest.mark.parametrize(
    "action",
    [
        {
            "kind": "play_card",
            "owner": 0,
            "card_id": 26_000_000,
            "hand_slot": 1,
            "target_kind": "grid",
            "target_grid": (8.5, 12.0),
            "execute_offset_ticks": 0,
        },
        {
            "kind": "play_card",
            "owner": 0,
            "card_id": 26_000_000,
            "hand_slot": 1,
            "target_kind": "grid",
            "target_grid": (8, 12),
        },
    ],
    ids=("fractional-target", "unknown-delay"),
)
def test_unrepresentable_known_previous_action_becomes_masked_unknown(runtime, action) -> None:
    tensorizer = ObservationTensorizer(runtime)
    temporal = _temporal(PreviousAction.known((action,), 90))

    decision = tensorizer.tensorize(
        _profile(rich=False), actor=0, action_mask=_mask(), temporal_context=temporal
    )

    assert decision.batch.match_scalars[0, tensorizer.PREVIOUS_ACTION_STATUS_SCALAR].item() == 0.5
    assert not decision.batch.previous_action.action_mask.any()
    model_input = decision.batch.to_model_input("cpu")
    assert model_input.match_scalars[0, tensorizer.PREVIOUS_ACTION_STATUS_SCALAR].item() == 0.5


def test_legal_candidate_requires_explicit_effective_cost(runtime) -> None:
    tensorizer = ObservationTensorizer(runtime)
    action_mask = _mask()
    del action_mask["placement_masks"]["1"]["effective_cost"]

    with pytest.raises(TensorizationError, match="effective_cost"):
        tensorizer.tensorize(_profile(rich=False), actor=0, action_mask=action_mask)


def _temporal(previous: PreviousAction) -> TemporalContext:
    identity = DecisionIdentity("episode-test", 95, 0)
    history = HistoryView(identity, 90, (), (), previous)
    state = PublicState("episode-test", 0, 95, (), (), ())
    return TemporalContext(identity, history, state)


def test_previous_action_start_wait_and_unrepresentable_states_stay_distinct(runtime) -> None:
    tensorizer = ObservationTensorizer(runtime)
    start = tensorizer.tensorize(_profile(rich=False), actor=0, action_mask=_mask(), temporal_context=_temporal(PreviousAction.start()))
    wait = tensorizer.tensorize(_profile(rich=False), actor=0, action_mask=_mask(), temporal_context=_temporal(PreviousAction.known((), 90)))
    unknown = tensorizer.tensorize(_profile(rich=False), actor=0, action_mask=_mask(), temporal_context=_temporal(PreviousAction.unknown_masked(90)))

    assert start.batch.match_scalars[0, tensorizer.PREVIOUS_ACTION_STATUS_SCALAR].item() == 0.0
    assert wait.batch.match_scalars[0, tensorizer.PREVIOUS_ACTION_STATUS_SCALAR].item() == 1.0
    assert unknown.batch.match_scalars[0, tensorizer.PREVIOUS_ACTION_STATUS_SCALAR].item() == 0.5
    assert start.batch.previous_action.gate[0].item() == wait.batch.previous_action.gate[0].item()
    assert not unknown.batch.previous_action.action_mask.any()
    assert not wait.batch.previous_action.action_mask.any()


def test_firstlight_backend_rejects_recurrent_state_from_another_episode() -> None:
    backend = FirstLightPolicyBackend.__new__(FirstLightPolicyBackend)
    identity = DecisionIdentity("episode-current", 100, 0)
    policy_input = PolicyInput(identity, "core", type("Decision", (), {"actor": 0})())
    state = RecurrentState(
        payload=object(),
        episode_id="episode-previous",
        last_identity=DecisionIdentity("episode-previous", 95, 0),
    )

    with pytest.raises(ValueError, match="recurrent state crossed an episode boundary"):
        backend.step(policy_input, state)


def test_firstlight_backend_rejects_recurrent_state_from_another_actor() -> None:
    backend = FirstLightPolicyBackend.__new__(FirstLightPolicyBackend)
    identity = DecisionIdentity("episode-current", 100, 0)
    policy_input = PolicyInput(identity, "core", type("Decision", (), {"actor": 0})())
    state = RecurrentState(
        payload=object(),
        episode_id="episode-current",
        last_identity=DecisionIdentity("episode-current", 95, 1),
    )

    with pytest.raises(ValueError, match="recurrent state crossed an actor boundary"):
        backend.step(policy_input, state)


def test_firstlight_backend_rejects_non_advancing_recurrent_tick() -> None:
    backend = FirstLightPolicyBackend.__new__(FirstLightPolicyBackend)
    identity = DecisionIdentity("episode-current", 90, 0)
    policy_input = PolicyInput(identity, "core", type("Decision", (), {"actor": 0})())
    state = RecurrentState(
        payload=object(),
        episode_id="episode-current",
        last_identity=DecisionIdentity("episode-current", 95, 0),
    )

    with pytest.raises(ValueError, match="tick must advance"):
        backend.step(policy_input, state)


def test_policy_input_builder_binds_observation_mask_profile_and_temporal_identity(runtime) -> None:
    raw = _profile(rich=True)
    observation = Observation(
        match=raw["match"],
        players=tuple(raw["players"]),
        towers=tuple(raw["towers"]),
        entities=tuple(raw["entities"]),
        terminal={"ended": False},
    )
    temporal = _temporal(PreviousAction.start())
    action_mask = ActionMask(temporal.identity, _mask())

    policy_input = PolicyInputBuilder(runtime).build(
        observation, action_mask, temporal, ObservationProfile.MIN
    )

    assert policy_input.identity == temporal.identity
    assert policy_input.profile == "core"
    assert policy_input.tensors.actor == temporal.identity.actor
    assert policy_input.tensors.batch.groups.child_features[0, 0, 9].item() == 0.0

    mismatched = ActionMask(DecisionIdentity("other-episode", 95, 0), _mask())
    with pytest.raises(ValueError, match="ActionMask identity"):
        PolicyInputBuilder(runtime).build(observation, mismatched, temporal, ObservationProfile.MIN)
