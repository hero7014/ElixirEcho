from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import importlib
import json
import os
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

from elixir_echo.datasets.trajectory import TrainingStep
from elixir_echo.domain import ActionMask, DecisionIdentity, ExpertAction, Observation
from elixir_echo.domain.values import content_sha256, jsonable
from elixir_echo.integrations.firstlight.policy import FirstLightPolicyBackend
from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
from elixir_echo.integrations.firstlight.tensorizer import PolicyInputBuilder
from elixir_echo.profiles import ObservationProfile, select_observation
from elixir_echo.temporal import PublicStateTracker


def _assert_tree_close(expected, actual, path: str = "batch") -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: tensor type differs"
        assert expected.dtype == actual.dtype, f"{path}: dtype differs"
        torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7, msg=path)
        return
    if is_dataclass(expected):
        assert is_dataclass(actual), f"{path}: dataclass type differs"
        assert tuple(item.name for item in fields(expected)) == tuple(
            item.name for item in fields(actual)
        ), f"{path}: field set differs"
        for item in fields(expected):
            _assert_tree_close(
                getattr(expected, item.name),
                getattr(actual, item.name),
                f"{path}.{item.name}",
            )
        return
    assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def _action_logits(model, batch, state, *, device: torch.device):
    context = model.forward(
        batch,
        state,
        episode_start=torch.ones(1, dtype=torch.bool, device=device),
        validate=True,
    )
    gate = model.gate_head(context.policy_context)
    decoder = model._decoder_context(
        model.decoder_initial(context.policy_context), context.encoded
    )
    candidate = model.candidate_logits(decoder, context.encoded.candidate_memory)
    return gate, candidate


def test_legacy_semantic_frame_path_audits_elixirecho_checkpoint_compatibility() -> None:
    baseline_root = os.environ.get("ELIXIR_ECHO_BASELINE_ROOT")
    episode_path = os.environ.get("ELIXIR_ECHO_LEGACY_EPISODE")
    provider_root = os.environ.get("ELIXIR_ECHO_FIRSTLIGHT_ROOT")
    checkpoint = os.environ.get("ELIXIR_ECHO_FIRSTLIGHT_CHECKPOINT")
    if not all((baseline_root, episode_path, provider_root, checkpoint)):
        pytest.skip(
            "set baseline, legacy episode, FirstLight root and checkpoint paths for old/new parity"
        )

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    baseline_src = Path(baseline_root).resolve() / "src"
    if not (baseline_src / "training" / "tensorizer_v1.py").is_file():
        pytest.skip("configured baseline checkout does not contain training/tensorizer_v1.py")
    sys.path.insert(0, str(baseline_src))
    try:
        legacy_cache = importlib.import_module("training.cache_reader")
        legacy_runtime_module = importlib.import_module("training.firstlight_v4")
        legacy_tensorizer_module = importlib.import_module("training.tensorizer_v1")
        legacy_history_module = importlib.import_module("training.history")
        legacy_tracker_module = importlib.import_module("training.deterministic_tracker")
        legacy_temporal_module = importlib.import_module("training.temporal_v4_input")
        legacy_profiles = importlib.import_module("semantic_frame.profiles")
        legacy_cache_record = importlib.import_module("semantic_frame.cache_record")
        legacy_contracts = importlib.import_module("semantic_frame.contracts")

        record_file = Path(episode_path).resolve() / "records.jsonl"
        # Spread probes across the same actor's full episode, including ticks after
        # card plays/abilities and late-game event history. Trackers still consume
        # every preceding record so the sampled temporal state is causal.
        sample_actor_indices = {0, 32, 124, 149, 493, 702}
        legacy_runtime = legacy_runtime_module.load_firstlight_v4(provider_root)
        old_history = legacy_history_module.BasicHistoryV1()
        old_tracker = legacy_tracker_module.DeterministicPublicTrackerV1()
        provider = FirstLightRuntime(provider_root)
        backend = FirstLightPolicyBackend(provider, device="cpu", checkpoint=checkpoint)
        old_tensorizer = legacy_tensorizer_module.SemanticFrameTensorizerV1(legacy_runtime)
        new_tracker = PublicStateTracker()
        new_input_builder = PolicyInputBuilder(provider, components=backend.components)
        backend.model.eval()
        zero_state = backend.initial_state().payload
        actor_index = 0
        compared: list[tuple[int, int]] = []
        reserved_feature_nonzero = 0
        shield_feature_differences = 0
        max_shield_feature_delta = 0.0
        max_live_logit_delta = 0.0
        with record_file.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record_payload = json.loads(line)
                stored_hash = record_payload.pop("content_sha256", None)
                assert stored_hash == content_sha256(record_payload), (
                    f"source record hash differs at line {line_number}"
                )
                raw_identity = record_payload["identity"]
                if raw_identity["actor"] != 0:
                    continue

                frame = record_payload["frame"]
                identity = DecisionIdentity(
                    episode_id=raw_identity["episode_id"],
                    tick=raw_identity["tick"],
                    actor=raw_identity["actor"],
                )
                legacy_identity = legacy_cache_record.CacheIdentityV1(
                    identity.episode_id, identity.tick, identity.actor
                )
                legacy_frame = legacy_contracts.SemanticFrameV1(
                    match=frame["match"],
                    players=tuple(frame["players"]),
                    towers=tuple(frame["towers"]),
                    entities=tuple(frame["entities"]),
                    terminal=frame["terminal"],
                    perspective=frame["perspective"],
                    version=frame["version"],
                )
                old_record = legacy_cache_record.SemanticFrameCacheRecord(
                    identity=legacy_identity,
                    episode_context_ref=record_payload["episode_context_ref"],
                    frame=legacy_frame,
                    action_mask=record_payload["action_mask"],
                    expert_action=record_payload.get("expert_action"),
                    event_delta=tuple(record_payload.get("event_delta", ())),
                    validation=record_payload["validation"],
                    fingerprints=record_payload["fingerprints"],
                )
                new_observation = Observation(
                    match=frame["match"],
                    players=tuple(frame["players"]),
                    towers=tuple(frame["towers"]),
                    entities=tuple(frame["entities"]),
                    terminal=frame["terminal"],
                    perspective=frame["perspective"],
                )
                new_action_mask = record_payload["action_mask"]
                mask_constraints = {
                    key: value
                    for key, value in new_action_mask.items()
                    if key not in {"schema", "identity"}
                }
                action_mask = ActionMask(identity, mask_constraints)
                expert_action = ExpertAction.from_legacy(record_payload.get("expert_action"))
                training_step = TrainingStep(
                    identity=identity,
                    observation=new_observation,
                    action_mask=action_mask,
                    expert_action=expert_action,
                    event_delta=tuple(record_payload.get("event_delta", ())),
                )

                if actor_index == 0:
                    old_history.reset(old_record.identity)
                    old_tracker.reset(identity.episode_id, identity.actor, identity.tick)
                old_view = old_history.before_tick(old_record)
                old_tracker.apply_public_events(identity.tick, old_view.current_public_events)
                temporal_context = new_tracker.before_step(training_step)

                old_tracker_state = old_tracker.snapshot()
                new_tracker_state = temporal_context.public_state
                assert old_tracker_state.revealed_cards == new_tracker_state.revealed_cards
                assert [
                    (item.tick, item.owner, item.card_id)
                    for item in old_tracker_state.public_play_history
                ] == [
                    (item.tick, item.owner, item.card_id)
                    for item in new_tracker_state.public_play_history
                ]
                assert [
                    (item.card_id, item.count, item.capped)
                    for item in old_tracker_state.public_plays_since_last_use
                ] == [
                    (item.card_id, item.count, item.capped)
                    for item in new_tracker_state.public_plays_since_last_use
                ]
                assert old_view.previous_action.status.value == (
                    temporal_context.history.previous_action.status.value
                )
                assert jsonable(old_view.previous_action.actions) == jsonable(
                    temporal_context.history.previous_action.actions
                )
                assert old_view.previous_action.source_tick == (
                    temporal_context.history.previous_action.source_tick
                )

                if actor_index in sample_actor_indices:
                    old_projection = jsonable(
                        legacy_profiles.project_observation(
                            legacy_frame, legacy_profiles.ObservationProfile.RICH
                        )
                    )
                    new_projection = jsonable(
                        select_observation(new_observation, ObservationProfile.RICH)
                    )
                    for projection, envelope_keys in (
                        (old_projection, ("profile", "semantic_frame_version")),
                        (new_projection, ("profile", "observation_schema")),
                    ):
                        for key in envelope_keys:
                            projection.pop(key, None)
                    assert old_projection == new_projection, (
                        f"RICH semantic projection differs at actor index {actor_index}"
                    )
                    typed_mask = jsonable(action_mask.to_dict())
                    typed_mask.pop("schema", None)
                    assert typed_mask == jsonable(new_action_mask), (
                        f"ActionMask differs at actor index {actor_index}"
                    )
                    old_example = legacy_cache.TrainingExampleV1(
                        identity=legacy_identity,
                        profile=legacy_profiles.project_observation(
                            legacy_frame, legacy_profiles.ObservationProfile.RICH
                        ),
                        action_mask=record_payload["action_mask"],
                        expert_action=record_payload.get("expert_action"),
                        source_record=f"{record_file}:{line_number}",
                    )
                    old_tensors = old_tensorizer.tensorize(old_example)
                    old_composed = legacy_temporal_module.compose_temporal_v4_input(
                        legacy_runtime,
                        old_tensors,
                        old_tracker_state,
                        old_view.previous_action,
                    )
                    new_input = new_input_builder.build(
                        new_observation,
                        action_mask,
                        temporal_context,
                        ObservationProfile.RICH,
                    )
                    # The legacy checkpoint's tensorizer leaves child feature
                    # channels 36:45 at zero. ElixirEcho populates the richer
                    # RICH-only runtime extension there. All other tensor values,
                    # including shield HP, must match the legacy input directly.
                    live_batch = new_input.tensors.batch
                    extension_features = live_batch.groups.child_features
                    reserved_feature_nonzero += int(
                        (extension_features[:, :, 36:45] != 0).sum().item()
                    )
                    old_group_features = old_composed.batch.groups.features
                    new_group_features = live_batch.groups.features
                    old_child_features = old_composed.batch.groups.child_features
                    new_child_features = live_batch.groups.child_features
                    shield_group_delta = (
                        old_group_features[:, :, 3] - new_group_features[:, :, 3]
                    ).abs()
                    shield_child_delta = (
                        old_child_features[:, :, 3] - new_child_features[:, :, 3]
                    ).abs()
                    shield_feature_differences += int(
                        (shield_group_delta > 1e-7).sum().item()
                        + (shield_child_delta > 1e-7).sum().item()
                    )
                    max_shield_feature_delta = max(
                        max_shield_feature_delta,
                        float(shield_group_delta.max().item()),
                        float(shield_child_delta.max().item()),
                    )
                    old_spatial = old_composed.batch.explicit_spatial_planes
                    new_spatial = live_batch.explicit_spatial_planes
                    shield_spatial_delta = (
                        old_spatial[:, (5, 12)] - new_spatial[:, (5, 12)]
                    ).abs()
                    shield_feature_differences += int(
                        (shield_spatial_delta > 1e-7).sum().item()
                    )
                    max_shield_feature_delta = max(
                        max_shield_feature_delta,
                        float(shield_spatial_delta.max().item()),
                    )
                    baseline_group_features = new_group_features.clone()
                    baseline_child_features = new_child_features.clone()
                    baseline_child_features[:, :, 36:45] = 0
                    baseline_groups = replace(
                        live_batch.groups,
                        features=baseline_group_features,
                        child_features=baseline_child_features,
                    )
                    baseline_batch = replace(
                        live_batch,
                        groups=baseline_groups,
                    )
                    _assert_tree_close(
                        old_composed.batch,
                        baseline_batch,
                        path=f"actor_index={actor_index} legacy-compatible batch",
                    )

                    old_model_input = old_composed.batch.to_model_input(backend.device)
                    new_model_input = live_batch.to_model_input(backend.device)
                    baseline_model_input = baseline_batch.to_model_input(backend.device)
                    with torch.inference_mode():
                        old_logits = _action_logits(
                            backend.model, old_model_input, zero_state, device=backend.device
                        )
                        new_logits = _action_logits(
                            backend.model, new_model_input, zero_state, device=backend.device
                        )
                        baseline_logits = _action_logits(
                            backend.model,
                            baseline_model_input,
                            zero_state,
                            device=backend.device,
                        )
                    torch.testing.assert_close(
                        old_logits[0], baseline_logits[0], rtol=1e-6, atol=1e-7
                    )
                    torch.testing.assert_close(
                        old_logits[1], baseline_logits[1], rtol=1e-6, atol=1e-7
                    )
                    live_delta = max(
                        float((old_logits[0] - new_logits[0]).abs().max().item()),
                        float((old_logits[1] - new_logits[1]).abs().max().item()),
                    )
                    max_live_logit_delta = max(max_live_logit_delta, live_delta)
                    compared.append((identity.tick, identity.actor))

                old_history.after_label(old_record)
                new_tracker.after_action(expert_action)
                actor_index += 1

        assert actor_index == 703, f"expected all 703 actor-0 records, got {actor_index}"
        assert compared == [
            (90, 0), (250, 0), (710, 0), (835, 0), (2555, 0), (3600, 0)
        ], f"unexpected parity sample identities: {compared}"
        assert reserved_feature_nonzero > 0, "sample did not exercise any RICH-only extension"
        assert shield_feature_differences == 0, (
            f"shield values still differ from legacy inputs: "
            f"count={shield_feature_differences}, max_delta={max_shield_feature_delta}"
        )
        print(
            "RICH compatibility audit: "
            f"samples={len(compared)}, actor0_records={actor_index}, "
            f"nonzero_legacy-reserved_child_features={reserved_feature_nonzero}, "
            f"shield_feature_differences={shield_feature_differences}, "
            f"max_shield_feature_delta={max_shield_feature_delta:.9g}, "
            f"max_live_checkpoint_logit_delta={max_live_logit_delta:.9g}"
        )
    finally:
        try:
            sys.path.remove(str(baseline_src))
        except ValueError:
            pass
