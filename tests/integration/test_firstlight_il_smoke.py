from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import fields, is_dataclass, replace

import pytest

torch = pytest.importorskip("torch")

from elixir_echo.datasets.trajectory import (
    TrainingEpisode,
    TrainingStep,
    TrajectoryEpisode,
    TrajectoryManifest,
    aggregate_record_hash,
)
from elixir_echo.domain import (
    ActionMask,
    DecisionIdentity,
    DecisionStep,
    ExpertAction,
    Observation,
)
from elixir_echo.domain.values import content_sha256
from elixir_echo.integrations.firstlight.il import compile_firstlight_actor_sequence
from elixir_echo.integrations.firstlight.policy import FirstLightPolicyBackend
from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
from elixir_echo.integrations.firstlight.sequence_cache import (
    FirstLightSequenceCacheError,
    _HEADER,
    _MAGIC,
    decode_actor_sequence,
    encode_actor_sequence,
)
from elixir_echo.integrations.firstlight.tensorizer import PolicyInputBuilder
from elixir_echo.integrations.firstlight.trainer import FirstLightILTrainer, TrainingRunProgress
from elixir_echo.datasets.sequence import (
    Layer2SequenceCache,
    SequenceDataset,
    SequenceDatasetCompiler,
)
from elixir_echo.profiles import ObservationProfile
from elixir_echo.training.config import HeadSupervision, TrainingConfig
from elixir_echo.training.labels import LabelCategory
from elixir_echo.domain.values import canonical_json


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


def _step(tick: int, actor: int, action, *, episode_id: str = "il-smoke-episode") -> TrainingStep:
    identity = DecisionIdentity(episode_id, tick, actor)
    return TrainingStep(
        identity=identity,
        observation=_observation(actor),
        action_mask=_mask(identity),
        expert_action=ExpertAction.from_legacy(action),
        event_delta=(),
    )


def _trajectory_episode(episode: TrainingEpisode) -> TrajectoryEpisode:
    metadata = {
        "episode_id": episode.episode_id,
        "replay_tag": "firstlight-sequence-cache-smoke",
        "outcome": episode.outcome,
        "split": episode.split,
        "episode_context_ref": "c" * 64,
    }
    records = tuple(
        DecisionStep(
            identity=step.identity,
            episode_context_ref=metadata["episode_context_ref"],
            observation=step.observation,
            action_mask=step.action_mask,
            expert_action=step.expert_action,
            event_delta=step.event_delta,
            validation={"accepted": True},
            fingerprints={"semantic_frame_schema": "synthetic-smoke"},
        )
        for step in episode.steps
    )
    record_hashes = tuple(item.content_sha256() for item in records)
    manifest = TrajectoryManifest(
        episode_id=episode.episode_id,
        replay_tag=metadata["replay_tag"],
        outcome=episode.outcome,
        first_tick=records[0].identity.tick,
        last_tick=records[-1].identity.tick,
        record_count=len(records),
        record_hashes=record_hashes,
        aggregate_hash=aggregate_record_hash(episode.episode_id, record_hashes),
        metadata_sha256=content_sha256(metadata),
        split=episode.split,
    )
    return TrajectoryEpisode(manifest, records, metadata)


def _tensor_fields(value, *, prefix=""):
    result = {}
    for item in fields(value):
        child = getattr(value, item.name)
        key = f"{prefix}.{item.name}" if prefix else item.name
        if torch.is_tensor(child):
            result[key] = child
        elif is_dataclass(child):
            result.update(_tensor_fields(child, prefix=key))
        else:
            raise AssertionError(f"non-tensor V4 field in smoke record: {key}")
    return result


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
    episode_id_bytes = compiled.episode_id.encode("utf-8")
    prefix_digest = hashlib.sha256(
        bytes.fromhex(hashlib.sha256(b"").hexdigest())
        + len(episode_id_bytes).to_bytes(8, "big")
        + episode_id_bytes
    ).hexdigest()
    progress = TrainingRunProgress(
        dataset_fingerprint="synthetic-cache-fingerprint",
        completed_episode_count=1,
        completed_episode_prefix_sha256=prefix_digest,
        optimizer_steps=recurrent_result.optimizer_steps,
        data_split_fingerprint="d" * 64,
    )
    saved = trainer.save_checkpoint(
        checkpoint_path,
        optimizer=optimizer,
        update_step=recurrent_result.optimizer_steps,
        progress=progress,
    )
    with torch.no_grad():
        next(backend.model.parameters()).add_(1.0)
    optimizer.state.clear()

    restored = trainer.restore_checkpoint(
        checkpoint_path,
        optimizer=optimizer,
        expected_dataset_fingerprint="synthetic-cache-fingerprint",
        expected_data_split_fingerprint="d" * 64,
    )

    actual_parameters = torch.nn.utils.parameters_to_vector(backend.model.parameters()).detach()
    assert torch.equal(actual_parameters, expected_parameters)
    assert len(optimizer.state) == optimizer_state_count
    assert saved.checkpoint_id == restored.checkpoint_id
    assert restored.update_step == recurrent_result.optimizer_steps
    assert restored.progress == progress


@pytest.mark.firstlight
@pytest.mark.skipif(
    os.environ.get("ELIXIR_ECHO_RUN_TRAINING_SMOKE") != "1",
    reason="set ELIXIR_ECHO_RUN_TRAINING_SMOKE=1 to opt in to FirstLight sequence cache checks",
)
def test_firstlight_compiled_sequence_layer2_cache_roundtrip(tmp_path):
    root = os.environ.get("ELIXIR_ECHO_FIRSTLIGHT_ROOT")
    if not root:
        pytest.skip("set ELIXIR_ECHO_FIRSTLIGHT_ROOT to run the FirstLight sequence-cache smoke")
    runtime = FirstLightRuntime(root)
    backend = FirstLightPolicyBackend(runtime, device="cpu")
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
        episode_id="layer2-smoke-episode",
        outcome="terminated",
        split="train",
        steps=(
            _step(90, 0, play, episode_id="layer2-smoke-episode"),
            _step(90, 1, None, episode_id="layer2-smoke-episode"),
            _step(95, 0, None, episode_id="layer2-smoke-episode"),
            _step(95, 1, None, episode_id="layer2-smoke-episode"),
        ),
    )
    input_builder = PolicyInputBuilder(runtime, components=backend.components)
    with pytest.raises(ValueError, match="input builder mode"):
        compile_firstlight_actor_sequence(
            episode,
            actor=0,
            input_builder=input_builder,
            components=backend.components,
            config=replace(config, input_mode="legacy_checkpoint"),
        )
    compiled_sequences = []
    encoded_by_actor = {}

    def compile_actor(training_episode, actor):
        sequence = compile_firstlight_actor_sequence(
            training_episode,
            actor=actor,
            input_builder=input_builder,
            components=backend.components,
            config=config,
        )
        compiled_sequences.append(sequence)
        return sequence

    def encode_sequence(sequence):
        payload = encode_actor_sequence(sequence, training_config=config)
        encoded_by_actor[sequence.actor] = payload
        return payload

    fingerprint = "a" * 64
    input_fingerprint = hashlib.sha256(b"synthetic composer and V4 contract").hexdigest()
    cache = Layer2SequenceCache(tmp_path / "layer2")
    source_episode = _trajectory_episode(episode)
    compiler = SequenceDatasetCompiler(
        cache=cache,
        dataset_fingerprint=fingerprint,
        training_config_fingerprint=config.fingerprint(),
        input_contract_fingerprint=input_fingerprint,
        profile=ObservationProfile.RICH,
        actor_sequence_compiler=compile_actor,
        actor_sequence_encoder=encode_sequence,
    )
    manifest = compiler.compile_episode(source_episode)
    sequences = tuple(compiled_sequences)
    encoded = tuple(encoded_by_actor[actor] for actor in (0, 1))
    assert manifest["source_episode_sha256"] == source_episode.manifest.content_sha256()
    incompatible_mode = replace(config, input_mode="legacy_checkpoint")
    with pytest.raises(FirstLightSequenceCacheError, match="input mode"):
        encode_actor_sequence(sequences[0], training_config=incompatible_mode)
    with pytest.raises(FirstLightSequenceCacheError, match="input mode"):
        decode_actor_sequence(
            encoded[0],
            runtime,
            components=backend.components,
            training_config=incompatible_mode,
        )
    decoded = tuple(
        decode_actor_sequence(
            payload,
            runtime,
            components=backend.components,
            training_config=config,
        )
        for payload in encoded
    )

    for original, restored in zip(sequences, decoded, strict=True):
        assert restored.identities == original.identities
        assert restored.profile == original.profile
        assert restored.label_counts == original.label_counts
        for original_input, restored_input in zip(original.inputs, restored.inputs, strict=True):
            assert original_input.action_mask is not restored_input.action_mask
            assert restored_input.action_mask is None
            assert restored_input.temporal_context is None
            original_tensors = _tensor_fields(original_input.tensors.batch)
            restored_tensors = _tensor_fields(restored_input.tensors.batch)
            assert original_tensors.keys() == restored_tensors.keys()
            assert all(torch.equal(original_tensors[key], restored_tensors[key]) for key in original_tensors)
        for original_label, restored_label in zip(original.labels, restored.labels, strict=True):
            assert original_label.category == restored_label.category
            assert original_label.supervision == restored_label.supervision
            original_tensors = _tensor_fields(original_label.action)
            restored_tensors = _tensor_fields(restored_label.action)
            assert original_tensors.keys() == restored_tensors.keys()
            assert all(torch.equal(original_tensors[key], restored_tensors[key]) for key in original_tensors)

    forged_label = replace(
        sequences[0].labels[0],
        category=LabelCategory.SINGLE_ABILITY,
        supervision=HeadSupervision(gate=True),
        trainable=True,
    )
    forged_sequence = replace(
        sequences[0],
        labels=(forged_label, *sequences[0].labels[1:]),
    )
    with pytest.raises(FirstLightSequenceCacheError, match="approved policy"):
        encode_actor_sequence(forged_sequence, training_config=config)

    magic, metadata_length = _HEADER.unpack_from(encoded[0])
    assert magic == _MAGIC
    metadata_start = _HEADER.size
    metadata_end = metadata_start + metadata_length
    cached_metadata = json.loads(encoded[0][metadata_start:metadata_end])
    cached_metadata["frames"][0]["category"] = LabelCategory.SINGLE_ABILITY.value
    cached_metadata["frames"][0]["trainable"] = True
    cached_metadata["frames"][0]["supervision"] = {
        "gate": True,
        "candidate": False,
        "target": False,
        "continue_action": False,
        "delay": False,
        "value": False,
    }
    forged_metadata_bytes = canonical_json(cached_metadata).encode("utf-8")
    forged_payload = (
        _HEADER.pack(_MAGIC, len(forged_metadata_bytes))
        + forged_metadata_bytes
        + encoded[0][metadata_end:]
    )
    with pytest.raises(FirstLightSequenceCacheError, match="approved policy"):
        decode_actor_sequence(
            forged_payload,
            runtime,
            components=backend.components,
            training_config=config,
        )

    dataset = SequenceDataset.from_layer2_cache(
        cache,
        sequence_decoder=lambda item: tuple(
            decode_actor_sequence(
                payload,
                runtime,
                components=backend.components,
                training_config=config,
            )
            for payload in item.actor_payloads
        ),
        selected_episode_ids=(episode.episode_id,),
        episode_splits={episode.episode_id: "train"},
        dataset_fingerprint=fingerprint,
        training_config_fingerprint=config.fingerprint(),
        input_contract_fingerprint=input_fingerprint,
    )
    loaded = tuple(dataset)
    assert [(item.episode_id, item.actor) for item in loaded] == [
        (episode.episode_id, 0),
        (episode.episode_id, 1),
    ]
