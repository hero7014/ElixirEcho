from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import importlib
import os
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

from elixir_echo.datasets.legacy import LegacyTrajectoryReader
from elixir_echo.datasets.sequence import Layer2SequenceCache, SequenceDatasetCompiler
from elixir_echo.integrations.firstlight.il import compile_firstlight_actor_sequence
from elixir_echo.integrations.firstlight.runtime import FirstLightRuntime
from elixir_echo.integrations.firstlight.sequence_cache import encode_actor_sequence
from elixir_echo.integrations.firstlight.tensorizer import PolicyInputBuilder, load_v4_components
from elixir_echo.profiles import ObservationProfile
from elixir_echo.training.config import TrainingConfig


def _tensor_fields(value, prefix: str = ""):
    if torch.is_tensor(value):
        return {prefix: value}
    assert is_dataclass(value), f"expected tensor record at {prefix}"
    result = {}
    for item in fields(value):
        child_prefix = f"{prefix}.{item.name}" if prefix else item.name
        result.update(_tensor_fields(getattr(value, item.name), child_prefix))
    return result


def _tensor_record_hash(value) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(_tensor_fields(value).items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _assert_tensor_records_equal(expected, actual, *, context: str) -> None:
    expected_fields = _tensor_fields(expected)
    actual_fields = _tensor_fields(actual)
    assert expected_fields.keys() == actual_fields.keys(), f"{context}: tensor field names differ"
    for name in expected_fields:
        assert expected_fields[name].dtype == actual_fields[name].dtype, f"{context}.{name}: dtype differs"
        if not torch.equal(expected_fields[name], actual_fields[name]):
            expected_tensor = expected_fields[name]
            actual_tensor = actual_fields[name]
            differing = torch.nonzero(expected_tensor != actual_tensor, as_tuple=False)
            first = tuple(int(item) for item in differing[0].tolist()) if differing.numel() else ()
            expected_value = expected_tensor[first].item() if first else expected_tensor.item()
            actual_value = actual_tensor[first].item() if first else actual_tensor.item()
            raise AssertionError(
                f"{context}.{name}: values differ at {first}: "
                f"sealed={expected_value!r}, current={actual_value!r}; "
                f"differing_elements={int(differing.shape[0])}"
            )
    assert _tensor_record_hash(expected) == _tensor_record_hash(actual), f"{context}: tensor hash differs"


def test_layer2_compiler_matches_existing_sealed_firstlight_sequence_cache(tmp_path) -> None:
    baseline_root = os.environ.get("ELIXIR_ECHO_BASELINE_ROOT")
    episode_path = os.environ.get("ELIXIR_ECHO_LEGACY_EPISODE")
    provider_root = os.environ.get("ELIXIR_ECHO_FIRSTLIGHT_ROOT")
    legacy_summary_path = os.environ.get("ELIXIR_ECHO_LEGACY_LAYER2_SUMMARY")
    if not all((baseline_root, episode_path, provider_root, legacy_summary_path)):
        pytest.skip(
            "set baseline, selected full episode, FirstLight root and sealed Layer 2 summary for real sequence parity"
        )

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    baseline_src = Path(baseline_root).resolve() / "src"
    sys.path.insert(0, str(baseline_src))
    try:
        source_episode = LegacyTrajectoryReader(episode_path).read_episode()
        assert source_episode.manifest.outcome == "terminated"
        assert source_episode.manifest.split == "train"

        runtime = FirstLightRuntime(provider_root)
        components = load_v4_components(runtime)
        # The sealed recurrent cache was produced through the legacy checkpoint
        # tensor path. Keep this parity assertion isolated from the separately
        # tested richer RICH tensor policy.
        input_builder = PolicyInputBuilder(
            runtime,
            components=components,
            legacy_checkpoint_input=True,
        )
        config = TrainingConfig(
            profile=ObservationProfile.RICH,
            input_mode="legacy_checkpoint",
        )
        compiled = []
        encoded_by_actor = {}

        def compile_actor(training_episode, actor):
            sequence = compile_firstlight_actor_sequence(
                training_episode,
                actor=actor,
                input_builder=input_builder,
                components=components,
                config=config,
            )
            compiled.append(sequence)
            return sequence

        def encode(sequence):
            payload = encode_actor_sequence(sequence, training_config=config)
            encoded_by_actor[sequence.actor] = payload
            return payload

        cache = Layer2SequenceCache(tmp_path / "layer2")
        compiler = SequenceDatasetCompiler(
            cache=cache,
            dataset_fingerprint=hashlib.sha256(b"sealed-layer1-sequence-parity").hexdigest(),
            training_config_fingerprint=config.fingerprint(),
            input_contract_fingerprint=hashlib.sha256(b"FirstLight-V4-RICH-TemporalV4").hexdigest(),
            profile=ObservationProfile.RICH,
            actor_sequence_compiler=compile_actor,
            actor_sequence_encoder=encode,
        )
        compiler.compile_episode(source_episode)
        cached = cache.read_episode(source_episode.manifest.episode_id)
        assert cached.ticks_by_actor[0] == cached.ticks_by_actor[1]
        assert cached.label_counts_by_actor[0] == compiled[0].label_counts
        assert cached.label_counts_by_actor[1] == compiled[1].label_counts

        old_runtime_module = importlib.import_module("training.firstlight_v4")
        old_cache_module = importlib.import_module("training.recurrent_cache")
        old_runtime = old_runtime_module.load_firstlight_v4(provider_root)
        old_cache = old_cache_module.load_recurrent_cache_shard(
            old_runtime,
            legacy_summary_path,
        )
        old_indices = {
            (descriptor.episode_id, descriptor.actor): index
            for index, descriptor in enumerate(old_cache.descriptors)
        }
        for actor in (0, 1):
            key = (source_episode.manifest.episode_id, actor)
            assert key in old_indices, f"sealed cache lacks actor sequence {key}"
            old_sequence = old_cache.sequence(old_indices[key])
            new_sequence = compiled[actor]
            descriptor = old_sequence.descriptor
            assert descriptor.split == source_episode.manifest.split
            assert descriptor.outcome == source_episode.manifest.outcome
            assert descriptor.ticks == tuple(identity.tick for identity in new_sequence.identities)
            assert len(old_sequence.sequence.observations) == len(new_sequence.inputs)
            assert len(old_sequence.sequence.actions) == len(new_sequence.labels)
            assert old_sequence.supervision.label_categories == tuple(
                label.category.value for label in new_sequence.labels
            )
            assert tuple(old_sequence.supervision.mask_row(i) for i in range(len(new_sequence.labels))) == tuple(
                tuple(label.supervision.to_dict().values()) for label in new_sequence.labels
            )
            for frame, (old_observation, old_action) in enumerate(
                zip(old_sequence.sequence.observations, old_sequence.sequence.actions, strict=True)
            ):
                _assert_tensor_records_equal(
                    old_observation,
                    new_sequence.inputs[frame].tensors.batch,
                    context=f"{key} tick {new_sequence.identities[frame].tick} Observation",
                )
                _assert_tensor_records_equal(
                    old_action,
                    new_sequence.labels[frame].action,
                    context=f"{key} tick {new_sequence.identities[frame].tick} expert action",
                )
            assert cached.actor_payloads[actor] == encoded_by_actor[actor]
    finally:
        try:
            sys.path.remove(str(baseline_src))
        except ValueError:
            pass
