from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from elixir_echo.datasets.sequence import (
    Layer2SequenceCache,
    SequenceDatasetCompiler,
    SequenceBatch,
    SequenceCacheError,
    SequenceDataset,
)
from elixir_echo.datasets.trajectory import (
    TrajectoryEpisode,
    TrajectoryManifest,
    aggregate_record_hash,
)
from elixir_echo.domain import ActionMask, DecisionIdentity, DecisionStep, ExpertAction, Observation
from elixir_echo.domain.values import content_sha256
from elixir_echo.profiles import ObservationProfile
from elixir_echo.training.labels import LabelCategory


def _sequence(episode_id: str, actor: int, length: int):
    return SimpleNamespace(
        episode_id=episode_id,
        actor=actor,
        profile="full",
        identities=tuple(
            DecisionIdentity(episode_id, 90 + 5 * index, actor)
            for index in range(length)
        ),
        inputs=tuple(f"{episode_id}:{actor}:input:{index}" for index in range(length)),
        labels=tuple(f"{episode_id}:{actor}:label:{index}" for index in range(length)),
    )


def _dataset(sequences, *, ids=("episode-a", "episode-b"), splits=None):
    source = tuple(sequences)
    return SequenceDataset(
        lambda: iter(source),
        selected_episode_ids=ids,
        episode_splits=splits or {episode_id: "train" for episode_id in ids},
        dataset_fingerprint="a" * 64,
    )


def test_sequence_dataset_is_deterministic_and_batches_only_complete_episode_groups():
    sequences = tuple(
        item
        for episode_id, lengths in (("episode-a", (2, 3)), ("episode-b", (1, 2)))
        for actor, length in enumerate(lengths)
        for item in (_sequence(episode_id, actor, length),)
    )
    dataset = _dataset(sequences)

    first = tuple(dataset)
    second = tuple(dataset)
    assert [(item.episode_id, item.actor) for item in first] == [
        ("episode-a", 0),
        ("episode-a", 1),
        ("episode-b", 0),
        ("episode-b", 1),
    ]
    assert [(item.episode_id, item.actor) for item in second] == [
        (item.episode_id, item.actor) for item in first
    ]

    batches = tuple(dataset.iter_batches(max_episodes=1))
    assert [batch.episode_ids for batch in batches] == [("episode-a",), ("episode-b",)]
    assert [[lane.actor for lane in batch.lanes] for batch in batches] == [[0, 1], [0, 1]]
    first_batch = batches[0]
    assert first_batch.valid_mask == ((True, True), (True, True), (False, True))
    assert first_batch.episode_start == ((True, True), (False, False), (False, False))
    assert first_batch.inputs_by_time[-1] == (None, "episode-a:1:input:2")
    assert first_batch.labels_by_time[-1] == (None, "episode-a:1:label:2")


def test_sequence_dataset_resume_starts_after_an_episode_boundary():
    sequences = tuple(_sequence(episode_id, actor, 1) for episode_id in ("episode-a", "episode-b") for actor in (0, 1))
    dataset = _dataset(sequences)

    assert [
        tuple((item.episode_id, item.actor) for item in group)
        for group in dataset.iter_episode_groups(start_after_episode_id="episode-a")
    ] == [(('episode-b', 0), ('episode-b', 1))]


@pytest.mark.parametrize(
    "sequences",
    [
        (_sequence("episode-a", 1, 1), _sequence("episode-a", 0, 1)),
        (_sequence("episode-a", 0, 1),),
        (_sequence("episode-a", 0, 1), _sequence("episode-a", 0, 1)),
        (
            _sequence("episode-a", 0, 1),
            _sequence("episode-a", 1, 1),
            _sequence("episode-extra", 0, 1),
            _sequence("episode-extra", 1, 1),
        ),
    ],
)
def test_sequence_dataset_rejects_missing_duplicate_or_out_of_order_actor_lanes(sequences):
    dataset = _dataset(sequences, ids=("episode-a",))

    with pytest.raises(ValueError):
        tuple(dataset)


def test_sequence_dataset_rejects_non_train_episode_and_invalid_fingerprint():
    sequences = (_sequence("episode-a", 0, 1), _sequence("episode-a", 1, 1))

    with pytest.raises(ValueError, match="train split"):
        _dataset(sequences, ids=("episode-a",), splits={"episode-a": "validation"})
    with pytest.raises(ValueError, match="SHA-256"):
        SequenceDataset(
            lambda: iter(sequences),
            selected_episode_ids=("episode-a",),
            episode_splits={"episode-a": "train"},
            dataset_fingerprint="not-a-hash",
        )


def test_sequence_batch_direct_construction_cannot_bypass_lane_invariants():
    lanes = (_sequence("episode-a", 0, 1), _sequence("episode-a", 1, 1))
    batch = SequenceBatch.from_episode_groups((lanes,))

    with pytest.raises(ValueError, match="actor 0 then actor 1"):
        replace(batch, lanes=(lanes[0], lanes[0]))


@pytest.mark.parametrize("present_field", ["input", "label"])
def test_sequence_batch_rejects_one_sided_padded_values(present_field):
    lanes = (_sequence("episode-a", 0, 1), _sequence("episode-a", 1, 2))
    batch = SequenceBatch.from_episode_groups((lanes,))
    padded_inputs = list(batch.inputs_by_time[-1])
    padded_labels = list(batch.labels_by_time[-1])
    if present_field == "input":
        padded_inputs[0] = "orphan-input"
    else:
        padded_labels[0] = "orphan-label"

    with pytest.raises(ValueError, match="padding mask is inconsistent"):
        if present_field == "input":
            replace(batch, inputs_by_time=(*batch.inputs_by_time[:-1], tuple(padded_inputs)))
        else:
            replace(batch, labels_by_time=(*batch.labels_by_time[:-1], tuple(padded_labels)))


def _cache(cache_root):
    return Layer2SequenceCache(cache_root)


def _commit_episode(cache, episode_id="episode-a", *, split="train"):
    return cache.commit_episode(
        episode_id=episode_id,
        split=split,
        outcome="terminated",
        source_episode_sha256="1" * 64,
        dataset_fingerprint="a" * 64,
        training_config_fingerprint="b" * 64,
        input_contract_fingerprint="c" * 64,
        profile="full",
        actor_payloads={0: b"actor-zero", 1: b"actor-one"},
        ticks_by_actor={0: (90, 95), 1: (90, 95)},
        label_counts_by_actor={
            0: {"wait": 1, "single_play_card": 1},
            1: {"wait": 2},
        },
    )


def test_layer2_cache_commits_and_verifies_complete_actor_pair_atomically(tmp_path):
    cache = _cache(tmp_path / "layer2")
    manifest = _commit_episode(cache)

    loaded = cache.read_episode(
        "episode-a",
        expected_dataset_fingerprint="a" * 64,
        expected_training_config_fingerprint="b" * 64,
        expected_input_contract_fingerprint="c" * 64,
    )
    assert manifest["actors"]["0"]["payload_sha256"]
    assert loaded.episode_id == "episode-a"
    assert loaded.split == "train"
    assert loaded.actor_payloads == (b"actor-zero", b"actor-one")
    assert loaded.ticks_by_actor == ((90, 95), (90, 95))
    assert cache.committed_episode_ids(split="train") == ("episode-a",)


@pytest.mark.parametrize(
    ("lane_update", "message"),
    (
        ({"profile": "core"}, "profile"),
        (
            {
                "identities": (
                    DecisionIdentity("episode-a", 90, 0),
                    DecisionIdentity("episode-a", 100, 0),
                )
            },
            "ticks",
        ),
    ),
)
def test_layer2_sequence_dataset_rejects_decoded_lane_contract_drift(
    tmp_path, lane_update, message
):
    cache = _cache(tmp_path / "layer2")
    _commit_episode(cache, "episode-a")

    def decoder(episode):
        lane0 = _sequence(episode.episode_id, 0, 2)
        lane1 = _sequence(episode.episode_id, 1, 2)
        if message == "profile":
            lane0 = SimpleNamespace(**{**vars(lane0), **lane_update})
            lane1 = SimpleNamespace(**{**vars(lane1), **lane_update})
        else:
            lane0 = SimpleNamespace(**{**vars(lane0), **lane_update})
            lane1 = SimpleNamespace(
                **{
                    **vars(lane1),
                    "identities": tuple(
                        DecisionIdentity("episode-a", 90 + 10 * index, 1)
                        for index in range(2)
                    ),
                }
            )
        return lane0, lane1

    dataset = SequenceDataset.from_layer2_cache(
        cache,
        sequence_decoder=decoder,
        selected_episode_ids=("episode-a",),
        episode_splits={"episode-a": "train"},
        dataset_fingerprint="a" * 64,
        training_config_fingerprint="b" * 64,
        input_contract_fingerprint="c" * 64,
    )

    with pytest.raises(ValueError, match=message):
        tuple(dataset.iter_episode_groups())


def test_layer2_cache_rejects_duplicate_identity_bad_tick_alignment_and_fingerprint(tmp_path):
    cache = _cache(tmp_path / "layer2")
    _commit_episode(cache)
    with pytest.raises(FileExistsError):
        _commit_episode(cache)
    with pytest.raises(SequenceCacheError, match="fingerprint"):
        cache.read_episode("episode-a", expected_dataset_fingerprint="d" * 64)

    with pytest.raises(ValueError, match="same decision ticks"):
        cache.commit_episode(
            episode_id="episode-b",
            split="train",
            outcome="terminated",
            source_episode_sha256="1" * 64,
            dataset_fingerprint="a" * 64,
            training_config_fingerprint="b" * 64,
            input_contract_fingerprint="c" * 64,
            profile="full",
            actor_payloads={0: b"a", 1: b"b"},
            ticks_by_actor={0: (90, 95), 1: (90, 95, 100)},
            label_counts_by_actor={0: {"wait": 2}, 1: {"wait": 3}},
        )


def test_layer2_cache_hides_uncommitted_episode_and_cleans_failed_staging(tmp_path, monkeypatch):
    cache = _cache(tmp_path / "layer2")
    import elixir_echo.datasets.sequence as sequence_module

    def failed_replace(_source, _destination):
        raise OSError("simulated atomic publish failure")

    monkeypatch.setattr(sequence_module.os, "replace", failed_replace)
    with pytest.raises(OSError, match="simulated"):
        _commit_episode(cache)
    episodes_dir = tmp_path / "layer2" / "episodes"
    assert cache.committed_episode_ids() == ()
    assert list(episodes_dir.iterdir()) == []


def test_layer2_cache_rejects_payload_tampering_and_rebuilds_sequence_dataset(tmp_path):
    cache = _cache(tmp_path / "layer2")
    _commit_episode(cache, "episode-a")
    _commit_episode(cache, "episode-b")

    decoded_episode_ids = []

    def decoder(episode):
        decoded_episode_ids.append(episode.episode_id)
        length = len(episode.ticks_by_actor[0])
        return (_sequence(episode.episode_id, 0, length), _sequence(episode.episode_id, 1, length))

    dataset = SequenceDataset.from_layer2_cache(
        cache,
        sequence_decoder=decoder,
        selected_episode_ids=("episode-a", "episode-b"),
        episode_splits={"episode-a": "train", "episode-b": "train"},
        dataset_fingerprint="a" * 64,
        training_config_fingerprint="b" * 64,
        input_contract_fingerprint="c" * 64,
    )
    assert [item.episode_id for item in dataset] == [
        "episode-a", "episode-a", "episode-b", "episode-b"
    ]
    decoded_episode_ids.clear()
    assert [
        (group[0].episode_id, group[1].episode_id)
        for group in dataset.iter_episode_groups(start_after_episode_id="episode-a")
    ] == [("episode-b", "episode-b")]
    assert decoded_episode_ids == ["episode-b"]

    payload_file = tmp_path / "layer2" / "episodes" / "episode-b" / "actor-1.bin"
    payload_file.write_bytes(b"changed!!")
    with pytest.raises(SequenceCacheError, match="SHA-256"):
        cache.read_episode("episode-b")


def _trajectory_episode(episode_id="episode-compile"):
    metadata = {
        "episode_id": episode_id,
        "replay_tag": "fixed-test-fixture",
        "outcome": "terminated",
        "split": "train",
        "episode_context_ref": "d" * 64,
    }
    records = tuple(
        DecisionStep(
            identity=DecisionIdentity(episode_id, tick, actor),
            episode_context_ref=metadata["episode_context_ref"],
            observation=Observation({}, (), (), (), {}),
            action_mask=ActionMask(DecisionIdentity(episode_id, tick, actor), {}),
            expert_action=ExpertAction.wait(),
            event_delta=(),
            validation={"accepted": True},
            fingerprints={"semantic_frame_schema": "test"},
        )
        for tick in (90, 95)
        for actor in (0, 1)
    )
    hashes = tuple(record.content_sha256() for record in records)
    manifest = TrajectoryManifest(
        episode_id=episode_id,
        replay_tag=metadata["replay_tag"],
        outcome=metadata["outcome"],
        first_tick=90,
        last_tick=95,
        record_count=len(records),
        record_hashes=hashes,
        aggregate_hash=aggregate_record_hash(episode_id, hashes),
        metadata_sha256=content_sha256(metadata),
        split="train",
    )
    return TrajectoryEpisode(manifest, records, metadata)


def test_sequence_dataset_compiler_builds_and_commits_both_actor_lanes_from_verified_episode(tmp_path):
    episode = _trajectory_episode()
    observed_training_inputs = []

    def compile_actor(training_episode, actor):
        observed_training_inputs.append(training_episode)
        identities = tuple(
            step.identity for step in training_episode.steps if step.identity.actor == actor
        )
        return SimpleNamespace(
            episode_id=training_episode.episode_id,
            actor=actor,
            profile=ObservationProfile.RICH,
            identities=identities,
            inputs=tuple(SimpleNamespace(identity=item, profile="full") for item in identities),
            labels=tuple(SimpleNamespace(category=LabelCategory.WAIT) for _ in identities),
            label_counts={"wait": len(identities)},
        )

    compiler = SequenceDatasetCompiler(
        cache=Layer2SequenceCache(tmp_path / "layer2"),
        dataset_fingerprint="a" * 64,
        training_config_fingerprint="b" * 64,
        input_contract_fingerprint="c" * 64,
        profile=ObservationProfile.RICH,
        actor_sequence_compiler=compile_actor,
        actor_sequence_encoder=lambda sequence: f"actor-{sequence.actor}".encode(),
    )

    manifest = compiler.compile_episode(episode)

    assert manifest["episode_id"] == episode.manifest.episode_id
    assert manifest["source_episode_sha256"] == episode.manifest.content_sha256()
    assert manifest["split"] == "train"
    assert manifest["profile"] == "full"
    assert manifest["actors"]["0"]["ticks"] == [90, 95]
    assert manifest["actors"]["1"]["ticks"] == [90, 95]
    assert manifest["actors"]["0"]["label_counts"] == {"wait": 2}
    cached = Layer2SequenceCache(tmp_path / "layer2").read_episode(episode.manifest.episode_id)
    assert cached.actor_payloads == (b"actor-0", b"actor-1")
    assert len(observed_training_inputs) == 2
    assert all(not hasattr(item, "episode_context") for item in observed_training_inputs)


def test_sequence_dataset_compiler_rejects_corrupt_source_or_nonterminated_episode(tmp_path):
    episode = _trajectory_episode()
    bad_episode = replace(episode, records=episode.records[:-1])

    compiler = SequenceDatasetCompiler(
        cache=Layer2SequenceCache(tmp_path / "layer2"),
        dataset_fingerprint="a" * 64,
        training_config_fingerprint="b" * 64,
        input_contract_fingerprint="c" * 64,
        profile=ObservationProfile.RICH,
        actor_sequence_compiler=lambda *_: pytest.fail("must reject source before compile"),
        actor_sequence_encoder=lambda _: b"unused",
    )
    with pytest.raises(ValueError, match="source record count"):
        compiler.compile_episode(bad_episode)

    truncated = replace(
        episode,
        manifest=replace(episode.manifest, outcome="truncated"),
    )
    with pytest.raises(ValueError, match="terminated episodes only"):
        compiler.compile_episode(truncated)
