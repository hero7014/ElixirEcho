"""Safe serialization for compiled FirstLight V4 actor sequences.

Only model-ready V4 tensor records and the already-compiled supervision target
are encoded. Raw semantic frames, EpisodeContext, temporal history, and source
replay records are intentionally outside this Layer 2 payload.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
import struct
from typing import Any, Mapping, get_type_hints

from elixir_echo.domain import DecisionIdentity
from elixir_echo.domain.values import canonical_json
from elixir_echo.policies.protocol import PolicyInput
from elixir_echo.profiles import ObservationProfile
from elixir_echo.training.config import HeadSupervision, TrainingConfig
from elixir_echo.training.labels import (
    CompiledLabel,
    LabelCategory,
    validate_label_supervision,
)
from elixir_echo.training.sequence import CompiledActorSequence


_SCHEMA = "elixir_echo.firstlight-v4-actor-sequence@1"
_MAGIC = b"EEFV4SQ1"
_HEADER = struct.Struct("<8sQ")


class FirstLightSequenceCacheError(ValueError):
    """Raised when a compiled V4 actor sequence is not safely representable."""


def _flatten_tensor_record(
    value: Any,
    *,
    prefix: str,
    tensors: dict[str, Any],
) -> None:
    import torch

    if not is_dataclass(value) or isinstance(value, type):
        raise FirstLightSequenceCacheError("V4 model inputs and action labels must be tensor records")
    for item in fields(value):
        child = getattr(value, item.name)
        key = f"{prefix}.{item.name}"
        if torch.is_tensor(child):
            if key in tensors:
                raise FirstLightSequenceCacheError(f"duplicate V4 tensor key {key!r}")
            tensors[key] = child.detach().to(device="cpu").contiguous()
        elif is_dataclass(child) and not isinstance(child, type):
            _flatten_tensor_record(child, prefix=key, tensors=tensors)
        else:
            raise FirstLightSequenceCacheError(f"V4 tensor record field {key!r} is not a tensor")


def _restore_tensor_record(
    record_type: type[Any],
    *,
    prefix: str,
    tensors: Mapping[str, Any],
    consumed: set[str],
) -> Any:
    if not isinstance(record_type, type) or not is_dataclass(record_type):
        raise FirstLightSequenceCacheError("V4 cache requested a non-dataclass record type")
    try:
        hints = get_type_hints(record_type)
    except (NameError, TypeError) as error:
        raise FirstLightSequenceCacheError(f"cannot resolve V4 tensor record {record_type.__name__}") from error
    values: dict[str, Any] = {}
    for item in fields(record_type):
        key = f"{prefix}.{item.name}"
        annotation = hints.get(item.name)
        if isinstance(annotation, type) and is_dataclass(annotation):
            values[item.name] = _restore_tensor_record(
                annotation,
                prefix=key,
                tensors=tensors,
                consumed=consumed,
            )
            continue
        try:
            values[item.name] = tensors[key]
        except KeyError as error:
            raise FirstLightSequenceCacheError(f"V4 sequence cache is missing tensor {key!r}") from error
        consumed.add(key)
    return record_type(**values)


def encode_actor_sequence(
    sequence: CompiledActorSequence,
    *,
    training_config: TrainingConfig | None = None,
) -> bytes:
    """Encode one actor lane with safetensors and strict JSON metadata."""

    try:
        import zstandard
        from safetensors.torch import save as save_safetensors
    except ImportError as error:
        raise RuntimeError("FirstLight sequence cache requires safetensors and zstandard") from error
    if not isinstance(sequence, CompiledActorSequence):
        raise TypeError("sequence must be a CompiledActorSequence")
    config = training_config or TrainingConfig(profile=sequence.profile)
    if config.profile is not sequence.profile:
        raise FirstLightSequenceCacheError("sequence profile differs from the pinned TrainingConfig")
    tensor_payload: dict[str, Any] = {}
    frame_metadata: list[dict[str, Any]] = []
    input_mode: str | None = None
    for index, (identity, policy_input, label) in enumerate(
        zip(sequence.identities, sequence.inputs, sequence.labels, strict=True)
    ):
        if policy_input.identity != identity or policy_input.profile != sequence.profile.value:
            raise FirstLightSequenceCacheError("compiled input identity/profile differs from sequence")
        tensorized = policy_input.tensors
        batch = getattr(tensorized, "batch", None)
        mode = getattr(tensorized, "input_mode", None)
        if batch is None or mode not in {"rich", "legacy_checkpoint"}:
            raise FirstLightSequenceCacheError("compiled input lacks a supported FirstLight V4 tensor record")
        if mode != config.input_mode:
            raise FirstLightSequenceCacheError("compiled input mode differs from the pinned TrainingConfig")
        if input_mode is None:
            input_mode = mode
        elif input_mode != mode:
            raise FirstLightSequenceCacheError("actor sequence mixes FirstLight input modes")
        action = getattr(label, "action", None)
        try:
            validate_label_supervision(
                label.category,
                label.supervision,
                label.trainable,
                config,
            )
        except (TypeError, ValueError) as error:
            raise FirstLightSequenceCacheError(f"compiled label violates TrainingConfig: {error}") from error
        _flatten_tensor_record(batch, prefix=f"frame.{index}.observation", tensors=tensor_payload)
        _flatten_tensor_record(action, prefix=f"frame.{index}.action", tensors=tensor_payload)
        supervision = label.supervision.to_dict()
        if set(supervision) != {
            "gate", "candidate", "target", "continue_action", "delay", "value"
        } or any(type(value) is not bool for value in supervision.values()):
            raise FirstLightSequenceCacheError("compiled label supervision mask is invalid")
        frame_metadata.append(
            {
                "identity": identity.to_dict(),
                "category": label.category.value,
                "trainable": label.trainable,
                "report_reason": label.report_reason,
                "supervision": supervision,
            }
        )
    if not frame_metadata or input_mode is None:
        raise FirstLightSequenceCacheError("cannot cache an empty actor sequence")
    safetensors_bytes = save_safetensors(tensor_payload)
    compressed_tensors = zstandard.ZstdCompressor(level=1, threads=0).compress(safetensors_bytes)
    metadata = {
        "schema": _SCHEMA,
        "episode_id": sequence.episode_id,
        "actor": sequence.actor,
        "profile": sequence.profile.value,
        "input_mode": input_mode,
        "tensor_codec": "safetensors+zstd",
        "tensor_bytes": len(safetensors_bytes),
        "frames": frame_metadata,
        "tensor_keys": sorted(tensor_payload),
    }
    metadata_bytes = canonical_json(metadata).encode("utf-8")
    return _HEADER.pack(_MAGIC, len(metadata_bytes)) + metadata_bytes + compressed_tensors


def decode_actor_sequence(
    payload: bytes,
    runtime: Any,
    *,
    components: Any | None = None,
    training_config: TrainingConfig | None = None,
) -> CompiledActorSequence:
    """Verify and restore one actor sequence using only the configured V4 types."""

    try:
        import zstandard
        from safetensors.torch import load as load_safetensors
    except ImportError as error:
        raise RuntimeError("FirstLight sequence cache requires safetensors and zstandard") from error
    if not isinstance(payload, bytes) or len(payload) < _HEADER.size:
        raise FirstLightSequenceCacheError("FirstLight V4 sequence payload is truncated")
    magic, metadata_length = _HEADER.unpack_from(payload)
    if magic != _MAGIC or metadata_length > len(payload) - _HEADER.size:
        raise FirstLightSequenceCacheError("unsupported FirstLight V4 sequence payload")
    metadata_start = _HEADER.size
    metadata_end = metadata_start + metadata_length
    try:
        metadata = json.loads(payload[metadata_start:metadata_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FirstLightSequenceCacheError("invalid FirstLight V4 sequence metadata") from error
    if not isinstance(metadata, dict) or metadata.get("schema") != _SCHEMA:
        raise FirstLightSequenceCacheError("unsupported FirstLight V4 actor sequence schema")
    tensor_bytes = metadata.get("tensor_bytes")
    if metadata.get("tensor_codec") != "safetensors+zstd" or type(tensor_bytes) is not int or tensor_bytes <= 0:
        raise FirstLightSequenceCacheError("unsupported FirstLight V4 tensor codec metadata")
    try:
        safetensors_payload = zstandard.ZstdDecompressor().decompress(
            payload[metadata_end:],
            max_output_size=tensor_bytes,
        )
        if len(safetensors_payload) != tensor_bytes:
            raise FirstLightSequenceCacheError("FirstLight V4 decompressed tensor length mismatch")
        tensors = load_safetensors(safetensors_payload)
    except FirstLightSequenceCacheError:
        raise
    except Exception as error:
        raise FirstLightSequenceCacheError("invalid FirstLight V4 safetensors payload") from error
    if sorted(tensors) != metadata.get("tensor_keys"):
        raise FirstLightSequenceCacheError("V4 tensor-key index differs from safetensors payload")
    if metadata.get("input_mode") not in {"rich", "legacy_checkpoint"}:
        raise FirstLightSequenceCacheError("unsupported FirstLight V4 input mode")
    if not isinstance(metadata.get("frames"), list) or not metadata["frames"]:
        raise FirstLightSequenceCacheError("FirstLight V4 actor sequence contains no frames")

    if components is None:
        from .tensorizer import load_v4_components

        components = load_v4_components(runtime)
    identities: list[DecisionIdentity] = []
    inputs: list[PolicyInput] = []
    labels: list[CompiledLabel] = []
    consumed: set[str] = set()
    try:
        profile = ObservationProfile(metadata["profile"])
        config = training_config or TrainingConfig(profile=profile)
        if config.profile is not profile:
            raise ValueError("cached sequence profile differs from the pinned TrainingConfig")
        if metadata["input_mode"] != config.input_mode:
            raise ValueError("cached input mode differs from the pinned TrainingConfig")
        episode_id = metadata["episode_id"]
        actor = metadata["actor"]
        if not isinstance(episode_id, str) or type(actor) is not int or actor not in (0, 1):
            raise ValueError("invalid actor sequence identity")
        for index, frame in enumerate(metadata["frames"]):
            identity_data = frame["identity"]
            identity = DecisionIdentity(
                episode_id=str(identity_data["episode_id"]),
                tick=int(identity_data["tick"]),
                actor=int(identity_data["actor"]),
            )
            if identity.episode_id != episode_id or identity.actor != actor:
                raise ValueError("frame identity differs from actor sequence")
            batch = _restore_tensor_record(
                components.types.UniversalSemanticBatchV4,
                prefix=f"frame.{index}.observation",
                tensors=tensors,
                consumed=consumed,
            )
            action = _restore_tensor_record(
                components.types.ActionSequenceV4,
                prefix=f"frame.{index}.action",
                tensors=tensors,
                consumed=consumed,
            )
            from .tensorizer import TensorizedDecision

            tensorized = TensorizedDecision(
                batch=batch,
                candidate_index_by_slot={},
                candidate_uid_by_slot={},
                actor=actor,
                input_mode=metadata["input_mode"],
            )
            inputs.append(
                PolicyInput(
                    identity=identity,
                    profile=profile.value,
                    tensors=tensorized,
                    action_mask=None,
                    temporal_context=None,
                )
            )
            category = LabelCategory(frame["category"])
            supervision = frame["supervision"]
            expected_masks = {
                "gate", "candidate", "target", "continue_action", "delay", "value"
            }
            if (
                not isinstance(supervision, dict)
                or set(supervision) != expected_masks
                or any(type(value) is not bool for value in supervision.values())
            ):
                raise ValueError("invalid label supervision mask")
            trainable = frame["trainable"]
            reason = frame.get("report_reason")
            if type(trainable) is not bool or reason is not None and not isinstance(reason, str):
                raise ValueError("invalid label status metadata")
            supervision_value = HeadSupervision(**supervision)
            try:
                validate_label_supervision(category, supervision_value, trainable, config)
            except (TypeError, ValueError) as error:
                raise ValueError(f"cached label violates TrainingConfig: {error}") from error
            labels.append(
                CompiledLabel(
                    category=category,
                    action=action,
                    supervision=supervision_value,
                    trainable=trainable,
                    report_reason=reason,
                )
            )
            identities.append(identity)
    except (KeyError, TypeError, ValueError) as error:
        raise FirstLightSequenceCacheError(f"invalid FirstLight actor sequence metadata: {error}") from error
    if consumed != set(tensors):
        raise FirstLightSequenceCacheError("FirstLight V4 sequence contains unindexed tensor data")
    return CompiledActorSequence(
        episode_id=episode_id,
        actor=actor,
        profile=profile,
        identities=tuple(identities),
        inputs=tuple(inputs),
        labels=tuple(labels),
    )
