from __future__ import annotations

import pytest

zstandard = pytest.importorskip("zstandard")

from elixir_echo.integrations.firstlight.sequence_cache import (
    FirstLightSequenceCacheError,
    _HEADER,
    _MAGIC,
    _SCHEMA,
    decode_actor_sequence,
)
from elixir_echo.domain.values import canonical_json


def test_decode_actor_sequence_preserves_decompressed_length_error() -> None:
    metadata = {
        "schema": _SCHEMA,
        "tensor_codec": "safetensors+zstd",
        "tensor_bytes": 2,
        "tensor_keys": [],
    }
    metadata_bytes = canonical_json(metadata).encode("utf-8")
    compressed = zstandard.ZstdCompressor().compress(b"x")
    payload = _HEADER.pack(_MAGIC, len(metadata_bytes)) + metadata_bytes + compressed

    with pytest.raises(FirstLightSequenceCacheError, match="decompressed tensor length mismatch"):
        decode_actor_sequence(payload, runtime=object())
