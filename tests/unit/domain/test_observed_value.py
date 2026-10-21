from __future__ import annotations

import pytest

from elixir_echo.domain.values import (
    DomainContractError,
    ObservedValue,
    canonical_json,
    content_sha256,
)


def test_observed_value_preserves_known_unknown_and_not_applicable_states() -> None:
    known = ObservedValue.known({"phase": "active"})
    unknown = ObservedValue.unknown()
    not_applicable = ObservedValue.not_applicable()

    assert known.to_dict() == {"status": "known", "value": {"phase": "active"}}
    assert unknown.to_dict() == {"status": "unknown"}
    assert not_applicable.to_dict() == {"status": "not_applicable"}


def test_observed_value_rejects_invalid_or_value_carrying_unknown_states() -> None:
    with pytest.raises(DomainContractError, match="invalid observed value status"):
        ObservedValue("missing")

    with pytest.raises(DomainContractError, match="cannot carry a value"):
        ObservedValue("unknown", 3)


def test_observed_value_is_recursively_immutable() -> None:
    observed = ObservedValue.known(
        {"effects": [{"phase": "active"}], "slots": [1, 2]}
    )

    with pytest.raises(TypeError):
        observed.value["effects"][0]["phase"] = "finished"

    with pytest.raises(TypeError):
        observed.value["slots"][0] = 99


def test_canonical_json_and_content_hash_ignore_mapping_insertion_order() -> None:
    first = {"actor": 0, "state": {"hp": 10, "owner": 1}}
    second = {"state": {"owner": 1, "hp": 10}, "actor": 0}

    assert canonical_json(first) == canonical_json(second)
    assert content_sha256(first) == content_sha256(second)
    assert len(content_sha256(first)) == 64
