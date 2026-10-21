from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from elixir_echo.integrations.firstlight.observation import FirstLightObservationAdapter
from elixir_echo.integrations.firstlight.runtime import (
    FirstLightConfigurationError,
    FirstLightRuntime,
)


pytestmark = pytest.mark.firstlight


def _source_inventory(root: Path) -> dict[str, tuple[int, str]]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix in {".py", ".pyc"} or "__pycache__" in path.parts)
    )
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in files
    }


def test_configured_firstlight_import_and_adaptation_leave_checkout_unchanged() -> None:
    try:
        runtime = FirstLightRuntime.from_environment()
    except FirstLightConfigurationError as error:
        pytest.skip(str(error))
    before = _source_inventory(runtime.root)

    contracts = runtime.import_module("native_runner.contracts")
    source = contracts.ObservationV1(
        tier=contracts.ObservationTier.FAIR,
        tick=90,
        owner=0,
        time={
            "elapsed_ms": 4500,
            "remaining_ms": 175_500,
            "elixir_multiplier": 1.0,
        },
        phase="normal",
        players=(
            {
                "owner": 0,
                "crowns": 0,
                "tower_ids": (),
                "elixir_exact": 5.0,
                "hand": (1, 2, 3, 4),
                "next_card": 5,
                "private_state_visible": True,
            },
            {
                "owner": 1,
                "crowns": 0,
                "tower_ids": (),
                "private_state_visible": False,
            },
        ),
    )
    adapted = FirstLightObservationAdapter().adapt(source, actor=0)

    assert isinstance(source, contracts.ObservationV1)
    assert adapted.perspective == "actor_fair"
    assert adapted.players[0]["self_private"]["hand"] == (1, 2, 3, 4)
    assert _source_inventory(runtime.root) == before
