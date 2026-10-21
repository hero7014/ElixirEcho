from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from elixir_echo.provenance import (
    InventoryEntry,
    build_inventory,
    render_inventory_manifest,
)


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_build_inventory_is_sorted_and_hashes_file_bytes(tmp_path: Path) -> None:
    root = tmp_path / "source"
    later = _write(root / "src" / "zeta.py", b"print('zeta')\n")
    earlier = _write(root / "src" / "alpha.py", b"print('alpha')\n")

    entries = build_inventory(root, [later, earlier])

    assert [entry.path for entry in entries] == ["src/alpha.py", "src/zeta.py"]
    assert all(isinstance(entry, InventoryEntry) for entry in entries)
    assert entries[0].size == len(b"print('alpha')\n")
    assert entries[0].sha256 == (
        "2fed55f1437cb5d2c9d3529fe2d09fe465a663b62f5caae159cae1048ba952a4"
    )


def test_build_inventory_rejects_paths_outside_source_root(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    outside = _write(tmp_path / "outside.py", b"pass\n")

    with pytest.raises(ValueError, match="outside source root"):
        build_inventory(root, [outside])


def test_build_inventory_classifies_material_without_reading_policy_into_tests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    paths = [
        _write(root / "src" / "module.py", b"pass\n"),
        _write(root / "tests" / "test_module.py", b"def test_ok(): pass\n"),
        _write(root / "docs" / "guide.md", b"# Guide\n"),
        _write(root / ".pytest_cache" / "README.md", b"cache\n"),
        _write(root / "runs" / "checkpoint.pt", b"private model bytes"),
        _write(root / ".env", b"IGNORED=1\n"),
    ]

    entries = build_inventory(root, paths)
    classes = {entry.path: entry.classification for entry in entries}

    assert classes == {
        ".env": "excluded",
        ".pytest_cache/README.md": "generated",
        "docs/guide.md": "documentation",
        "runs/checkpoint.pt": "private_artifact",
        "src/module.py": "source",
        "tests/test_module.py": "test",
    }


def test_manifest_is_deterministic_and_contains_only_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    source = _write(root / "src" / "module.py", b"pass\n")
    entries = build_inventory(root, [source])
    checkpoint = {
        "artifact_name": "ElixirEchoBaseline",
        "checkpoint_id": "3fa9bd4e716744a49e30ca0e2ce7b8e3",
        "sha256": "49a76994020c49e60b3b7d90b7eaf7cc53ffaef3c679f3caa4130cb5c0c71548",
        "size": 145_280_508,
        "training_report_sha256": "abc123",
    }

    first = render_inventory_manifest(entries, checkpoint=checkpoint)
    second = render_inventory_manifest(entries, checkpoint=checkpoint)
    payload = json.loads(first)

    assert first == second
    assert payload["schema"] == "elixir_echo.source_inventory@1"
    assert payload["formal_checkpoint"] == checkpoint
    assert "private model bytes" not in first
    assert str(root.resolve()) not in first


def test_committed_manifest_contains_only_publishable_entries_and_metadata() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    manifest_path = repository_root / "docs" / "provenance" / "source-inventory.json"
    rendered = manifest_path.read_text(encoding="utf-8")
    payload = json.loads(rendered)

    assert payload["entry_count"] == 102
    assert payload["classification_counts"] == {
        "documentation": 12,
        "source": 55,
        "test": 35,
    }
    assert payload["formal_checkpoint"]["checkpoint_id"] == (
        "3fa9bd4e716744a49e30ca0e2ce7b8e3"
    )
    assert payload["formal_checkpoint"]["sha256"] == (
        "49a76994020c49e60b3b7d90b7eaf7cc53ffaef3c679f3caa4130cb5c0c71548"
    )
    assert payload["formal_checkpoint"]["size"] == 145_280_508
    assert all(
        entry["classification"] in {"source", "test", "documentation"}
        for entry in payload["entries"]
    )
    assert re.search(r"[A-Za-z]:[\\/]", rendered) is None
    assert "source_root" not in payload
    assert "private model bytes" not in rendered
