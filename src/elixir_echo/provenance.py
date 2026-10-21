"""Deterministic source inventory utilities for migration provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


INVENTORY_SCHEMA = "elixir_echo.source_inventory@1"
PUBLISHABLE_CLASSIFICATIONS = frozenset({"source", "test", "documentation"})


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    """One immutable file identity relative to an inventory root."""

    path: str
    size: int
    sha256: str
    classification: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification(relative_path: Path) -> str:
    parts = tuple(part.lower() for part in relative_path.parts)
    name = relative_path.name.lower()
    suffix = relative_path.suffix.lower()

    if (
        name == ".env"
        or name.startswith(".env.")
        or suffix in {".key", ".pem"}
        or ".git" in parts
    ):
        return "excluded"

    if (
        any(part in {"data", "runs", "artifacts", "checkpoints", "models"} for part in parts)
        or suffix in {".pt", ".pth", ".ckpt", ".parquet", ".apk", ".exe", ".dll"}
    ):
        return "private_artifact"

    if (
        any(
            part
            in {
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                "build",
                "dist",
            }
            for part in parts
        )
        or suffix in {".pyc", ".pyo", ".log"}
    ):
        return "generated"

    if "tests" in parts or name.startswith("test_"):
        return "test"

    if "docs" in parts or suffix in {".md", ".rst"}:
        return "documentation"

    if suffix in {".py", ".ps1", ".toml"}:
        return "source"

    return "excluded"


def _files_beneath(root: Path, candidate: Path) -> tuple[Path, ...]:
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"inventory path is outside source root: {candidate}") from error

    if resolved.is_file():
        return (resolved,)
    if resolved.is_dir():
        return tuple(path for path in resolved.rglob("*") if path.is_file())
    return ()


def build_inventory(
    root: Path,
    include: Sequence[Path],
) -> tuple[InventoryEntry, ...]:
    """Hash included files beneath ``root`` in stable relative-path order."""

    resolved_root = root.resolve(strict=True)
    files: set[Path] = set()
    for item in include:
        candidate = item if item.is_absolute() else resolved_root / item
        files.update(_files_beneath(resolved_root, candidate))

    entries = []
    for path in sorted(files, key=lambda value: value.relative_to(resolved_root).as_posix()):
        relative = path.relative_to(resolved_root)
        entries.append(
            InventoryEntry(
                path=relative.as_posix(),
                size=path.stat().st_size,
                sha256=_sha256(path),
                classification=_classification(relative),
            )
        )
    return tuple(entries)


def _aggregate_sha256(entries: Sequence[InventoryEntry]) -> str:
    canonical = json.dumps(
        [asdict(entry) for entry in entries],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def render_inventory_manifest(
    entries: Sequence[InventoryEntry],
    *,
    checkpoint: Mapping[str, Any],
) -> str:
    """Render a byte-stable manifest without timestamps or machine-local roots."""

    ordered = tuple(sorted(entries, key=lambda entry: entry.path))
    counts = Counter(entry.classification for entry in ordered)
    payload = {
        "schema": INVENTORY_SCHEMA,
        "source_alias": "legacy_source",
        "entry_count": len(ordered),
        "classification_counts": dict(sorted(counts.items())),
        "aggregate_sha256": _aggregate_sha256(ordered),
        "formal_checkpoint": dict(checkpoint),
        "entries": [asdict(entry) for entry in ordered],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _checkpoint_metadata(
    checkpoint_path: Path,
    training_report_path: Path,
    checkpoint_id: str,
) -> dict[str, Any]:
    return {
        "artifact_name": "ElixirEchoBaseline",
        "checkpoint_id": checkpoint_id,
        "relative_source_path": "runs/checkpoint_v1_formal_epoch1/checkpoint-v1.pt",
        "sha256": _sha256(checkpoint_path),
        "size": checkpoint_path.stat().st_size,
        "training_report_sha256": _sha256(training_report_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    entries = build_inventory(args.root, args.include)
    publishable = tuple(
        entry for entry in entries if entry.classification in PUBLISHABLE_CLASSIFICATIONS
    )
    checkpoint = _checkpoint_metadata(
        args.checkpoint,
        args.training_report,
        args.checkpoint_id,
    )
    rendered = render_inventory_manifest(publishable, checkpoint=checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
