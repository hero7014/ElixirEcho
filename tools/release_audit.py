"""Inspect the prospective Git payload without publishing or changing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


_BINARY_SUFFIXES = frozenset({
    ".apk", ".bin", ".ckpt", ".dll", ".exe", ".h5", ".onnx",
    ".parquet", ".pt", ".pth", ".safetensors", ".so", ".zip",
})
_PRIVATE_PARTS = frozenset({"artifacts", "checkpoints", "data", "models", "runs", ".venv", "__pycache__"})
_MAX_FILE_BYTES = 2_000_000


def prospective_files(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("release audit requires a readable Git checkout")
    return tuple(sorted(
        (root / name for name in result.stdout.decode("utf-8").split("\0") if name),
        key=lambda path: path.as_posix(),
    ))


def audit_repository(root: Path) -> dict[str, Any]:
    base = Path(root).resolve()
    findings: list[dict[str, str]] = []
    files = prospective_files(base)
    for file in files:
        relative = file.relative_to(base)
        name = relative.as_posix()
        if not file.exists() or not file.is_file():
            findings.append({"path": name, "reason": "missing_or_non_file"})
            continue
        if file.is_symlink() or not file.resolve().is_relative_to(base):
            findings.append({"path": name, "reason": "symlink_or_outside_checkout"})
            continue
        if any(part.casefold() in _PRIVATE_PARTS for part in relative.parts[:-1]):
            findings.append({"path": name, "reason": "private_artifact_directory"})
        if file.suffix.casefold() in _BINARY_SUFFIXES:
            findings.append({"path": name, "reason": "private_or_binary_artifact"})
        if file.stat().st_size > _MAX_FILE_BYTES:
            findings.append({"path": name, "reason": "file_exceeds_public_source_limit"})
    return {
        "schema": "elixir_echo.release_audit@1",
        "status": "passed" if not findings else "failed",
        "prospective_file_count": len(files),
        "findings": findings,
        "note": "Checks prospective file paths, type, and size; text secrets/paths are checked by tests/release/test_no_local_paths.py.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = audit_repository(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
