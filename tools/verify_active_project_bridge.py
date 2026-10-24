"""Run read-only compatibility gates against an explicitly selected legacy run."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Iterator, Mapping


class BridgeConfigurationError(ValueError):
    """Raised when a bridge input is absent or could violate read-only scope."""


_LAYER2_CONTAINER_MAGIC = b"SFRCV401"
_LAYER2_CONTAINER_HEADER = struct.Struct("<8sQ")
_LAYER2_MANIFEST_SCHEMA = "semantic-recurrent-v4-cache-shard.v1"
_LAYER2_SUMMARY_SCHEMA = "semantic-recurrent-v4-cache-summary.v1"
_MAX_LAYER2_MANIFEST_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BridgeInputs:
    baseline_root: Path
    episode_path: Path
    firstlight_root: Path
    checkpoint_path: Path
    layer2_summary_path: Path


def validate_inputs(inputs: BridgeInputs) -> BridgeInputs:
    """Resolve and validate every explicit legacy/provider input before imports."""

    resolved = BridgeInputs(
        baseline_root=inputs.baseline_root.expanduser().resolve(),
        episode_path=inputs.episode_path.expanduser().resolve(),
        firstlight_root=inputs.firstlight_root.expanduser().resolve(),
        checkpoint_path=inputs.checkpoint_path.expanduser().resolve(),
        layer2_summary_path=inputs.layer2_summary_path.expanduser().resolve(),
    )
    required = (
        (resolved.baseline_root / "src" / "training" / "tensorizer_v1.py", "legacy tensorizer"),
        (resolved.baseline_root / "src" / "training" / "recurrent_cache.py", "legacy recurrent cache loader"),
        (resolved.episode_path / "records.jsonl", "legacy episode records.jsonl"),
        (resolved.episode_path / "episode_manifest.json", "legacy episode manifest"),
        (resolved.episode_path / "episode_context.json", "legacy episode context"),
        (resolved.firstlight_root / "native_runner" / "__init__.py", "FirstLight native_runner"),
        (resolved.checkpoint_path, "formal checkpoint"),
        (resolved.layer2_summary_path, "legacy Layer 2 shard summary"),
    )
    for path, description in required:
        if not path.is_file():
            raise BridgeConfigurationError(f"{description} is missing: {path}")

    try:
        summary = json.loads(resolved.layer2_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeConfigurationError(f"cannot read Layer 2 shard summary: {error}") from error
    try:
        episode_manifest = json.loads(
            (resolved.episode_path / "episode_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeConfigurationError(f"cannot read selected episode manifest: {error}") from error
    episode_id = episode_manifest.get("episode_id") if isinstance(episode_manifest, dict) else None
    if not isinstance(episode_id, str) or not episode_id:
        raise BridgeConfigurationError("selected episode manifest has no episode_id")
    if resolved.episode_path.name != episode_id:
        raise BridgeConfigurationError(
            "selected episode directory name does not match its manifest episode_id"
        )
    data_file = summary.get("data_file") if isinstance(summary, dict) else None
    if not isinstance(data_file, str) or not data_file:
        raise BridgeConfigurationError(
            "Layer 2 path must be a shard summary with a relative data_file, not a cache catalog"
        )
    data_path = (resolved.layer2_summary_path.parent / data_file).resolve()
    if not data_path.is_relative_to(resolved.layer2_summary_path.parent):
        raise BridgeConfigurationError("Layer 2 data_file must stay beside its summary")
    if not data_path.is_file():
        raise BridgeConfigurationError(f"Layer 2 shard data is missing: {data_path}")
    if summary.get("version") != _LAYER2_SUMMARY_SCHEMA:
        raise BridgeConfigurationError("unsupported Layer 2 shard summary schema")
    _require_episode_in_layer2_shard(data_path, summary, episode_id)
    if not resolved.baseline_root.is_dir() or not resolved.firstlight_root.is_dir():
        raise BridgeConfigurationError("legacy baseline and FirstLight roots must be directories")
    return resolved


def _require_episode_in_layer2_shard(
    data_path: Path,
    summary: Mapping[str, object],
    episode_id: str,
) -> None:
    """Verify the selected episode is in this shard without decoding its tensors."""

    try:
        import zstandard
    except ImportError as error:
        raise BridgeConfigurationError("Layer 2 membership validation requires zstandard") from error
    try:
        compressed = data_path.read_bytes()
    except OSError as error:
        raise BridgeConfigurationError(f"cannot read Layer 2 shard data: {error}") from error
    if len(compressed) != summary.get("compressed_bytes"):
        raise BridgeConfigurationError("Layer 2 shard compressed byte count mismatch")
    expected_compressed_hash = summary.get("compressed_sha256")
    if not isinstance(expected_compressed_hash, str) or _sha256_bytes(compressed) != expected_compressed_hash:
        raise BridgeConfigurationError("Layer 2 shard compressed SHA-256 mismatch")

    try:
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
            header = reader.read(_LAYER2_CONTAINER_HEADER.size)
            if len(header) != _LAYER2_CONTAINER_HEADER.size:
                raise BridgeConfigurationError("Layer 2 shard container header is truncated")
            magic, manifest_length = _LAYER2_CONTAINER_HEADER.unpack(header)
            if magic != _LAYER2_CONTAINER_MAGIC:
                raise BridgeConfigurationError("unsupported Layer 2 shard container")
            if manifest_length <= 0 or manifest_length > _MAX_LAYER2_MANIFEST_BYTES:
                raise BridgeConfigurationError("Layer 2 shard manifest length is invalid")
            manifest_payload = reader.read(manifest_length)
    except BridgeConfigurationError:
        raise
    except (OSError, zstandard.ZstdError) as error:
        raise BridgeConfigurationError(f"cannot read Layer 2 shard manifest: {error}") from error
    if len(manifest_payload) != manifest_length:
        raise BridgeConfigurationError("Layer 2 shard manifest is truncated")
    expected_manifest_hash = summary.get("manifest_sha256")
    if not isinstance(expected_manifest_hash, str) or _sha256_bytes(manifest_payload) != expected_manifest_hash:
        raise BridgeConfigurationError("Layer 2 shard manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_payload)
    except json.JSONDecodeError as error:
        raise BridgeConfigurationError(f"Layer 2 shard manifest is invalid JSON: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != _LAYER2_MANIFEST_SCHEMA:
        raise BridgeConfigurationError("unsupported Layer 2 shard manifest schema")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list):
        raise BridgeConfigurationError("Layer 2 shard manifest has no sequence list")
    if not any(
        isinstance(sequence, dict) and sequence.get("episode_id") == episode_id
        for sequence in sequences
    ):
        raise BridgeConfigurationError(
            f"Layer 2 shard does not contain selected episode {episode_id}"
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_report_path(report_path: Path, inputs: BridgeInputs) -> Path:
    """Refuse to write a report into either explicitly read-only external tree."""

    target = report_path.expanduser().resolve()
    roots = (
        inputs.baseline_root.resolve(),
        inputs.firstlight_root.resolve(),
        inputs.episode_path.resolve(),
        inputs.checkpoint_path.resolve().parent,
        inputs.layer2_summary_path.resolve().parent,
    )
    if any(target == root or target.is_relative_to(root) for root in roots):
        raise BridgeConfigurationError("report output is inside a read-only legacy/provider tree")
    layer2_payload = inputs.layer2_summary_path.parent / json.loads(
        inputs.layer2_summary_path.read_text(encoding="utf-8")
    )["data_file"]
    protected_files = (
        inputs.checkpoint_path.resolve(),
        inputs.layer2_summary_path.resolve(),
        layer2_payload.resolve(),
    )
    if target in protected_files:
        raise BridgeConfigurationError("report output matches a read-only input file")
    if target.exists():
        raise BridgeConfigurationError(f"refusing to overwrite an existing report: {target}")
    return target


@contextmanager
def temporary_environment(values: Mapping[str, str | None]) -> Iterator[None]:
    """Temporarily set/remove variables and restore them even after a failed check."""

    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_inventory(root: Path) -> dict[str, tuple[int, str]]:
    if not root.exists():
        return {}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (path.suffix in {".py", ".pyc"} or "__pycache__" in path.parts)
    )
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, _hash_file(path))
        for path in files
    }


def _input_inventory(inputs: BridgeInputs) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    episode_files = sorted(path for path in inputs.episode_path.rglob("*") if path.is_file())
    summary_payload = json.loads(inputs.layer2_summary_path.read_text(encoding="utf-8"))
    shard_data = (inputs.layer2_summary_path.parent / summary_payload["data_file"]).resolve()
    files = [
        *episode_files,
        inputs.checkpoint_path,
        inputs.layer2_summary_path,
        shard_data,
    ]
    for path in files:
        result[str(path)] = (path.stat().st_size, _hash_file(path))
    return result


class _PytestResultCollector:
    def __init__(self) -> None:
        self.results: dict[str, str] = {}

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" or report.failed or report.skipped:
            self.results[report.nodeid] = report.outcome

    def pytest_collectreport(self, report) -> None:
        if report.failed or report.skipped:
            self.results[report.nodeid] = report.outcome


_REQUIRED_TESTS = {
    "legacy episode integrity and deck-free reader":
        "test_opt_in_legacy_episode_is_verified_and_mapped_without_rewrite",
    "legacy checkpoint semantic/logit parity":
        "test_legacy_semantic_frame_path_audits_elixirecho_checkpoint_compatibility",
    "sealed Layer 2 parity":
        "test_layer2_compiler_matches_existing_sealed_firstlight_sequence_cache",
    "checkpoint wrapper parity":
        "test_formal_checkpoint_policy_wrapper_matches_raw_v4_forward",
    "FirstLight read-only import":
        "test_configured_firstlight_import_and_adaptation_leave_checkout_unchanged",
}


def _required_test_results(results: Mapping[str, str]) -> tuple[dict[str, str], bool]:
    """Require each named compatibility gate to be collected and pass exactly once."""

    checks: dict[str, str] = {}
    for label, function_name in _REQUIRED_TESTS.items():
        matches = [
            status
            for nodeid, status in results.items()
            if nodeid == function_name or nodeid.endswith(f"::{function_name}")
        ]
        checks[label] = matches[0] if len(matches) == 1 else "missing" if not matches else "duplicate"
    return checks, all(status == "passed" for status in checks.values())


def _external_import_path_deltas(
    before: tuple[str, ...] | list[str],
    after: tuple[str, ...] | list[str],
    inputs: BridgeInputs,
) -> list[str]:
    external_roots = {
        str((inputs.baseline_root / "src").resolve()),
        str(inputs.firstlight_root.resolve()),
        str((inputs.firstlight_root / "native_runner").resolve()),
    }

    def normalized(paths: tuple[str, ...] | list[str]) -> set[str]:
        output: set[str] = set()
        for item in paths:
            if item:
                try:
                    output.add(str(Path(item).resolve()))
                except OSError:
                    output.add(str(Path(item)))
        return output

    return sorted((normalized(after) - normalized(before)) & external_roots)


def _external_import_paths_rolled_back(
    before: tuple[str, ...] | list[str],
    after: tuple[str, ...] | list[str],
    inputs: BridgeInputs,
) -> bool:
    return not _external_import_path_deltas(before, after, inputs)


def _restore_sys_path(original: tuple[str, ...], current: list[str]) -> bool:
    """Restore the bridge process import path after compatibility code has run."""

    current[:] = original
    return tuple(current) == original


def _package_import_paths(package: object) -> tuple[Path, ...]:
    """Return concrete package locations for normal or namespace packages."""

    package_file = getattr(package, "__file__", None)
    if isinstance(package_file, str) and package_file:
        return (Path(package_file).resolve(),)
    package_locations = getattr(package, "__path__", None)
    if package_locations is not None:
        locations = tuple(dict.fromkeys(Path(item).resolve() for item in package_locations))
        if locations:
            return locations
    raise BridgeConfigurationError("ElixirEcho import has no reportable package location")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the installed ElixirEcho bridge against explicit read-only legacy cache, "
            "FirstLight provider and checkpoint paths."
        )
    )
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--legacy-episode", required=True, type=Path)
    parser.add_argument("--firstlight-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--layer2-summary", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="optional new JSON report path; by default the report is printed only",
    )
    return parser


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report_path: Path | None = None
    try:
        inputs = validate_inputs(
            BridgeInputs(
                baseline_root=args.baseline_root,
                episode_path=args.legacy_episode,
                firstlight_root=args.firstlight_root,
                checkpoint_path=args.checkpoint,
                layer2_summary_path=args.layer2_summary,
            )
        )
        report_path = validate_report_path(args.report, inputs) if args.report else None
        package = importlib.import_module("elixir_echo")
        package_version = importlib.metadata.version("elixir-echo")
        package_paths = _package_import_paths(package)
    except (BridgeConfigurationError, importlib.metadata.PackageNotFoundError, ImportError) as error:
        report: dict[str, object] = {
            "schema": "elixir_echo.active-project-bridge-report@1",
            "status": "configuration_error",
            "error": str(error),
            "note": "No compatibility tests ran; no legacy assets were modified.",
        }
        if report_path is not None:
            _write_report(report_path, report)
            report["report_path"] = str(report_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    baseline_source = inputs.baseline_root / "src"
    external_source_roots = (
        baseline_source,
        inputs.firstlight_root / "native_runner",
    )
    source_before = {str(root): _file_inventory(root) for root in external_source_roots}
    inputs_before = _input_inventory(inputs)
    initial_sys_path = tuple(sys.path)
    collector = _PytestResultCollector()
    test_paths = (
        repo_root / "tests" / "compatibility" / "test_legacy_trajectory.py",
        repo_root / "tests" / "compatibility" / "test_legacy_checkpoint_parity.py",
        repo_root / "tests" / "compatibility" / "test_legacy_sequence_cache_parity.py",
        repo_root / "tests" / "compatibility" / "test_firstlight_checkpoint_logits.py",
        repo_root / "tests" / "integration" / "test_firstlight_read_only.py",
    )
    env = {
        "ELIXIR_ECHO_BASELINE_ROOT": str(inputs.baseline_root),
        "ELIXIR_ECHO_LEGACY_EPISODE": str(inputs.episode_path),
        "ELIXIR_ECHO_FIRSTLIGHT_ROOT": str(inputs.firstlight_root),
        "ELIXIR_ECHO_FIRSTLIGHT_CHECKPOINT": str(inputs.checkpoint_path),
        "ELIXIR_ECHO_LEGACY_LAYER2_SUMMARY": str(inputs.layer2_summary_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    old_cwd = Path.cwd()
    external_imports_added: list[str] = []
    sys_path_restored = False
    try:
        os.chdir(repo_root)
        with temporary_environment(env):
            pytest = importlib.import_module("pytest")
            result_code = int(
                pytest.main(
                    [
                        "-q",
                        "-s",
                        "-p",
                        "no:cacheprovider",
                        *(str(path) for path in test_paths),
                    ],
                    plugins=[collector],
                )
            )
    except Exception as error:
        result_code = 2
        collector.results["bridge_runner"] = f"failed: {type(error).__name__}: {error}"
    finally:
        external_imports_added = _external_import_path_deltas(initial_sys_path, sys.path, inputs)
        sys_path_restored = _restore_sys_path(initial_sys_path, sys.path)
        os.chdir(old_cwd)
        sys.dont_write_bytecode = old_dont_write_bytecode

    source_after = {str(root): _file_inventory(root) for root in external_source_roots}
    inputs_after = _input_inventory(inputs)
    external_import_paths_rolled_back = sys_path_restored
    source_unchanged = source_before == source_after
    inputs_unchanged = inputs_before == inputs_after
    required_test_results, required_tests_passed = _required_test_results(collector.results)
    expected_tests = len(_REQUIRED_TESTS)
    passed_count = sum(status == "passed" for status in collector.results.values())
    skipped_count = sum(status == "skipped" for status in collector.results.values())
    failed_count = sum(status != "passed" for status in collector.results.values())
    status = "passed" if (
        result_code == 0
        and passed_count == expected_tests
        and skipped_count == 0
        and failed_count == 0
        and required_tests_passed
        and source_unchanged
        and inputs_unchanged
        and external_import_paths_rolled_back
    ) else "failed"
    report: dict[str, object] = {
        "schema": "elixir_echo.active-project-bridge-report@1",
        "status": status,
        "elixir_echo": {
            "version": package_version,
            "import_paths": [str(path) for path in package_paths],
        },
        "legacy_episode_id": inputs.episode_path.name,
        "checkpoint": str(inputs.checkpoint_path),
        "legacy_layer2_summary": str(inputs.layer2_summary_path),
        "checks": {
            "pytest_result": result_code,
            "passed": passed_count,
            "skipped": skipped_count,
            "failed_or_missing": failed_count,
            "source_inventory_unchanged": source_unchanged,
            "cache_and_checkpoint_hashes_unchanged": inputs_unchanged,
            "temporary_external_import_paths_rolled_back": external_import_paths_rolled_back,
            "external_import_paths_added_during_checks": external_imports_added,
            "process_sys_path_restored": sys_path_restored,
            "required_compatibility_checks": required_test_results,
        },
        "test_results": dict(sorted(collector.results.items())),
        "firstlight_root": str(inputs.firstlight_root),
        "baseline_root": str(inputs.baseline_root),
        "note": "Read-only verification only; no Native capture, training, or checkpoint writes were run.",
    }
    if report_path is not None:
        _write_report(report_path, report)
        report["report_path"] = str(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
