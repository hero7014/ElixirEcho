from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest

zstandard = pytest.importorskip("zstandard")

from tools.verify_active_project_bridge import (
    BridgeConfigurationError,
    BridgeInputs,
    _PytestResultCollector,
    _external_import_paths_rolled_back,
    _package_import_paths,
    _restore_sys_path,
    _required_test_results,
    temporary_environment,
    validate_inputs,
    validate_report_path,
)


def _bridge_tree(
    root: Path,
    *,
    episode_outside_baseline: bool = False,
    assets_outside_baseline: bool = False,
) -> BridgeInputs:
    baseline = root / "legacy-project"
    (baseline / "src" / "training").mkdir(parents=True)
    (baseline / "src" / "training" / "tensorizer_v1.py").write_text("# legacy\n")
    (baseline / "src" / "training" / "recurrent_cache.py").write_text("# cache\n")
    episode = (
        root / "external-episode" / "episode-1"
        if episode_outside_baseline
        else baseline / "data" / "episodes" / "episode-1"
    )
    episode.mkdir(parents=True)
    (episode / "records.jsonl").write_text("{}\n")
    (episode / "episode_manifest.json").write_text(
        json.dumps({"episode_id": "episode-1"}) + "\n"
    )
    (episode / "episode_context.json").write_text("{}\n")
    provider = root / "firstlight"
    (provider / "native_runner").mkdir(parents=True)
    (provider / "native_runner" / "__init__.py").write_text("\n")
    asset_root = root / "external-assets" if assets_outside_baseline else baseline
    checkpoint = asset_root / "runs" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    shard = asset_root / "cache" / "train-00000.crv4.json"
    shard.parent.mkdir(parents=True)
    manifest_payload = json.dumps(
        {
            "schema": "semantic-recurrent-v4-cache-shard.v1",
            "sequences": [{"episode_id": "episode-1"}],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    container = struct.pack("<8sQ", b"SFRCV401", len(manifest_payload)) + manifest_payload
    compressed = zstandard.ZstdCompressor().compress(container)
    data_file = shard.parent / "train-00000.crv4.zst"
    data_file.write_bytes(compressed)
    shard.write_text(
        json.dumps(
            {
                "shard_id": "train-00000",
                "data_file": data_file.name,
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "container_sha256": hashlib.sha256(container).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "compressed_bytes": len(compressed),
                "container_bytes": len(container),
                "sequence_count": 1,
                "frame_count": 1,
                "excluded_count": 0,
                "version": "semantic-recurrent-v4-cache-summary.v1",
            }
        )
        + "\n"
    )
    return BridgeInputs(
        baseline_root=baseline,
        episode_path=episode,
        firstlight_root=provider,
        checkpoint_path=checkpoint,
        layer2_summary_path=shard,
    )


def test_bridge_requires_complete_read_only_inputs(tmp_path: Path) -> None:
    inputs = _bridge_tree(tmp_path)

    validated = validate_inputs(inputs)

    assert validated == inputs


def test_bridge_rejects_incomplete_legacy_episode(tmp_path: Path) -> None:
    inputs = _bridge_tree(tmp_path)
    (inputs.episode_path / "records.jsonl").unlink()

    with pytest.raises(BridgeConfigurationError, match="records.jsonl"):
        validate_inputs(inputs)


def test_report_cannot_write_inside_read_only_project_or_provider(tmp_path: Path) -> None:
    inputs = _bridge_tree(tmp_path)

    with pytest.raises(BridgeConfigurationError, match="read-only"):
        validate_report_path(inputs.baseline_root / "bridge-report.json", inputs)
    with pytest.raises(BridgeConfigurationError, match="read-only"):
        validate_report_path(inputs.firstlight_root / "bridge-report.json", inputs)


def test_report_refuses_to_overwrite_an_existing_file(tmp_path: Path) -> None:
    inputs = _bridge_tree(tmp_path)
    existing = tmp_path / "bridge-report.json"
    existing.write_text("keep this report\n")

    with pytest.raises(BridgeConfigurationError, match="overwrite"):
        validate_report_path(existing, inputs)
    assert existing.read_text() == "keep this report\n"


def test_layer2_summary_cannot_escape_its_shard_directory(tmp_path: Path) -> None:
    inputs = _bridge_tree(tmp_path)
    inputs.layer2_summary_path.write_text('{"data_file":"../outside.zst"}\n')

    with pytest.raises(BridgeConfigurationError, match="stay beside"):
        validate_inputs(inputs)


def test_layer2_shard_must_contain_the_selected_episode(tmp_path: Path) -> None:
    inputs = _bridge_tree(tmp_path)
    shard = inputs.layer2_summary_path.parent / "train-00000.crv4.zst"
    manifest_payload = json.dumps(
        {"schema": "semantic-recurrent-v4-cache-shard.v1", "sequences": []},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    container = struct.pack("<8sQ", b"SFRCV401", len(manifest_payload)) + manifest_payload
    compressed = zstandard.ZstdCompressor().compress(container)
    shard.write_bytes(compressed)
    summary = json.loads(inputs.layer2_summary_path.read_text())
    summary.update(
        {
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "container_sha256": hashlib.sha256(container).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "compressed_bytes": len(compressed),
            "container_bytes": len(container),
            "sequence_count": 0,
            "frame_count": 0,
        }
    )
    inputs.layer2_summary_path.write_text(json.dumps(summary) + "\n")

    with pytest.raises(BridgeConfigurationError, match="does not contain selected episode"):
        validate_inputs(inputs)


def test_report_cannot_write_inside_selected_episode_outside_project_root(tmp_path: Path) -> None:
    inputs = _bridge_tree(tmp_path, episode_outside_baseline=True)

    with pytest.raises(BridgeConfigurationError, match="read-only"):
        validate_report_path(inputs.episode_path / "bridge-report.json", inputs)


def test_report_cannot_write_inside_external_checkpoint_or_shard_directory(tmp_path: Path) -> None:
    inputs = _bridge_tree(tmp_path, assets_outside_baseline=True)

    with pytest.raises(BridgeConfigurationError, match="read-only"):
        validate_report_path(inputs.layer2_summary_path.parent / "bridge-report.json", inputs)
    checkpoint_inputs = replace(
        inputs,
        layer2_summary_path=inputs.baseline_root / "cache" / "unused-summary.json",
    )
    with pytest.raises(BridgeConfigurationError, match="read-only"):
        validate_report_path(inputs.checkpoint_path.parent / "bridge-report.json", checkpoint_inputs)


def test_temporary_environment_restores_values_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELIXIR_ECHO_BRIDGE_TEST_VALUE", "before")

    with pytest.raises(RuntimeError, match="expected test failure"):
        with temporary_environment({"ELIXIR_ECHO_BRIDGE_TEST_VALUE": "during"}):
            assert __import__("os").environ["ELIXIR_ECHO_BRIDGE_TEST_VALUE"] == "during"
            raise RuntimeError("expected test failure")

    assert __import__("os").environ["ELIXIR_ECHO_BRIDGE_TEST_VALUE"] == "before"


def test_required_test_gate_rejects_missing_and_skipped_checks() -> None:
    all_passed = {
        "test_opt_in_legacy_episode_is_verified_and_mapped_without_rewrite": "passed",
        "test_legacy_semantic_frame_path_audits_elixirecho_checkpoint_compatibility": "passed",
        "test_layer2_compiler_matches_existing_sealed_firstlight_sequence_cache": "passed",
        "test_formal_checkpoint_policy_wrapper_matches_raw_v4_forward": "passed",
        "test_configured_firstlight_import_and_adaptation_leave_checkout_unchanged": "passed",
    }
    checks, passed = _required_test_results(all_passed)
    assert passed
    assert all(status == "passed" for status in checks.values())

    skipped = dict(all_passed)
    skipped["test_formal_checkpoint_policy_wrapper_matches_raw_v4_forward"] = "skipped"
    checks, passed = _required_test_results(skipped)
    assert not passed
    assert "skipped" in checks["checkpoint wrapper parity"]

    missing = dict(all_passed)
    missing.pop("test_layer2_compiler_matches_existing_sealed_firstlight_sequence_cache")
    checks, passed = _required_test_results(missing)
    assert not passed
    assert checks["sealed Layer 2 parity"] == "missing"


def test_pytest_collector_records_collection_failure() -> None:
    collector = _PytestResultCollector()

    collector.pytest_collectreport(
        SimpleNamespace(
            failed=True,
            skipped=False,
            outcome="failed",
            nodeid="tests/compatibility/test_missing.py",
        )
    )

    assert collector.results == {"tests/compatibility/test_missing.py": "failed"}
    _, passed = _required_test_results(collector.results)
    assert not passed


def test_external_import_rollback_checks_both_read_only_sources(tmp_path: Path) -> None:
    inputs = _bridge_tree(tmp_path)
    baseline_source = str((inputs.baseline_root / "src").resolve())
    firstlight_root = str(inputs.firstlight_root.resolve())

    assert _external_import_paths_rolled_back([], [], inputs)
    assert not _external_import_paths_rolled_back([], [baseline_source], inputs)
    assert not _external_import_paths_rolled_back([], [firstlight_root], inputs)
    assert _external_import_paths_rolled_back([baseline_source], [baseline_source], inputs)


def test_bridge_restores_full_import_path_after_legacy_loader(tmp_path: Path) -> None:
    inputs = _bridge_tree(tmp_path)
    initial = ("project-src", "installed-package")
    current = [*initial, str(inputs.firstlight_root.resolve())]

    assert not _external_import_paths_rolled_back(initial, current, inputs)
    assert _restore_sys_path(initial, current)
    assert tuple(current) == initial
    assert _external_import_paths_rolled_back(initial, current, inputs)


def test_namespace_package_import_path_is_reportable(tmp_path: Path) -> None:
    namespace = SimpleNamespace(__file__=None, __path__=[str(tmp_path / "installed" / "elixir_echo")])

    assert _package_import_paths(namespace) == ((tmp_path / "installed" / "elixir_echo").resolve(),)
