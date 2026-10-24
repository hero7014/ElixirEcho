from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clean_cli_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("ELIXIR_ECHO_"):
            environment.pop(key)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return environment


def test_fixture_runs_dependency_free_from_an_unrelated_working_directory(tmp_path: Path) -> None:
    working_directory = tmp_path / "clean-checkout-user"
    working_directory.mkdir()
    output = working_directory / "demo-output"
    result = subprocess.run(
        [sys.executable, "-m", "elixir_echo.cli", "fixture", "--output", str(output)],
        cwd=working_directory,
        env=_clean_cli_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    manifest_path = output / "fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["mode"] == "fixture"
    assert manifest["schema"] == "elixir_echo.fixture-run@1"
    assert manifest["source"] == "synthetic"
    assert manifest["supervised_steps"] == 1
    assert manifest["optimizer_updates"] == 1
    assert manifest["loss_after_one_update"] < manifest["initial_loss"]
    assert manifest["trajectory_manifest"]["record_count"] == 2
    checkpoint_manifest = json.loads((output / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert checkpoint_manifest["loadable"] is False
    assert checkpoint_manifest["source"] == "synthetic_fake_policy"
    assert (output / "trajectory" / "episodes").is_dir()
    assert "FirstLight" not in result.stdout


def test_doctor_reports_missing_optional_assets_with_actionable_setup(tmp_path: Path) -> None:
    working_directory = tmp_path / "clean-checkout-user"
    working_directory.mkdir()
    environment = _clean_cli_environment()
    result = subprocess.run(
        [sys.executable, "-m", "elixir_echo.cli", "doctor"],
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["fixture_ready"] is True
    assert report["native_ready"] is False
    diagnostics = "\n".join(item["action"] for item in report["checks"] if item["status"] != "ready")
    assert "ELIXIR_ECHO_FIRSTLIGHT_ROOT" in diagnostics
    assert "ELIXIR_ECHO_GAME_ASSETS_PATH" in diagnostics
    assert "ELIXIR_ECHO_DATASET_PATH" in diagnostics
    assert "ELIXIR_ECHO_MUMU_MANAGER_PATH" in diagnostics
    assert "ELIXIR_ECHO_ADB_EXECUTABLE" in diagnostics


def test_capture_preflight_reports_missing_runtime_without_starting_native(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "elixir_echo.cli", "capture"],
        cwd=tmp_path,
        env=_clean_cli_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "preflight_only"
    assert report["ready"] is False
    assert {item["name"] for item in report["checks"] if not item["ready"]} >= {
        "firstlight_root", "replay_dataset", "replay_tag"
    }
    assert not (tmp_path / "data").exists()


def test_capture_run_requires_an_explicit_source_identity(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "elixir_echo.cli", "capture", "--run"],
        cwd=tmp_path,
        env=_clean_cli_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "capture --run requires" in result.stderr
    assert "replay_tag" in result.stderr
