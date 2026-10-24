from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from release_audit import audit_repository  # noqa: E402


def test_forced_git_add_of_private_artifact_is_rejected(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("data/\n", encoding="utf-8")
    replay = tmp_path / "data" / "replay.parquet"
    replay.parent.mkdir()
    replay.write_bytes(b"private replay")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", "data/replay.parquet"], check=True)

    report = audit_repository(tmp_path)
    assert report["status"] == "failed"
    assert {item["reason"] for item in report["findings"]} == {
        "private_artifact_directory", "private_or_binary_artifact"
    }


def test_current_checkout_has_no_prospective_binary_or_cache_payload():
    report = audit_repository(PROJECT_ROOT)
    assert report["status"] == "passed", report["findings"]
