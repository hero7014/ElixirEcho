from __future__ import annotations

from pathlib import Path
import re
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROVENANCE_ALLOWLIST = Path("docs/provenance")


def _scan_project(root: Path) -> list[str]:
    forbidden_paths = (
        "D:" + "\\" + "FirstLight_CR",
        "D:" + "\\" + "Downloads",
        "C:" + "\\" + "Users" + "\\" + "feng",
        "." + "codex",
    )
    device_patterns = (
        re.compile(r"127\.0\.0\.1:\d{4,5}"),
        re.compile(r"(?i)\bemulator-\d+\b"),
    )
    secret_patterns = (
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        re.compile(r"(?i)(api[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"]?[^\s'\"]{16,}"),
    )
    violations: list[str] = []
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=False,
    )
    if listing.returncode == 0:
        candidates = (root / Path(item) for item in listing.stdout.decode("utf-8").split("\0") if item)
    else:
        candidates = root.rglob("*")
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if ".git" in relative.parts or _PROVENANCE_ALLOWLIST in relative.parents:
            continue
        try:
            raw = path.read_bytes()
            if b"\0" in raw:
                continue
            content = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(value.casefold() in content.casefold() for value in forbidden_paths):
            violations.append(relative.as_posix())
            continue
        if any(pattern.search(content) for pattern in (*device_patterns, *secret_patterns)):
            violations.append(relative.as_posix())

    return violations


def test_project_text_has_no_developer_paths_device_ids_or_secrets() -> None:
    violations = _scan_project(PROJECT_ROOT)
    assert not violations, f"machine-bound or secret-like text found: {sorted(violations)}"


def test_scanner_rejects_forbidden_path_and_secret_in_supported_text(
    tmp_path: Path,
) -> None:
    sensitive_path = "D:" + chr(92) + "FirstLight_CR"
    fake_secret = "gh" + "p_" + "x" * 36
    script = tmp_path / "sample.ps1"
    script.write_text(sensitive_path, encoding="utf-8")
    assert _scan_project(tmp_path) == ["sample.ps1"]
    extensionless = tmp_path / "notes"
    extensionless.write_text(fake_secret, encoding="utf-8")
    assert _scan_project(tmp_path) == ["notes", "sample.ps1"]


def test_scanner_includes_tracked_text_under_ignored_data_directory(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("data/\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
    data = tmp_path / "data"
    data.mkdir()
    forbidden = "D:" + chr(92) + "Downloads"
    file_path = data / "tracked.ini"
    file_path.write_text(forbidden, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", "data/tracked.ini"], check=True)

    assert _scan_project(tmp_path) == ["data/tracked.ini"]
