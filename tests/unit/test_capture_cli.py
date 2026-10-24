from __future__ import annotations

import json
from pathlib import Path

from elixir_echo import cli
from elixir_echo.integrations.firstlight import capture


def test_explicit_capture_dispatches_one_fixed_replay_without_native_in_test(tmp_path, monkeypatch, capsys):
    source = tmp_path / "IL_Replay"
    (source / "replays").mkdir(parents=True)
    firstlight = tmp_path / "FirstLight"
    (firstlight / "native_runner").mkdir(parents=True)
    (firstlight / "native_runner" / "__init__.py").write_text("", encoding="utf-8")
    output = tmp_path / "trajectories"
    captured: dict = {}

    monkeypatch.setattr(capture, "load_replay_payload", lambda path, tag: {"replay_tag": tag})

    def fake_capture(payload, **kwargs):
        captured.update(payload=payload, **kwargs)
        return capture.CaptureResult("fixed", "fixed", output / "episodes" / "fi" / "fixed", "terminated", 3, 6, 6)

    monkeypatch.setattr(capture, "capture_replay", fake_capture)
    code = cli.main([
        "capture", "--replay-dataset", str(source), "--replay-tag", "fixed",
        "--firstlight-root", str(firstlight), "--output", str(output), "--run",
    ])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["mode"] == "captured"
    assert captured["payload"]["replay_tag"] == "fixed"
    assert captured["firstlight_root"] == firstlight
    assert captured["output_root"] == output
    assert not output.exists()


def test_capture_preflight_rejects_output_elsewhere_inside_legacy_project(tmp_path: Path, capsys):
    source = tmp_path / "legacy" / "data" / "raw" / "IL_Replay"
    (source / "replays").mkdir(parents=True)
    legacy_capture = tmp_path / "legacy" / "src" / "semantic_frame" / "firstlight_full_episode.py"
    legacy_capture.parent.mkdir(parents=True)
    legacy_capture.write_text("# earlier capture\n", encoding="utf-8")
    firstlight = tmp_path / "provider"
    (firstlight / "native_runner").mkdir(parents=True)
    (firstlight / "native_runner" / "__init__.py").write_text("", encoding="utf-8")
    code = cli.main([
        "capture", "--replay-dataset", str(source), "--replay-tag", "fixed",
        "--firstlight-root", str(firstlight),
        "--output", str(tmp_path / "legacy" / "data" / "training"),
    ])
    assert code == 2
    assert "outside read-only inputs" in capsys.readouterr().err
