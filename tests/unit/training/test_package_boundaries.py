from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_core_training_import_does_not_load_firstlight_integration():
    repository = Path(__file__).resolve().parents[3]
    source = repository / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(source), env.get("PYTHONPATH", "")) if item
    )

    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import elixir_echo.training; "
            "assert not any(name.startswith('elixir_echo.integrations.firstlight') "
            "for name in sys.modules)",
        ],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
