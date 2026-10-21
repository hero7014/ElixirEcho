"""Explicit, bytecode-free imports from a user-supplied FirstLight checkout."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from types import ModuleType


FIRSTLIGHT_ROOT_ENV = "ELIXIR_ECHO_FIRSTLIGHT_ROOT"


class FirstLightConfigurationError(RuntimeError):
    """Raised when the external FirstLight checkout is absent or inconsistent."""


class FirstLightRuntime:
    """Resolve and import FirstLight modules without writing into its checkout."""

    def __init__(self, root: str | Path) -> None:
        candidate = Path(root).expanduser().resolve()
        if not (candidate / "native_runner" / "__init__.py").is_file():
            raise FirstLightConfigurationError(
                f"{candidate} is not a FirstLight checkout: native_runner/__init__.py is missing"
            )
        self.root = candidate

    @classmethod
    def from_environment(cls) -> "FirstLightRuntime":
        raw = os.environ.get(FIRSTLIGHT_ROOT_ENV)
        if not raw:
            raise FirstLightConfigurationError(
                f"set {FIRSTLIGHT_ROOT_ENV} to the local FirstLight checkout"
            )
        return cls(raw)

    def import_module(self, name: str) -> ModuleType:
        if name != "native_runner" and not name.startswith("native_runner."):
            raise FirstLightConfigurationError(
                "FirstLight imports must stay inside the native_runner package"
            )

        existing = sys.modules.get(name)
        if existing is not None:
            self._assert_module_location(existing)
            return existing

        root_text = str(self.root)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        # Environment variables are only interpreted at interpreter startup.
        # Keep the runtime flag enabled so later imports from this external
        # package cannot create bytecode in the configured checkout either.
        sys.dont_write_bytecode = True
        sys.path.insert(0, root_text)
        try:
            module = importlib.import_module(name)
        finally:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass
        self._assert_module_location(module)
        return module

    def _assert_module_location(self, module: ModuleType) -> None:
        raw_file = getattr(module, "__file__", None)
        if raw_file is None:
            raise FirstLightConfigurationError("imported FirstLight module has no file")
        module_file = Path(raw_file).resolve()
        if not module_file.is_relative_to(self.root):
            raise FirstLightConfigurationError(
                f"module resolved outside configured FirstLight checkout: {module_file}"
            )
