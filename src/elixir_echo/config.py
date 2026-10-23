"""Portable, precedence-aware configuration for ElixirEcho commands."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when project configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Resolved settings; every configured path is absolute and user-owned."""

    config_path: Path | None
    data_root: Path
    firstlight_root: Path | None
    dataset_path: Path
    output_path: Path
    mumu_manager_path: Path | None
    game_assets_path: Path | None
    adb_executable: str
    device_serial: str | None
    compute_device: str
    max_episodes: int
    checkpoint_every: int


_ENVIRONMENT_NAMES = {
    "data_root": "ELIXIR_ECHO_DATA_ROOT",
    "firstlight_root": "ELIXIR_ECHO_FIRSTLIGHT_ROOT",
    "dataset_path": "ELIXIR_ECHO_DATASET_PATH",
    "output_path": "ELIXIR_ECHO_OUTPUT_PATH",
    "mumu_manager_path": "ELIXIR_ECHO_MUMU_MANAGER_PATH",
    "game_assets_path": "ELIXIR_ECHO_GAME_ASSETS_PATH",
    "adb_executable": "ELIXIR_ECHO_ADB_EXECUTABLE",
    "device_serial": "ELIXIR_ECHO_DEVICE_SERIAL",
    "compute_device": "ELIXIR_ECHO_DEVICE",
    "max_episodes": "ELIXIR_ECHO_MAX_EPISODES",
    "checkpoint_every": "ELIXIR_ECHO_CHECKPOINT_EVERY",
}

_TOML_LOCATIONS = {
    "data_root": ("paths", "data_root"),
    "firstlight_root": ("paths", "firstlight_root"),
    "dataset_path": ("paths", "dataset_path"),
    "output_path": ("paths", "output_path"),
    "mumu_manager_path": ("paths", "mumu_manager_path"),
    "game_assets_path": ("paths", "game_assets_path"),
    "adb_executable": ("paths", "adb_executable"),
    "device_serial": ("paths", "device_serial"),
    "compute_device": ("runtime", "device"),
    "max_episodes": ("training", "max_episodes"),
    "checkpoint_every": ("training", "checkpoint_every"),
}

_OPTIONAL_PATHS = frozenset({"firstlight_root", "mumu_manager_path", "game_assets_path"})
_OPTIONAL_STRINGS = frozenset({"device_serial"})
_PATH_FIELDS = frozenset(
    {"data_root", "firstlight_root", "dataset_path", "output_path", "mumu_manager_path", "game_assets_path"}
)
_DEFAULTS: dict[str, Any] = {
    "data_root": Path("data"),
    "firstlight_root": None,
    "dataset_path": Path("data/full_episode_cache"),
    "output_path": Path("runs"),
    "mumu_manager_path": None,
    "game_assets_path": None,
    "adb_executable": "adb",
    "device_serial": None,
    "compute_device": "cpu",
    "max_episodes": 1,
    "checkpoint_every": 10,
}


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"configuration section [{name}] must be a table")
    return value


def _read_toml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML configuration at {path}: {error}") from error
    except OSError as error:
        raise ConfigError(f"cannot read configuration file {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ConfigError("configuration root must be a TOML table")
    return payload


def _path_value(name: str, value: Any, *, base: Path) -> Path | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if name in _OPTIONAL_PATHS:
            return None
        raise ConfigError(f"{name} must be a non-empty path")
    if not isinstance(value, (str, Path)):
        location = _TOML_LOCATIONS.get(name)
        key = f"{location[0]}.{location[1]}" if location else name
        raise ConfigError(f"{key} must be a string path")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _string_value(name: str, value: Any) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if name in _OPTIONAL_STRINGS:
            return None
        raise ConfigError(f"{name} must be a non-empty string")
    if not isinstance(value, str):
        location = _TOML_LOCATIONS.get(name)
        key = f"{location[0]}.{location[1]}" if location else name
        raise ConfigError(f"{key} must be a string")
    return value.strip()


def _resolve_adb(value: Any, *, base: Path, source: str) -> str:
    command = _string_value("adb_executable", value)
    assert command is not None
    if source == "toml":
        candidate = Path(command).expanduser()
        if candidate.is_absolute() or "/" in command or "\\" in command:
            return str((candidate if candidate.is_absolute() else base / candidate).resolve())
    return command


def _positive_integer(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _environment_integer(name: str, value: str) -> int:
    try:
        parsed = int(value.strip(), 10)
    except (AttributeError, ValueError) as error:
        raise ConfigError(f"{_ENVIRONMENT_NAMES[name]} must be a positive integer") from error
    return _positive_integer(name, parsed)


def load_config(
    *,
    config_path: str | Path | None = None,
    cwd: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ProjectConfig:
    """Resolve CLI overrides > environment > TOML > relative defaults.

    TOML-relative paths are anchored at the configuration file's directory.
    CLI and environment paths are anchored at the current working directory.
    If no explicit config is provided, ``configs/elixir_echo_baseline.toml``
    under the current working directory is optional.
    """

    current_dir = Path(cwd or Path.cwd()).expanduser().resolve()
    env = os.environ if environ is None else environ
    cli_values = dict(overrides or {})
    unknown = set(cli_values) - set(_DEFAULTS)
    if unknown:
        raise ConfigError(f"unsupported CLI configuration override: {sorted(unknown)[0]}")

    raw_config = config_path
    explicit_config = raw_config is not None
    if raw_config is None:
        raw_config = env.get("ELIXIR_ECHO_CONFIG")
        explicit_config = raw_config is not None
    config_file: Path | None = None
    toml_values: Mapping[str, Any] = {}
    if raw_config is not None:
        config_file = Path(raw_config).expanduser()
        if not config_file.is_absolute():
            config_file = current_dir / config_file
        config_file = config_file.resolve()
    else:
        default_file = current_dir / "configs" / "elixir_echo_baseline.toml"
        if default_file.is_file():
            config_file = default_file.resolve()
    if config_file is not None:
        if not config_file.is_file():
            if explicit_config:
                raise ConfigError(f"configuration file does not exist: {config_file}")
            config_file = None
        else:
            toml_values = _read_toml(config_file)

    sections = {name: _section(toml_values, name) for name in ("paths", "runtime", "training")}
    toml_base = config_file.parent if config_file is not None else current_dir
    resolved: dict[str, Any] = {}
    for name, default in _DEFAULTS.items():
        if name in cli_values and cli_values[name] is not None:
            raw_value, source, base = cli_values[name], "cli", current_dir
        elif _ENVIRONMENT_NAMES[name] in env:
            raw_value, source, base = env[_ENVIRONMENT_NAMES[name]], "environment", current_dir
        else:
            section_name, key = _TOML_LOCATIONS[name]
            if key in sections[section_name]:
                raw_value, source, base = sections[section_name][key], "toml", toml_base
            else:
                raw_value, source, base = default, "default", current_dir

        if name in _PATH_FIELDS:
            resolved[name] = _path_value(name, raw_value, base=base)
        elif name in {"max_episodes", "checkpoint_every"} and source == "environment":
            resolved[name] = _environment_integer(name, raw_value)
        elif name == "adb_executable":
            resolved[name] = _resolve_adb(raw_value, base=base, source=source)
        elif name in _OPTIONAL_STRINGS:
            resolved[name] = _string_value(name, raw_value)
        elif name == "compute_device":
            resolved[name] = _string_value(name, raw_value)
        else:
            resolved[name] = _positive_integer(name, raw_value)

    return ProjectConfig(config_path=config_file, **resolved)
