from __future__ import annotations

from pathlib import Path

import pytest

from elixir_echo.config import ConfigError, load_config


def test_config_precedence_is_cli_then_environment_then_toml_then_defaults(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "settings"
    config_dir.mkdir()
    config_path = config_dir / "baseline.toml"
    config_path.write_text(
        '[paths]\ndata_root = "toml-data"\nfirstlight_root = "toml-provider"\n'
        '[runtime]\ndevice = "cpu"\n',
        encoding="utf-8",
    )

    config = load_config(
        config_path=config_path,
        cwd=tmp_path,
        environ={
            "ELIXIR_ECHO_DATA_ROOT": "environment-data",
            "ELIXIR_ECHO_FIRSTLIGHT_ROOT": "environment-provider",
        },
        overrides={"data_root": "cli-data"},
    )

    assert config.data_root == (tmp_path / "cli-data").resolve()
    assert config.firstlight_root == (tmp_path / "environment-provider").resolve()
    assert config.dataset_path == (tmp_path / "data" / "full_episode_cache").resolve()
    assert config.output_path == (tmp_path / "runs").resolve()
    assert config.compute_device == "cpu"


def test_toml_relative_paths_resolve_from_config_file_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "portable-project" / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "elixir_echo_baseline.toml"
    config_path.write_text(
        '[paths]\ndata_root = "../data"\ndataset_path = "../data/cache"\n'
        'output_path = "../runs"\n',
        encoding="utf-8",
    )

    config = load_config(config_path=config_path, cwd=tmp_path, environ={})

    assert config.data_root == (config_dir / "../data").resolve()
    assert config.dataset_path == (config_dir / "../data/cache").resolve()
    assert config.output_path == (config_dir / "../runs").resolve()


def test_config_rejects_missing_explicit_file_and_invalid_settings(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="configuration file does not exist"):
        load_config(config_path=tmp_path / "missing.toml", cwd=tmp_path, environ={})

    invalid = tmp_path / "invalid.toml"
    invalid.write_text('[training]\nmax_episodes = 0\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="max_episodes"):
        load_config(config_path=invalid, cwd=tmp_path, environ={})


def test_invalid_toml_path_type_is_not_silently_coerced(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-path.toml"
    config_path.write_text("[paths]\ndata_root = 42\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="paths.data_root must be a string"):
        load_config(config_path=config_path, cwd=tmp_path, environ={})


def test_numeric_environment_overrides_are_parsed_and_validated(tmp_path: Path) -> None:
    config = load_config(
        cwd=tmp_path,
        environ={"ELIXIR_ECHO_MAX_EPISODES": "12", "ELIXIR_ECHO_CHECKPOINT_EVERY": "3"},
    )
    assert config.max_episodes == 12
    assert config.checkpoint_every == 3

    with pytest.raises(ConfigError, match="ELIXIR_ECHO_MAX_EPISODES"):
        load_config(cwd=tmp_path, environ={"ELIXIR_ECHO_MAX_EPISODES": "not-a-number"})


def test_checked_in_baseline_paths_resolve_from_project_root(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    baseline = repository_root / "configs" / "elixir_echo_baseline.toml"
    config = load_config(config_path=baseline, cwd=tmp_path, environ={})
    assert config.data_root == (repository_root / "data").resolve()
    assert config.dataset_path == (repository_root / "data" / "full_episode_cache").resolve()
    assert config.output_path == (repository_root / "runs").resolve()
