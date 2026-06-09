"""Tests for utility functions."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from churnguard.utils import (
    DEFAULT_CONFIG,
    ensure_dir,
    get_config,
    load_config,
    merge_configs,
    truncate_dict,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_yaml(self, tmp_path: Path):
        """Test loading a valid YAML config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"key": "value", "nested": {"a": 1}}))
        config = load_config(config_path)
        assert config["key"] == "value"
        assert config["nested"]["a"] == 1

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_empty_yaml(self, tmp_path: Path):
        """Test loading an empty YAML file."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        config = load_config(config_path)
        assert config == {}


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_simple_merge(self):
        """Test merging two flat configs."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        """Test deep merging of nested configs."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 5, "z": 6}}
        result = merge_configs(base, override)
        assert result == {"a": {"x": 1, "y": 5, "z": 6}, "b": 3}

    def test_override_replaces_non_dict(self):
        """Test that non-dict values are replaced, not merged."""
        base = {"a": [1, 2, 3]}
        override = {"a": [4, 5]}
        result = merge_configs(base, override)
        assert result["a"] == [4, 5]


class TestGetConfig:
    """Tests for get_config function."""

    def test_default_config(self):
        """Test getting default config without file."""
        config = get_config()
        assert "features" in config
        assert "models" in config
        assert "evaluation" in config

    def test_config_with_override(self, tmp_path: Path):
        """Test config with user override file."""
        config_path = tmp_path / "override.yaml"
        config_path.write_text(yaml.dump({"features": {"scaling": "minmax"}}))
        config = get_config(config_path)
        assert config["features"]["scaling"] == "minmax"
        # Other defaults should still be present
        assert config["features"]["numeric_impute_strategy"] == "median"

    def test_config_nonexistent_file(self):
        """Test config with nonexistent override falls back to defaults."""
        config = get_config("/nonexistent/config.yaml")
        assert config == DEFAULT_CONFIG.copy()


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_create_new_dir(self, tmp_path: Path):
        """Test creating a new directory."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        result = ensure_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_existing_dir(self, tmp_path: Path):
        """Test that existing directory is returned."""
        result = ensure_dir(tmp_path)
        assert result == tmp_path


class TestTruncateDict:
    """Tests for truncate_dict function."""

    def test_short_dict_unchanged(self):
        """Test that short dicts are returned unchanged."""
        d = {"a": 1, "b": 2, "c": 3}
        result = truncate_dict(d, max_items=5)
        assert result == d

    def test_long_dict_truncated(self):
        """Test that long dicts are truncated."""
        d = {f"key_{i}": i for i in range(20)}
        result = truncate_dict(d, max_items=5)
        assert len(result) == 5
        # Should keep highest values
        assert max(result.values()) == 19

    def test_empty_dict(self):
        """Test with empty dict."""
        assert truncate_dict({}, max_items=5) == {}
