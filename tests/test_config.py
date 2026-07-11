"""Tests for the configuration loader."""

from src.config import load_config


def test_load_config_returns_dict():
    config = load_config()
    assert isinstance(config, dict)
    assert "data" in config
    assert "columns" in config


def test_config_target_is_churn():
    config = load_config()
    assert config["columns"]["target"] == "Churn"
