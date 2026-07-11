"""
Configuration loader.

Reads the project's YAML config file and returns it as a Python dictionary,
so every module gets its paths and parameters from ONE central place.
"""

from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load the YAML configuration file into a dictionary.

    Args:
        config_path: Path to the YAML config file (relative to project root).

    Returns:
        A dictionary containing all configuration settings.
    """
    config_path = Path(config_path)

    # Fail loudly and early if the config file is missing.
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


# Runs ONLY when this file is executed directly (python -m src.config),
# not when it's imported elsewhere. A handy built-in self-test.
if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully ✅")
    print(cfg)
