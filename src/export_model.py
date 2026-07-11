"""
Model export module.

Loads the current champion model from the MLflow Registry and saves it to a
local, self-contained folder. This exported folder is what gets baked into the
Docker image, so the container needs no MLflow backend at runtime.
"""

import shutil
from pathlib import Path

import mlflow

from src.config import load_config
from src.logger import get_logger

logger = get_logger(__name__)


def export_champion_model() -> None:
    """Export models:/<name>@champion to a local directory."""
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    model_uri = f"models:/{config['registry']['model_name']}@{config['registry']['champion_alias']}"
    dest = Path(config["serving"]["model_path"])

    logger.info(f"Loading champion from registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # Clear any previous export so save_model starts clean (it errors on existing dirs).
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    mlflow.sklearn.save_model(model, path=str(dest), serialization_format="cloudpickle")
    logger.info(f"Champion model exported to '{dest}' (self-contained)")


if __name__ == "__main__":
    export_champion_model()
