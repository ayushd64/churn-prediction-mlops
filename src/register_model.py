"""
Model registration module.

Finds the best run in the experiment (by the primary metric), registers its
model into the MLflow Model Registry as a new version, and assigns the
'champion' alias to that version so downstream services can always load
the current production model via  models:/<name>@champion.
"""

import mlflow
from mlflow.tracking import MlflowClient

from src.config import load_config
from src.logger import get_logger

logger = get_logger(__name__)


def find_best_run(client: MlflowClient, experiment_name: str, metric: str):
    """Return the run with the highest value of `metric` in the experiment."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. Train first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No runs found to register.")
    return runs[0]


def register_best_model() -> None:
    """Register the best run's model and promote it to 'champion'."""
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    experiment_name = config["mlflow"]["experiment_name"]
    model_name = config["registry"]["model_name"]
    metric = config["registry"]["primary_metric"]
    champion_alias = config["registry"]["champion_alias"]

    client = MlflowClient()

    # 1) Find the winning run.
    best_run = find_best_run(client, experiment_name, metric)
    run_id = best_run.info.run_id
    run_name = best_run.data.tags.get("mlflow.runName", "unknown")
    score = best_run.data.metrics.get(metric)
    logger.info(f"Best run: '{run_name}' | {metric}={score:.4f} | run_id={run_id}")

    # 2) Register that run's model as a new version in the registry.
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(f"Registered '{model_name}' as version {result.version}")

    # 3) Point the 'champion' alias at this new version.
    client.set_registered_model_alias(model_name, champion_alias, result.version)
    logger.info(f"Alias '@{champion_alias}' -> '{model_name}' v{result.version}")

    # 4) Tag the version with useful metadata (nice for auditing).
    client.set_model_version_tag(model_name, result.version, "source_run_name", run_name)
    client.set_model_version_tag(model_name, result.version, metric, f"{score:.4f}")


if __name__ == "__main__":
    register_best_model()
    