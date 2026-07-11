"""
Orchestration with Prefect.

Wraps each pipeline stage as a Prefect task and composes them into flows:
  - training_flow   : ingest -> train -> register -> export
  - monitoring_flow : check drift -> (only if drift) run training_flow

Tasks use cache_policy=NO_CACHE because they have side effects (writing files,
training models) and must run every time — Prefect 3 caches by default.
"""

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from src.data_ingestion import run_data_ingestion
from src.export_model import export_champion_model
from src.monitoring import check_drift
from src.register_model import register_best_model
from src.train import train


@task(log_prints=True, cache_policy=NO_CACHE)
def ingest_task():
    run_data_ingestion()


@task(log_prints=True, cache_policy=NO_CACHE)
def train_task():
    train()


@task(log_prints=True, cache_policy=NO_CACHE)
def register_task():
    register_best_model()


@task(log_prints=True, cache_policy=NO_CACHE)
def export_task():
    export_champion_model()


@task(log_prints=True, cache_policy=NO_CACHE)
def drift_task(current_path=None) -> bool:
    return check_drift(current_path)


@flow(name="churn-training-pipeline", log_prints=True)
def training_flow():
    """Full retrain: ingest -> train -> register -> export."""
    ingest_task()
    train_task()
    register_task()
    export_task()


@flow(name="churn-monitoring-pipeline", log_prints=True)
def monitoring_flow(current_path: str | None = None):
    """Check for drift; retrain ONLY if drift is detected."""
    drift_detected = drift_task(current_path)
    if drift_detected:
        print("🚨 Drift detected -> triggering retraining")
        training_flow()
    else:
        print("✅ No drift -> model healthy, skipping retraining")


if __name__ == "__main__":
    monitoring_flow.serve(
        name="churn-monitoring-deployment",
        cron="0 2 * * *",   # every day at 2:00 AM
    )