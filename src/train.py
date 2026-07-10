"""
Training module.

Assembles the full model Pipeline (preprocessor + classifier), trains it,
evaluates it on the test set, and logs EVERYTHING (params, metrics, and the
model itself with a signature) to MLflow Tracking.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.config import load_config
from src.logger import get_logger
from src.preprocessing import (
    clean_data,
    separate_features_target,
    get_feature_lists,
    build_preprocessor,
)
from src.evaluation import evaluate_classification

logger = get_logger(__name__)


def load_processed_data(processed_dir: str, id_column: str, target: str) -> tuple:
    """Load the train/test CSVs, clean them, and split into X/y."""
    train_df = clean_data(pd.read_csv(f"{processed_dir}/train.csv"), id_column)
    test_df = clean_data(pd.read_csv(f"{processed_dir}/test.csv"), id_column)

    X_train, y_train = separate_features_target(train_df, target)
    X_test, y_test = separate_features_target(test_df, target)

    logger.info(f"Loaded data | X_train={X_train.shape} | X_test={X_test.shape}")
    return X_train, y_train, X_test, y_test


def build_model(model_params: dict, numeric_features: list,
                categorical_features: list) -> Pipeline:
    """Assemble the end-to-end pipeline: preprocessing + classifier."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    classifier = LogisticRegression(**model_params)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    return model


def train() -> None:
    """Full training run with MLflow tracking."""
    config = load_config()

    target = config["columns"]["target"]
    id_col = config["columns"]["id_column"]
    numeric = config["features"]["numeric"]
    processed_dir = config["data"]["processed_dir"]
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]

    # --- Load data ---
    X_train, y_train, X_test, y_test = load_processed_data(processed_dir, id_col, target)
    numeric_features, categorical_features = get_feature_lists(X_train, numeric)

    # --- Point MLflow at our backend and experiment ---
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"{model_type}_baseline") as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")

        # 1) Log the inputs (params).
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        # 2) Build and train the model.
        model = build_model(model_params, numeric_features, categorical_features)
        model.fit(X_train, y_train)
        logger.info("Model training complete")

        # 3) Predict on the held-out test set.
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # 4) Evaluate and log the results (metrics).
        metrics = evaluate_classification(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        rounded = {k: round(v, 4) for k, v in metrics.items()}
        logger.info(f"Metrics: {rounded}")

        # 5) Log the trained model itself, with an input/output schema.
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_test.head(2),
        )
        logger.info("Model logged to MLflow")

    logger.info("===== Training run complete =====")


if __name__ == "__main__":
    train()
