"""
Training module.

Trains one or more models (defined in config), logging each as its own MLflow
run so they can be compared side by side. Each run records params, metrics,
and the trained pipeline (preprocessing + model) as one artifact.
"""

import shutil
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import load_config
from src.evaluation import evaluate_classification
from src.logger import get_logger
from src.preprocessing import (
    build_preprocessor,
    clean_data,
    get_feature_lists,
    separate_features_target,
)

logger = get_logger(__name__)

# Factory: maps a config "type" string to the actual classifier class.
MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}


def gpu_available() -> bool:
    """Return True only if an NVIDIA GPU is visible (via nvidia-smi).

    XGBoost silently downgrades to CPU when no GPU is present, so we detect
    the GPU ourselves to (a) log it clearly and (b) set device='cpu'
    explicitly on GPU-less machines (e.g. a CI server), avoiding noisy warnings.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def create_classifier(model_type: str, params: dict):
    """Instantiate a classifier by name using params from config."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    params = dict(params)  # copy so we never mutate the config dict

    # For XGBoost, gracefully fall back to CPU if no usable GPU is present.
    if model_type == "XGBClassifier" and params.get("device") == "cuda":
        if gpu_available():
            logger.info("CUDA GPU detected — training XGBoost on GPU ⚡")
        else:
            logger.warning("No usable CUDA GPU — falling back to device='cpu' for XGBoost")
            params["device"] = "cpu"

    return MODEL_REGISTRY[model_type](**params)


def load_processed_data(processed_dir: str, id_column: str, target: str) -> tuple:
    """Load the train/test CSVs, clean them, and split into X/y."""
    train_df = clean_data(pd.read_csv(f"{processed_dir}/train.csv"), id_column)
    test_df = clean_data(pd.read_csv(f"{processed_dir}/test.csv"), id_column)
    X_train, y_train = separate_features_target(train_df, target)
    X_test, y_test = separate_features_target(test_df, target)
    logger.info(f"Loaded data | X_train={X_train.shape} | X_test={X_test.shape}")
    return X_train, y_train, X_test, y_test


def build_model(model_type: str, model_params: dict, numeric_features: list,
                categorical_features: list) -> Pipeline:
    """Assemble the end-to-end pipeline: preprocessing + classifier."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    classifier = create_classifier(model_type, model_params)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_single_model(model_name, model_cfg, data, feature_lists) -> dict:
    """Train ONE model inside its own MLflow run; return its metrics."""
    X_train, y_train, X_test, y_test = data
    numeric_features, categorical_features = feature_lists
    model_type = model_cfg["type"]
    model_params = model_cfg["params"]

    with mlflow.start_run(run_name=model_name) as run:
        logger.info(f"[{model_name}] run {run.info.run_id}")
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        model = build_model(model_type, model_params, numeric_features, categorical_features)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_classification(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_test.head(2),
            serialization_format="cloudpickle",  # handles XGBoost; skops default rejects it
        )

        logger.info(f"[{model_name}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def train() -> None:
    """Train every model in config and report a comparison."""
    config = load_config()
    data = load_processed_data(
        config["data"]["processed_dir"],
        config["columns"]["id_column"],
        config["columns"]["target"],
    )
    feature_lists = get_feature_lists(data[0], config["features"]["numeric"])

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    results = {}
    for model_name, model_cfg in config["models"].items():
        results[model_name] = train_single_model(model_name, model_cfg, data, feature_lists)

    logger.info("===== Model comparison (by ROC-AUC) =====")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True):
        logger.info(f"  {name:22s} roc_auc={m['roc_auc']:.4f}  f1={m['f1']:.4f}")


if __name__ == "__main__":
    train()
