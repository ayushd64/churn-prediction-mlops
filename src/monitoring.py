"""
Monitoring module.

Detects data drift between the training data (reference) and a new batch of
incoming data (current) using Evidently. If the share of drifted columns
exceeds a configured threshold, drift is 'detected' — the signal to retrain.
"""

import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import load_config
from src.logger import get_logger
from src.preprocessing import clean_data

logger = get_logger(__name__)


def _feature_frame(df: pd.DataFrame, target: str, id_column: str) -> pd.DataFrame:
    """Clean the data and drop the target, so we compare only input features."""
    df = clean_data(df, id_column)
    return df.drop(columns=[target], errors="ignore")


def detect_drift(reference_df, current_df, threshold):
    """Run Evidently drift detection. Returns (detected, share, count, snapshot)."""
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_df, reference_df)   # (current, reference)

    share, count = 0.0, 0
    for metric in snapshot.dict()["metrics"]:
        if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
            share = metric["value"]["share"]
            count = int(metric["value"]["count"])
            break

    return share >= threshold, share, count, snapshot


def check_drift(current_path: str | None = None) -> bool:
    """Compare training data vs a current batch; save an HTML report; return drift flag."""
    config = load_config()
    target = config["columns"]["target"]
    id_col = config["columns"]["id_column"]
    threshold = config["monitoring"]["drift_threshold"]
    processed_dir = config["data"]["processed_dir"]

    reference_df = _feature_frame(pd.read_csv(f"{processed_dir}/train.csv"), target, id_col)

    current_source = current_path or f"{processed_dir}/test.csv"
    current_df = _feature_frame(pd.read_csv(current_source), target, id_col)

    detected, share, count, snapshot = detect_drift(reference_df, current_df, threshold)

    report_path = Path(config["monitoring"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_path))

    logger.info(f"Current batch: '{current_source}'")
    logger.info(f"Drifted columns: {count}/{len(reference_df.columns)} | share={share:.3f}")
    logger.info(f"Threshold={threshold} -> DRIFT DETECTED={detected}")
    logger.info(f"Report saved to '{report_path}'")
    return detected


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    check_drift(path)
