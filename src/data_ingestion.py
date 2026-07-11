"""
Data ingestion module.

Loads the raw churn dataset, performs a reproducible, STRATIFIED
train/test split, and saves the resulting sets to data/processed/.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """Load the raw CSV file into a DataFrame."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw data from '{raw_path}' | shape={df.shape}")
    return df


def split_data(df: pd.DataFrame, target: str, test_size: float,
               random_state: int) -> tuple:
    """Split into train/test, preserving the target's class ratio (stratified)."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )
    logger.info(f"Split complete | train={train_df.shape} | test={test_df.shape}")

    # Log the churn ratio in each split to PROVE stratification worked.
    train_ratio = train_df[target].value_counts(normalize=True).round(3).to_dict()
    test_ratio = test_df[target].value_counts(normalize=True).round(3).to_dict()
    logger.info(f"Train target ratio: {train_ratio}")
    logger.info(f"Test  target ratio: {test_ratio}")

    return train_df, test_df


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame,
              processed_dir: str) -> None:
    """Save train and test sets as CSV files in the processed directory."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train set to '{train_path}'")
    logger.info(f"Saved test set to '{test_path}'")


def run_data_ingestion(config_path: str = "configs/config.yaml") -> None:
    """Orchestrate the full data ingestion step."""
    logger.info("===== Starting data ingestion =====")
    config = load_config(config_path)

    df = load_raw_data(config["data"]["raw_path"])

    train_df, test_df = split_data(
        df=df,
        target=config["columns"]["target"],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    save_data(train_df, test_df, config["data"]["processed_dir"])
    logger.info("===== Data ingestion complete =====")


if __name__ == "__main__":
    run_data_ingestion()
