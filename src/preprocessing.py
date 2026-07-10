"""
Preprocessing module.

Turns raw churn data into model-ready features, encoding every decision from
our EDA Cleaning Log:
  - TotalCharges : text -> numeric, blanks (tenure=0) -> 0
  - customerID   : dropped (identifier, no predictive signal)
  - Churn target : Yes/No -> 1/0
  - numeric features : scaled | categorical features : one-hot encoded

Built as a scikit-learn ColumnTransformer so preprocessing travels WITH the
model as one artifact (prevents train/serve skew).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import load_config
from src.logger import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """
    Stateless cleaning applied to ANY raw input (train, test, or live request).
    Fixes the TotalCharges trap and drops the ID column.
    """
    df = df.copy()  # never mutate the caller's DataFrame

    # Fix the TotalCharges trap: convert text -> numeric; blanks become NaN,
    # then fill those NaNs with 0 (tenure=0 -> no charges accrued yet).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Drop the identifier column (no predictive value). Guard with 'if' so the
    # function is safe to call even on data that has already had it removed.
    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    return df


def separate_features_target(df: pd.DataFrame, target: str) -> tuple:
    """Split a cleaned DataFrame into X (features) and y (target encoded 1/0)."""
    X = df.drop(columns=[target])
    y = (df[target] == "Yes").astype(int)  # Yes -> 1, No -> 0
    return X, y


def get_feature_lists(X: pd.DataFrame, numeric_features: list) -> tuple:
    """Numeric features come from config; everything else is categorical."""
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Build the preprocessing transformer:
      - numeric columns  -> StandardScaler (zero mean, unit variance)
      - categorical cols -> OneHotEncoder (handle_unknown='ignore')
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Self-test: python -m src.preprocessing
# Demonstrates the correct FIT-ON-TRAIN, TRANSFORM-BOTH workflow (no leakage).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config()
    target = config["columns"]["target"]
    id_col = config["columns"]["id_column"]
    numeric = config["features"]["numeric"]
    processed_dir = config["data"]["processed_dir"]

    # Load the splits produced by data_ingestion.
    train_df = pd.read_csv(f"{processed_dir}/train.csv")
    test_df = pd.read_csv(f"{processed_dir}/test.csv")

    # Clean both sets with the SAME stateless function.
    train_df = clean_data(train_df, id_col)
    test_df = clean_data(test_df, id_col)

    # Separate features/target.
    X_train, y_train = separate_features_target(train_df, target)
    X_test, y_test = separate_features_target(test_df, target)

    # Determine feature lists and build the preprocessor.
    numeric_features, categorical_features = get_feature_lists(X_train, numeric)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # THE KEY STEP: fit ONLY on training data, then transform BOTH sets.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    logger.info(f"X_train processed shape: {X_train_processed.shape}")
    logger.info(f"X_test  processed shape: {X_test_processed.shape}")
    logger.info(f"y_train churn ratio: {y_train.mean():.3f} | y_test: {y_test.mean():.3f}")
    logger.info("Preprocessing self-test passed ✅")
