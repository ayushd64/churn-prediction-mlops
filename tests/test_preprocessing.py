"""Tests for the preprocessing functions (self-contained, no data files needed)."""

import pandas as pd

from src.preprocessing import (
    build_preprocessor,
    clean_data,
    get_feature_lists,
    separate_features_target,
)


def _sample_df():
    """A tiny DataFrame that reproduces the TotalCharges blank-string trap."""
    return pd.DataFrame({
        "customerID": ["A1", "A2", "A3"],
        "tenure": [0, 12, 24],
        "MonthlyCharges": [50.0, 70.0, 90.0],
        "TotalCharges": [" ", "840.5", "2160.0"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Churn": ["No", "Yes", "No"],
    })


def test_clean_data_fixes_totalcharges():
    df = clean_data(_sample_df(), id_column="customerID")
    assert df["TotalCharges"].dtype != object
    assert df["TotalCharges"].iloc[0] == 0.0


def test_clean_data_drops_id():
    df = clean_data(_sample_df(), id_column="customerID")
    assert "customerID" not in df.columns


def test_separate_features_target_encodes_target():
    df = clean_data(_sample_df(), id_column="customerID")
    X, y = separate_features_target(df, target="Churn")
    assert "Churn" not in X.columns
    assert y.tolist() == [0, 1, 0]


def test_get_feature_lists_splits_numeric_and_categorical():
    df = clean_data(_sample_df(), id_column="customerID")
    X, _ = separate_features_target(df, target="Churn")
    numeric, categorical = get_feature_lists(X, ["tenure", "MonthlyCharges", "TotalCharges"])
    assert numeric == ["tenure", "MonthlyCharges", "TotalCharges"]
    assert "Contract" in categorical


def test_preprocessor_produces_rows():
    df = clean_data(_sample_df(), id_column="customerID")
    X, _ = separate_features_target(df, target="Churn")
    numeric, categorical = get_feature_lists(X, ["tenure", "MonthlyCharges", "TotalCharges"])
    preprocessor = build_preprocessor(numeric, categorical)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 3
