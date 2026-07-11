"""Tests for the evaluation metrics."""

from src.evaluation import evaluate_classification


def test_all_metrics_present_and_in_range():
    metrics = evaluate_classification([0, 0, 1, 1], [0, 1, 1, 1], [0.2, 0.6, 0.8, 0.9])
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_perfect_predictions_score_one():
    metrics = evaluate_classification([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
