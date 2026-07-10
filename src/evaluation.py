"""
Evaluation module.

Computes the classification metrics that matter for IMBALANCED churn data.
Returns a plain dict so the caller (training, monitoring) can log or compare them.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_classification(y_true, y_pred, y_proba) -> dict:
    """
    Args:
        y_true:  true labels (0/1)
        y_pred:  predicted labels (0/1)
        y_proba: predicted probability of the positive class (churn=1)

    Returns:
        dict of metric_name -> value
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
