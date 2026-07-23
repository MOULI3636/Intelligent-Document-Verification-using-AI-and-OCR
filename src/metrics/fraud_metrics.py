"""
Fraud Detection Research Evaluation Metrics.

Computes Precision, Recall, F1-Score, False Positive Rate (FPR),
and Area Under the ROC Curve (ROC-AUC) for document tampering classifier benchmark runs.
"""

from typing import Dict, List, Any
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("FraudMetrics")


def compute_fraud_metrics(
    predicted_scores: List[float],
    ground_truth_labels: List[int],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculates classification metrics for document forgery detection.

    Args:
        predicted_scores (List[float]): Fraud probability scores [0.0 - 1.0].
        ground_truth_labels (List[int]): Binary ground truth (1 for fraudulent, 0 for authentic).
        threshold (float): Decision classification boundary.

    Returns:
        Dict[str, float]: Precision, Recall, F1, Accuracy, and ROC-AUC.
    """
    preds = [1 if s >= threshold else 0 for s in predicted_scores]

    tp = sum(1 for p, g in zip(preds, ground_truth_labels) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, ground_truth_labels) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, ground_truth_labels) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(preds, ground_truth_labels) if p == 0 and g == 0)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-5, precision + recall)
    accuracy = (tp + tn) / max(1, len(ground_truth_labels))

    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(ground_truth_labels, predicted_scores))
    except Exception:
        roc_auc = 0.5

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": roc_auc
    }
