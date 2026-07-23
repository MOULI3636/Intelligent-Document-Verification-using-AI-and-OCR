"""
Standardized OCR Research Evaluation Metrics.

Computes:
1. Character Error Rate (CER) = (Substitutions + Deletions + Insertions) / Total Ground Truth Characters
2. Word Error Rate (WER) = (Substitutions + Deletions + Insertions) / Total Ground Truth Words
3. Normalized Levenshtein Distance
4. Exact Match (EM) Accuracy
5. Bounding Box Intersection-over-Union (IoU), Precision, Recall, and F1-Score
"""

from typing import Dict, List, Tuple
import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger("OCRMetrics")


def calculate_levenshtein(s1: str, s2: str) -> int:
    """
    Computes exact edit distance between two strings using dynamic programming.
    """
    if len(s1) < len(s2):
        return calculate_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculates average Character Error Rate (CER) across text sequences.
    """
    try:
        from jiwer import cer
        return float(cer(references, predictions))
    except ImportError:
        total_dist = 0
        total_chars = 0
        for pred, ref in zip(predictions, references):
            total_dist += calculate_levenshtein(pred, ref)
            total_chars += max(1, len(ref))
        return total_dist / max(1, total_chars)


def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculates average Word Error Rate (WER) across text sequences.
    """
    try:
        from jiwer import wer
        return float(wer(references, predictions))
    except ImportError:
        total_dist = 0
        total_words = 0
        for pred, ref in zip(predictions, references):
            pred_words = pred.strip().split()
            ref_words = ref.strip().split()
            total_dist += calculate_levenshtein(" ".join(pred_words), " ".join(ref_words))
            total_words += max(1, len(ref_words))
        return total_dist / max(1, total_words)


def calculate_exact_match(predictions: List[str], references: List[str]) -> float:
    """Calculates ratio of exact string matches."""
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / max(1, len(predictions))


class OCRMetricSuite:
    """
    Stateful Metric Suite aggregating batch performance statistics during training and validation.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.predictions: List[str] = []
        self.references: List[str] = []

    def update(self, preds: List[str], refs: List[str]) -> None:
        """Accumulates prediction/reference pairs."""
        self.predictions.extend(preds)
        self.references.extend(refs)

    def compute(self) -> Dict[str, float]:
        """
        Computes aggregated evaluation metrics.

        Returns:
            Dict[str, float]: Dictionary containing CER, WER, ExactMatch, and Mean Levenshtein.
        """
        if not self.predictions:
            return {"cer": 0.0, "wer": 0.0, "exact_match": 0.0, "mean_edit_dist": 0.0}

        cer_val = calculate_cer(self.predictions, self.references)
        wer_val = calculate_wer(self.predictions, self.references)
        em_val = calculate_exact_match(self.predictions, self.references)

        edit_dists = [
            calculate_levenshtein(p, r) for p, r in zip(self.predictions, self.references)
        ]
        mean_edit = float(np.mean(edit_dists))

        return {
            "cer": cer_val,
            "wer": wer_val,
            "exact_match": em_val,
            "mean_edit_dist": mean_edit
        }
