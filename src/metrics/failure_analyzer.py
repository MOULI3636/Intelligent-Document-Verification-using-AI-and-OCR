"""
Automated Document AI & OCR Failure Analysis Engine for DocVision AI.

Automatically identifies, diagnoses, categorizes, renders visual overlays,
and stores failure cases across:
1. False Positives (FP)
2. False Negatives (FN)
3. Low Confidence Predictions (< confidence threshold)
4. Blur Failures (Diagnosed via Laplacian Variance)
5. Lighting Failures (Diagnosed via Underexposed / Overexposed Luminosity)
6. OCR Sequence Mismatches (Diagnosed via High CER / WER)

Generates diagnostic reports explaining root cause rationale for model failures.
"""

from dataclasses import dataclass, asdict, field
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import cv2
import numpy as np

try:
    from src.preprocessing.document_pipeline import QualityAssessor
    from src.utils.io_utils import save_json, load_image
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from ..preprocessing.document_pipeline import QualityAssessor
    from ..utils.io_utils import save_json, load_image
    from ..utils.logger import get_logger

logger = get_logger("FailureAnalysisEngine")


@dataclass
class FailureCase:
    """
    Payload representing single identified model failure case.

    Attributes:
        sample_id: Unique identifier or image filename.
        failure_type: Taxonomy classification ('False Positive', 'False Negative', 'Low Confidence', 'Blur Failure', 'Lighting Failure', 'OCR Mismatch').
        true_label: Ground truth reference label.
        predicted_label: Model prediction label.
        confidence: Prediction confidence score.
        blur_score: Laplacian variance score.
        brightness_mean: Average pixel intensity.
        root_cause_explanation: Detailed text rationale explaining why prediction failed.
        image_path: Source image filepath.
        saved_artifact_path: Saved failure image overlay filepath.
    """
    sample_id: str
    failure_type: str
    true_label: str
    predicted_label: str
    confidence: float
    blur_score: float
    brightness_mean: float
    root_cause_explanation: str
    image_path: str
    saved_artifact_path: Optional[str] = None


class FailureAnalysisEngine:
    """
    Automated Failure Analysis Engine diagnosing failure root-causes and storing examples.
    """

    def __init__(
        self,
        output_dir: str = "failure_analysis",
        confidence_threshold: float = 0.60,
        blur_threshold: float = 100.0,
        min_brightness: float = 70.0,
        max_brightness: float = 210.0
    ) -> None:
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

        self.quality_assessor = QualityAssessor(blur_threshold=blur_threshold)

        # Create subdirectories for failure categories
        self.categories = [
            "false_positives",
            "false_negatives",
            "low_confidence",
            "blur_failures",
            "lighting_failures",
            "ocr_mismatches"
        ]
        for cat in self.categories:
            os.makedirs(os.path.join(self.output_dir, cat), exist_ok=True)

    def diagnose_sample(
        self,
        sample_id: str,
        image_input: Union[str, np.ndarray],
        true_label: str,
        predicted_label: str,
        confidence: float
    ) -> Optional[FailureCase]:
        """
        Diagnoses whether a prediction constitutes a failure and determines root cause rationale.

        Args:
            sample_id (str): Identifier.
            image_input (Union[str, np.ndarray]): Image path or array.
            true_label (str): Ground truth string.
            predicted_label (str): Model prediction string.
            confidence (float): Confidence probability score.

        Returns:
            Optional[FailureCase]: Failure case payload if sample failed, else None.
        """
        image_path = image_input if isinstance(image_input, str) else f"sample_{sample_id}.png"
        image = load_image(image_input)

        # Quality metrics assessment
        q_metrics = self.quality_assessor.evaluate(image)
        blur_score = q_metrics.blur_score
        brightness = q_metrics.brightness_mean

        is_incorrect = true_label.strip().lower() != predicted_label.strip().lower()
        is_low_conf = confidence < self.confidence_threshold

        if not is_incorrect and not is_low_conf:
            return None # Prediction succeeded cleanly

        # Determine Primary Failure Taxonomy & Root Cause Rationale
        explanations = []
        failure_type = "Unclassified Failure"
        cat_folder = "low_confidence"

        if q_metrics.is_blurry:
            failure_type = "Blur Failure"
            cat_folder = "blur_failures"
            explanations.append(f"Image is out-of-focus or motion blurry (Laplacian variance: {blur_score:.1f} < threshold {self.blur_threshold:.1f}). Crisp text edges lost.")

        if brightness < self.min_brightness:
            failure_type = "Lighting Failure (Underexposed)"
            cat_folder = "lighting_failures"
            explanations.append(f"Document is severely underexposed (Mean luminosity: {brightness:.1f} < {self.min_brightness:.1f}). Text contrast obscured by darkness.")
        elif brightness > self.max_brightness:
            failure_type = "Lighting Failure (Overexposed)"
            cat_folder = "lighting_failures"
            explanations.append(f"Document is overexposed / washed out (Mean luminosity: {brightness:.1f} > {self.max_brightness:.1f}). High specular reflection.")

        if is_incorrect and failure_type not in ["Blur Failure", "Lighting Failure (Underexposed)", "Lighting Failure (Overexposed)"]:
            if confidence >= self.confidence_threshold:
                failure_type = "False Positive"
                cat_folder = "false_positives"
                explanations.append(f"High-confidence false prediction (Conf: {confidence:.2f}). Model misclassified '{true_label}' as '{predicted_label}'.")
            else:
                failure_type = "False Negative / OCR Mismatch"
                cat_folder = "ocr_mismatches"
                explanations.append(f"Prediction sequence mismatch (Conf: {confidence:.2f}). Expected '{true_label}', got '{predicted_label}'.")

        if is_low_conf and not is_incorrect and not explanations:
            failure_type = "Low Confidence"
            cat_folder = "low_confidence"
            explanations.append(f"Correct label prediction but confidence score ({confidence:.2f}) fell below operational threshold ({self.confidence_threshold:.2f}).")

        root_cause = " | ".join(explanations)

        # Render visual diagnostic overlay banner
        canvas = image.copy()
        if len(canvas.shape) == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        h, w = canvas.shape[:2]
        banner_h = 70
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)

        color = (0, 0, 255) if is_incorrect else (0, 165, 255)
        cv2.rectangle(banner, (0, 0), (w, banner_h), color, -1)

        txt1 = f"FAILURE: {failure_type} | Conf: {confidence:.2f}"
        txt2 = f"True: '{true_label}' | Pred: '{predicted_label}'"
        cv2.putText(banner, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(banner, txt2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        annotated = np.vstack([banner, canvas])
        saved_path = os.path.join(self.output_dir, cat_folder, f"failure_{sample_id}.png")
        cv2.imwrite(saved_path, annotated)

        return FailureCase(
            sample_id=sample_id,
            failure_type=failure_type,
            true_label=true_label,
            predicted_label=predicted_label,
            confidence=confidence,
            blur_score=blur_score,
            brightness_mean=brightness,
            root_cause_explanation=root_cause,
            image_path=image_path,
            saved_artifact_path=saved_path
        )

    def analyze_batch(
        self,
        predictions_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes failure analysis over a batch of sample prediction records.

        Args:
            predictions_data (List[Dict[str, Any]]): List of records containing:
                {'sample_id', 'image_input', 'true_label', 'predicted_label', 'confidence'}

        Returns:
            Dict[str, Any]: Consolidated failure report payload.
        """
        failure_cases: List[FailureCase] = []
        failure_counts: Dict[str, int] = {}

        for record in predictions_data:
            case = self.diagnose_sample(
                sample_id=str(record.get("sample_id", "0")),
                image_input=record.get("image_input"),
                true_label=str(record.get("true_label", "")),
                predicted_label=str(record.get("predicted_label", "")),
                confidence=float(record.get("confidence", 0.0))
            )
            if case is not None:
                failure_cases.append(case)
                failure_counts[case.failure_type] = failure_counts.get(case.failure_type, 0) + 1

        report_payload = {
            "total_analyzed": len(predictions_data),
            "total_failures": len(failure_cases),
            "failure_rate_pct": (len(failure_cases) / max(1, len(predictions_data))) * 100.0,
            "failure_breakdown": failure_counts,
            "failure_cases": [asdict(c) for c in failure_cases]
        }

        report_path = os.path.join(self.output_dir, "failure_analysis_report.json")
        save_json(report_payload, report_path)
        logger.info(f"Failure Analysis Complete -> Identified {len(failure_cases)} failures. Saved report to: '{report_path}'")

        return report_payload
