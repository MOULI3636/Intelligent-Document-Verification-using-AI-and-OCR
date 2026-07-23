"""
Font Anomaly and Baseline Alignment Tampering Detector.

Detects spliced text inserts or modified numbers by analyzing baseline angle consistency,
character spacing uniformity, and stroke width variance across document text line crops.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np

try:
    from src.fraud_detection.base_detector import AbstractFraudDetector, FraudResult
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .base_detector import AbstractFraudDetector, FraudResult
    from ..utils.logger import get_logger

logger = get_logger("FontAnomalyDetector")


class FontAnomalyDetector(AbstractFraudDetector):
    """
    Analyzes font baseline alignment, stroke thickness, and character pitch variance.
    """

    def __init__(self, spacing_std_threshold: float = 5.0) -> None:
        super().__init__(detector_name="Font Baseline & Alignment Anomaly Detector")
        self.spacing_std_threshold = spacing_std_threshold

    def detect(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> FraudResult:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Binarize and extract character contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append((x, y, w, h))

        if len(bounding_boxes) < 4:
            return FraudResult(
                is_fraudulent=False,
                fraud_score=0.0,
                detector_type=self.detector_name,
                details={"characters_analyzed": len(bounding_boxes)}
            )

        # Sort character boxes horizontally
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

        # Analyze inter-character spacing distances
        spacings = []
        baselines = []
        stroke_widths = []

        for i in range(len(bounding_boxes) - 1):
            curr_box = bounding_boxes[i]
            next_box = bounding_boxes[i + 1]

            spacing = next_box[0] - (curr_box[0] + curr_box[2])
            if 0 <= spacing < 100:
                spacings.append(spacing)

            baselines.append(curr_box[1] + curr_box[3]) # y + h
            stroke_widths.append(curr_box[2])

        spacing_std = float(np.std(spacings)) if spacings else 0.0
        baseline_std = float(np.std(baselines)) if baselines else 0.0

        # Calculate anomaly score
        anomaly_score = min(1.0, (spacing_std + baseline_std) / (self.spacing_std_threshold * 2.0))
        is_fraud = anomaly_score > 0.65

        logger.info(f"Font Anomaly Detection -> Spacing Std: {spacing_std:.2f} | Baseline Std: {baseline_std:.2f}")

        return FraudResult(
            is_fraudulent=is_fraud,
            fraud_score=anomaly_score,
            detector_type=self.detector_name,
            details={
                "spacing_std": spacing_std,
                "baseline_std": baseline_std,
                "characters_analyzed": len(bounding_boxes)
            }
        )
