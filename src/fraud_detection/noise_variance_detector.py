"""
Local Noise Variance & High-Pass Noise Inconsistency Detector.

Divides a document image into spatial grid blocks and calculates local high-pass
Laplacian noise variance. Spliced regions pasted from external files have different sensor noise signatures.
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

logger = get_logger("NoiseVarianceDetector")


class NoiseVarianceDetector(AbstractFraudDetector):
    """
    Analyzes local noise variance across document grid tiles using Laplacian high-pass filtering.
    """

    def __init__(self, block_size: int = 32, variance_threshold: float = 3.0) -> None:
        super().__init__(detector_name="Local Noise Variance Inconsistency Detector")
        self.block_size = block_size
        self.variance_threshold = variance_threshold

    def detect(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> FraudResult:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply high-pass Laplacian filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        h, w = gray.shape
        bs = self.block_size
        block_variances = []

        heatmap = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - bs + 1, bs):
            for x in range(0, w - bs + 1, bs):
                block = laplacian[y:y+bs, x:x+bs]
                var = np.var(block)
                block_variances.append(var)
                heatmap[y:y+bs, x:x+bs] = var

        if not block_variances:
            return FraudResult(
                is_fraudulent=False,
                fraud_score=0.0,
                detector_type=self.detector_name
            )

        mean_var = float(np.mean(block_variances))
        std_var = float(np.std(block_variances))

        # Normalize heatmap to [0, 255]
        if np.max(heatmap) > 0:
            heatmap_norm = (heatmap / np.max(heatmap) * 255.0).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        else:
            heatmap_color = None

        fraud_score = min(1.0, std_var / (mean_var * self.variance_threshold + 1e-5))
        is_fraud = fraud_score > 0.70

        logger.info(f"Noise Variance Analysis Complete -> Mean Var: {mean_var:.2f} | Std Var: {std_var:.2f}")

        return FraudResult(
            is_fraudulent=is_fraud,
            fraud_score=fraud_score,
            detector_type=self.detector_name,
            heatmap=heatmap_color,
            details={
                "mean_noise_variance": mean_var,
                "std_noise_variance": std_var
            }
        )
