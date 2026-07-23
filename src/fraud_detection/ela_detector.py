"""
Error Level Analysis (ELA) Document Tampering Detector.

Re-saves the image at a known JPEG compression quality level (e.g. 90%) and computes
the absolute differential heatmap. In un-edited documents, error levels are uniform across
the entire image; spliced or digitally modified regions exhibit distinct error spikes.
"""

import io
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

try:
    from src.fraud_detection.base_detector import AbstractFraudDetector, FraudResult
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .base_detector import AbstractFraudDetector, FraudResult
    from ..utils.logger import get_logger

logger = get_logger("ELADetector")


class ErrorLevelAnalysisDetector(AbstractFraudDetector):
    """
    Error Level Analysis (ELA) Detector for JPEG compression artifact anomalies.
    """

    def __init__(self, quality: int = 90, scale_multiplier: float = 15.0, threshold: float = 40.0) -> None:
        """
        Args:
            quality (int): JPEG resave quality level (1-100).
            scale_multiplier (float): Amplification factor for error differences.
            threshold (float): Variance threshold score for flagging fraud.
        """
        super().__init__(detector_name="Error Level Analysis (ELA)")
        self.quality = quality
        self.scale_multiplier = scale_multiplier
        self.threshold = threshold

    def detect(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> FraudResult:
        """
        Executes Error Level Analysis on image array.

        Args:
            image (np.ndarray): Document image array (BGR or RGB).

        Returns:
            FraudResult: ELA analysis with normalized heatmap and fraud probability.
        """
        if len(image.shape) == 2:
            pil_original = Image.fromarray(image).convert("RGB")
        else:
            pil_original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Save to buffer at specified JPEG quality
        buffer = io.BytesIO()
        pil_original.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        pil_resaved = Image.open(buffer)

        # Compute absolute difference
        ela_img = ImageChops.difference(pil_original, pil_resaved)

        # Scale difference pixel values for visualization
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        scale = self.scale_multiplier if max_diff == 0 else (255.0 / max(1, max_diff))

        ela_scaled = ImageEnhance.Brightness(ela_img).enhance(scale)
        heatmap_np = np.array(ela_scaled)

        # Compute variance across heatmap
        gray_heatmap = cv2.cvtColor(heatmap_np, cv2.COLOR_RGB2GRAY)
        std_dev = float(np.std(gray_heatmap))
        mean_val = float(np.mean(gray_heatmap))

        # Normalize score between 0.0 and 1.0
        fraud_score = min(1.0, std_dev / self.threshold)
        is_fraud = fraud_score > 0.65

        logger.info(f"ELA Detection Complete -> Standard Dev: {std_dev:.2f} | Fraud Score: {fraud_score:.2f}")

        return FraudResult(
            is_fraudulent=is_fraud,
            fraud_score=fraud_score,
            detector_type=self.detector_name,
            heatmap=heatmap_np,
            details={
                "ela_std_dev": std_dev,
                "ela_mean_intensity": mean_val,
                "max_extrema_diff": max_diff
            }
        )
