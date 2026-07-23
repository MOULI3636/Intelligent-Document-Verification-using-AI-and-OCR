"""
Copy-Move Forgery Detector using SIFT/ORB Keypoint Matching & Spatial Clustering.

Detects duplicated or cloned region patches within a document (e.g. copied signatures,
forged numbers, or duplicated stamps) by matching keypoint descriptors with spatial distance constraints.
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

logger = get_logger("CopyMoveDetector")


class CopyMoveDetector(AbstractFraudDetector):
    """
    Copy-Move Forgery Detector utilizing feature descriptors (SIFT / ORB).
    """

    def __init__(self, min_match_count: int = 8, spatial_distance_min: float = 25.0) -> None:
        """
        Args:
            min_match_count (int): Minimum number of matched keypoint pairs to flag duplication.
            spatial_distance_min (float): Minimum Euclidean pixel distance required between keypoint pairs to exclude self-matches.
        """
        super().__init__(detector_name="Copy-Move Forgery Detector")
        self.min_match_count = min_match_count
        self.spatial_distance_min = spatial_distance_min

    def detect(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> FraudResult:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Initialize detector (Try SIFT, fallback to ORB if SIFT non-free)
        try:
            detector = cv2.SIFT_create(nfeatures=2000)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
        except Exception:
            detector = cv2.ORB_create(nfeatures=2000)
            keypoints, descriptors = detector.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) < 10:
            return FraudResult(
                is_fraudulent=False,
                fraud_score=0.0,
                detector_type=self.detector_name,
                details={"keypoints_found": 0, "copy_move_matches": 0}
            )

        # Match descriptors against self to find duplicated keypoints
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=10)

        cloned_matches = []
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        for match_group in matches:
            for m in match_group[1:]: # Skip self-match at index 0
                pt1 = np.array(keypoints[m.queryIdx].pt)
                pt2 = np.array(keypoints[m.trainIdx].pt)
                dist = np.linalg.norm(pt1 - pt2)

                # Filter spatially separated matches with high similarity
                if dist > self.spatial_distance_min and m.distance < 120.0:
                    cloned_matches.append((pt1, pt2))
                    cv2.line(canvas, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 1)

        num_clones = len(cloned_matches)
        fraud_score = min(1.0, num_clones / max(1, self.min_match_count * 2))
        is_fraud = num_clones >= self.min_match_count

        logger.info(f"Copy-Move Detection -> Cloned Matches Found: {num_clones} | Fraud Score: {fraud_score:.2f}")

        return FraudResult(
            is_fraudulent=is_fraud,
            fraud_score=fraud_score,
            detector_type=self.detector_name,
            heatmap=canvas if is_fraud else None,
            details={
                "keypoints_count": len(keypoints),
                "cloned_matches_count": num_clones
            }
        )
