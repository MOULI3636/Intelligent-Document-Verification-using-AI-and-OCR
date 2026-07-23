"""
Abstract Base Fraud Detector Interface.

Follows SOLID Open/Closed Principles to define a unified contract for diverse
document forgery and digital tampering detection strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class FraudResult:
    """
    Standardized result data payload returned by document fraud detectors.

    Attributes:
        is_fraudulent (bool): Flag indicating if fraud probability exceeds detection threshold.
        fraud_score (float): Confidence score between 0.0 (authentic) and 1.0 (highly fraudulent).
        detector_type (str): Identifier of the detection algorithm.
        heatmap (Optional[np.ndarray]): Visual 2D heatmap highlighting tampered image regions.
        details (Dict[str, Any]): Additional metric breakdown (e.g., EXIF software, SIFT matches).
    """
    is_fraudulent: bool
    fraud_score: float
    detector_type: str
    heatmap: Optional[np.ndarray] = None
    details: Dict[str, Any] = field(default_factory=dict)


class AbstractFraudDetector(ABC):
    """
    Abstract Base Class for all document fraud & digital forgery detectors.
    """

    def __init__(self, detector_name: str) -> None:
        self.detector_name = detector_name

    @abstractmethod
    def detect(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> FraudResult:
        """
        Analyzes document image or metadata for digital forgery or tampering.

        Args:
            image (np.ndarray): Document image array.
            metadata (Optional[Dict[str, Any]]): Optional image EXIF metadata dict.

        Returns:
            FraudResult: Standardized result containing fraud flag, score, and visual heatmap.
        """
        pass
