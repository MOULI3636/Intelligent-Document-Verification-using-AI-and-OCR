"""
Multi-Modal Document Fraud & Digital Tampering Detection Subpackage.
"""

from src.fraud_detection.base_detector import AbstractFraudDetector, FraudResult
from src.fraud_detection.ela_detector import ErrorLevelAnalysisDetector
from src.fraud_detection.copy_move_detector import CopyMoveDetector
from src.fraud_detection.font_anomaly_detector import FontAnomalyDetector
from src.fraud_detection.noise_variance_detector import NoiseVarianceDetector
from src.fraud_detection.metadata_verifier import MetadataVerifier

__all__ = [
    "AbstractFraudDetector",
    "FraudResult",
    "ErrorLevelAnalysisDetector",
    "CopyMoveDetector",
    "FontAnomalyDetector",
    "NoiseVarianceDetector",
    "MetadataVerifier",
]
