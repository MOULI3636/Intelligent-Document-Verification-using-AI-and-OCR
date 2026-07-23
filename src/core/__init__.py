"""
DocVision AI Core Package. Provides exception classes and core infrastructure.
"""

from src.core.exceptions import (
    DocVisionException,
    OCREngineError,
    FraudDetectionError,
    ConfigValidationError,
    DocumentPreprocessingError
)

__all__ = [
    "DocVisionException",
    "OCREngineError",
    "FraudDetectionError",
    "ConfigValidationError",
    "DocumentPreprocessingError",
]
