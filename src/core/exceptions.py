"""
Custom Exception Hierarchy for DocVision AI.

Enables precise exception handling and structured error reporting across OCR,
preprocessing, neural network inference, and fraud detection modules.
"""


class DocVisionException(Exception):
    """Base Exception class for all DocVision AI errors."""
    def __init__(self, message: str, error_code: str = "DOCVISION_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class ConfigValidationError(DocVisionException):
    """Raised when YAML configuration schema validation fails."""
    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="CONFIG_VALIDATION_ERROR")


class DocumentPreprocessingError(DocVisionException):
    """Raised when OpenCV deskewing, binarization, or layout segmentation fails."""
    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="PREPROCESSING_ERROR")


class OCREngineError(DocVisionException):
    """Raised when an OCR engine fails during model loading or text recognition."""
    def __init__(self, message: str, engine_name: str = "UNKNOWN") -> None:
        super().__init__(f"Engine '{engine_name}': {message}", error_code="OCR_ENGINE_ERROR")
        self.engine_name = engine_name


class FraudDetectionError(DocVisionException):
    """Raised when document fraud analysis or EXIF verification encounters an unrecoverable failure."""
    def __init__(self, message: str, detector_type: str = "UNKNOWN") -> None:
        super().__init__(f"Detector '{detector_type}': {message}", error_code="FRAUD_DETECTION_ERROR")
        self.detector_type = detector_type
