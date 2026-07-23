"""
FastAPI REST API Service Subpackage.
"""

from src.service.schemas import OCRRequest, FraudCheckResponse, VerificationResponse
from src.service.api_router import app

__all__ = ["OCRRequest", "FraudCheckResponse", "VerificationResponse", "app"]
