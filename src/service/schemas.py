"""
Pydantic Data Schemas for REST API Input/Output Validation.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    engine_name: str = Field("easyocr", description="Target OCR engine ('easyocr', 'paddleocr', 'trocr')")
    do_preprocess: bool = Field(True, description="Enable OpenCV pre-processing")


class OCRPredictionSchema(BaseModel):
    text: str = Field(..., description="Recognized text string")
    bbox: List[List[int]] = Field(..., description="Bounding box polygon points")
    confidence: float = Field(..., description="Model confidence score")


class OCRResponse(BaseModel):
    engine_used: str
    full_text: str
    mean_confidence: float
    predictions: List[OCRPredictionSchema]


class FraudCheckResponse(BaseModel):
    is_fraudulent: bool
    fraud_score: float = Field(..., description="Aggregate fraud score [0.0 - 1.0]")
    detector_used: str
    details: Dict[str, Any]


class VerificationResponse(BaseModel):
    document_category: str
    ocr_result: OCRResponse
    fraud_analysis: FraudCheckResponse
    extracted_fields: Dict[str, Any]
    is_verified: bool
