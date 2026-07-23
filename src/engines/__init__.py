"""
Multi-Engine OCR Strategy Zoo Subpackage.

Implements SOLID Open/Closed and Interface Segregation principles via Abstract Base Classes.
"""

from src.engines.base_engine import AbstractOCREngine, OCRResult
from src.engines.easyocr_engine import EasyOCREngine
from src.engines.paddleocr_engine import PaddleOCREngine
from src.engines.trocr_engine import TrOCREngine

__all__ = [
    "AbstractOCREngine",
    "OCRResult",
    "EasyOCREngine",
    "PaddleOCREngine",
    "TrOCREngine",
]
