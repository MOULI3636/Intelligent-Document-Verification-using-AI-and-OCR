"""
OCR Engine Zoo Interface Tests.
"""

import pytest
from src.engines.base_engine import AbstractOCREngine, OCRResult
from src.engines.easyocr_engine import EasyOCREngine
from src.engines.paddleocr_engine import PaddleOCREngine
from src.engines.trocr_engine import TrOCREngine


def test_easyocr_engine_interface(sample_image):
    engine = EasyOCREngine(gpu=False)
    assert isinstance(engine, AbstractOCREngine)
    if engine.is_available():
        results = engine.recognize(sample_image)
        assert isinstance(results, list)


def test_paddleocr_engine_interface(sample_image):
    engine = PaddleOCREngine()
    assert isinstance(engine, AbstractOCREngine)
    if engine.is_available():
        results = engine.recognize(sample_image)
        assert isinstance(results, list)


def test_trocr_engine_interface(sample_image):
    engine = TrOCREngine(device="cpu")
    assert isinstance(engine, AbstractOCREngine)
    if engine.is_available():
        results = engine.recognize(sample_image)
        assert isinstance(results, list)
