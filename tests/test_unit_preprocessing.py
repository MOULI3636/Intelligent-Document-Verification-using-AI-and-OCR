"""
Unit Tests for Document Preprocessing Pipeline.
"""

import numpy as np
import pytest
from src.preprocessing.image_processor import DocumentImageProcessor
from src.preprocessing.document_pipeline import ProfessionalDocumentPreprocessor, QualityAssessor
from src.preprocessing.layout_parser import DocumentLayoutParser


def test_grayscale_conversion(sample_image):
    processor = DocumentImageProcessor()
    gray = processor.to_grayscale(sample_image)
    assert len(gray.shape) == 2


def test_resize_and_pad(sample_grayscale_image):
    processor = DocumentImageProcessor(target_height=32, target_width=128)
    padded = processor.resize_and_pad(sample_grayscale_image)
    assert padded.shape == (32, 128)


def test_quality_assessor(sample_image):
    assessor = QualityAssessor()
    metrics = assessor.evaluate(sample_image)
    assert metrics.quality_score >= 0.0
    assert metrics.quality_score <= 100.0
    assert metrics.blur_score >= 0.0


def test_layout_parser(sample_grayscale_image):
    parser = DocumentLayoutParser()
    bboxes = parser.extract_text_regions(sample_grayscale_image)
    assert isinstance(bboxes, list)


def test_professional_preprocessor(sample_image):
    preprocessor = ProfessionalDocumentPreprocessor(target_height=32, target_width=128)
    processed, metrics = preprocessor.process(sample_image)
    assert processed.shape == (32, 128)
    assert metrics.quality_score >= 0.0
