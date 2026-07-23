"""
End-to-End System Integration Tests.
"""

import os
import pytest
from src.fraud_detection.ela_detector import ErrorLevelAnalysisDetector
from src.inference.pipeline import DocumentUnderstandingPipeline
from src.utils.pdf_reporter import EnterprisePDFReporter
from src.verification.information_extraction_engine import InformationExtractionEngine


def test_end_to_end_verification_pipeline(sample_image, tmp_path):
    # 1. Pipeline OCR Inference
    pipeline = DocumentUnderstandingPipeline(engine_name="easyocr")
    ocr_res = pipeline.run(sample_image, do_preprocess=True)
    assert "full_text" in ocr_res

    # 2. Fraud Check
    ela_detector = ErrorLevelAnalysisDetector()
    fraud_res = ela_detector.detect(sample_image)
    assert hasattr(fraud_res, "fraud_score")

    # 3. Field Extraction
    extractor = InformationExtractionEngine()
    ext_res = extractor.extract_information(ocr_res["full_text"], document_category="invoice")
    assert ext_res.document_type == "Invoice"

    # 4. Report Generation
    reporter = EnterprisePDFReporter(output_dir=str(tmp_path))
    pdf_path = reporter.generate_pdf_report(
        image_path="sample.png",
        ocr_results=ocr_res,
        forgery_results={"forgery_score": fraud_res.fraud_score},
        quality_results={"quality_score": 88.0},
        extracted_fields=ext_res.to_dict()["fields"]
    )
    assert os.path.exists(pdf_path)
