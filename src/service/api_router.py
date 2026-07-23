"""
FastAPI Service Router for DocVision AI REST Microservice.

Provides production endpoints:
- POST /api/v1/ocr
- POST /api/v1/fraud-check
- POST /api/v1/verify-document
"""

import io
import cv2
import numpy as np
from PIL import Image
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Query
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from src.engines.easyocr_engine import EasyOCREngine
from src.fraud_detection.ela_detector import ErrorLevelAnalysisDetector
from src.inference.pipeline import DocumentUnderstandingPipeline
from src.utils.logger import get_logger
from src.verification.field_extractor import DocumentFieldExtractor
from src.verification.document_verifier import DocumentVerifier

logger = get_logger("APIService")

if HAS_FASTAPI:
    app = FastAPI(
        title="DocVision AI REST API",
        description="Production API for Document Verification, Multi-Engine OCR, and Fraud Detection",
        version="1.0.0"
    )

    @app.get("/health")
    def health_check() -> dict:
        return {"status": "healthy", "service": "DocVision AI"}

    @app.post("/api/v1/ocr")
    async def process_ocr(
        file: UploadFile = File(...),
        engine: str = Query("easyocr", description="OCR engine: easyocr, paddleocr, trocr")
    ):
        try:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            pipeline = DocumentUnderstandingPipeline(engine_name=engine)
            res = pipeline.run(bgr_img)
            return JSONResponse(content=res)
        except Exception as e:
            logger.error(f"OCR Endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/fraud-check")
    async def check_fraud(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            ela_detector = ErrorLevelAnalysisDetector()
            res = ela_detector.detect(bgr_img)

            return JSONResponse(content={
                "is_fraudulent": res.is_fraudulent,
                "fraud_score": res.fraud_score,
                "detector_used": res.detector_type,
                "details": res.details
            })
        except Exception as e:
            logger.error(f"Fraud Check error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/verify-document")
    async def verify_document(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 1. Pipeline OCR
            pipeline = DocumentUnderstandingPipeline(engine_name="easyocr")
            ocr_res = pipeline.run(bgr_img)

            # 2. Fraud Check
            ela_detector = ErrorLevelAnalysisDetector()
            fraud_res = ela_detector.detect(bgr_img)

            # 3. Field Extraction & Verification
            extractor = DocumentFieldExtractor()
            fields = extractor.extract_fields(ocr_res["full_text"])

            verifier = DocumentVerifier()
            verification = verifier.verify_invoice(fields)

            return JSONResponse(content={
                "ocr": ocr_res,
                "fraud_analysis": {
                    "is_fraudulent": fraud_res.is_fraudulent,
                    "fraud_score": fraud_res.fraud_score,
                    "detector": fraud_res.detector_type
                },
                "extracted_fields": fields,
                "verification_status": verification
            })
        except Exception as e:
            logger.error(f"Verification Endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

else:
    app = None
    logger.warning("FastAPI not installed. REST API router disabled.")
