"""
PaddleOCR Engine Strategy Concrete Implementation.

Wraps PaddlePaddle DBNet text detector and SVTR/CRNN recognition models.
"""

import time
from typing import List, Tuple, Optional
import numpy as np

try:
    from src.engines.base_engine import AbstractOCREngine, OCRResult
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .base_engine import AbstractOCREngine, OCRResult
    from ..utils.logger import get_logger

logger = get_logger("PaddleOCREngine")


class PaddleOCREngine(AbstractOCREngine):
    """
    PaddleOCR Implementation wrapping DBNet detection and SVTR text recognition.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True) -> None:
        super().__init__(engine_name="PaddleOCR")
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.ocr_instance = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        try:
            from paddleocr import PaddleOCR
            logger.info(f"Initializing PaddleOCR instance (lang={self.lang})...")
            self.ocr_instance = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False
            )
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logger.warning(f"PaddleOCR initialization failed: {str(e)}. Engine marked unavailable.")
            self.ocr_instance = None

    def is_available(self) -> bool:
        return self.ocr_instance is not None

    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        if not self.is_available():
            logger.error("PaddleOCR instance is not available.")
            return []

        start_time = time.time()
        try:
            results = self.ocr_instance.ocr(image, cls=self.use_angle_cls)
            elapsed_ms = (time.time() - start_time) * 1000.0

            ocr_results = []
            if results and results[0]:
                for line in results[0]:
                    bbox_poly, (text, conf) = line
                    formatted_bbox = [(int(pt[0]), int(pt[1])) for pt in bbox_poly]
                    ocr_results.append(
                        OCRResult(
                            text=text,
                            bbox=formatted_bbox,
                            confidence=float(conf),
                            engine_name=self.engine_name,
                            inference_time_ms=elapsed_ms / max(1, len(results[0]))
                        )
                    )
            return ocr_results

        except Exception as e:
            logger.error(f"PaddleOCR recognition error: {str(e)}")
            return []
