"""
EasyOCR Engine Strategy Concrete Implementation.

Wraps PyTorch-based EasyOCR engine (CRAFT Text Detector + ResNet-LSTM Recognizer).
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

logger = get_logger("EasyOCREngine")


class EasyOCREngine(AbstractOCREngine):
    """
    EasyOCR Implementation wrapping PyTorch CRAFT text detection and recognition.
    """

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = True) -> None:
        """
        Initializes EasyOCR Reader.

        Args:
            languages (Optional[List[str]]): Supported ISO language codes (default: ['en']).
            gpu (bool): Enable CUDA GPU acceleration if available.
        """
        super().__init__(engine_name="EasyOCR")
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.reader = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        try:
            import easyocr
            logger.info(f"Initializing EasyOCR Reader (languages={self.languages}, gpu={self.gpu})...")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR initialized successfully.")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {str(e)}. Engine marked unavailable.")
            self.reader = None

    def is_available(self) -> bool:
        return self.reader is not None

    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """
        Executes EasyOCR text detection and recognition.

        Args:
            image (np.ndarray): BGR/RGB image array.

        Returns:
            List[OCRResult]: Standardized predictions list.
        """
        if not self.is_available():
            logger.error("EasyOCR reader is not available.")
            return []

        start_time = time.time()
        try:
            results = self.reader.readtext(image)
            elapsed_ms = (time.time() - start_time) * 1000.0

            ocr_results = []
            for bbox_poly, text, conf in results:
                formatted_bbox = [(int(pt[0]), int(pt[1])) for pt in bbox_poly]
                ocr_results.append(
                    OCRResult(
                        text=text,
                        bbox=formatted_bbox,
                        confidence=float(conf),
                        engine_name=self.engine_name,
                        inference_time_ms=elapsed_ms / max(1, len(results))
                    )
                )
            return ocr_results

        except Exception as e:
            logger.error(f"EasyOCR recognition error: {str(e)}")
            return []
