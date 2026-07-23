"""
Abstract Base OCR Engine Interface.

Follows SOLID principles (Liskov Substitution & Dependency Inversion) to provide
a unified abstraction layer for diverse OCR backends (EasyOCR, PaddleOCR, TrOCR).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class OCRResult:
    """
    Standardized result data structure emitted by all OCR engine implementations.

    Attributes:
        text (str): Recognized text prediction string.
        bbox (List[Tuple[int, int]]): 4-point polygon bounding box coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)].
        confidence (float): Engine confidence score between 0.0 and 1.0.
        engine_name (str): Identifier of the engine that produced the prediction.
        inference_time_ms (float): Inference execution latency in milliseconds.
    """
    text: str
    bbox: List[Tuple[int, int]]
    confidence: float
    engine_name: str
    inference_time_ms: float = 0.0


class AbstractOCREngine(ABC):
    """
    Abstract Base Class enforcing standard interface contract for all OCR engines.
    """

    def __init__(self, engine_name: str) -> None:
        self.engine_name = engine_name

    @abstractmethod
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """
        Executes text detection and recognition on an input document image array.

        Args:
            image (np.ndarray): Input image in BGR or Grayscale format.

        Returns:
            List[OCRResult]: List of standardized OCR results containing text, bbox, and confidence.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Checks if the engine dependencies and pre-trained weights are loaded successfully.

        Returns:
            bool: True if engine is ready for inference, False otherwise.
        """
        pass
