"""
OpenCV Document Layout Parser and Contour Segmenter.

Parses document layouts into discrete text lines and bounding boxes,
establishing reading order sorting (top-to-bottom, left-to-right).
"""

from typing import List, Tuple
import cv2
import numpy as np

try:
    from src.core.exceptions import DocumentPreprocessingError
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from ..core.exceptions import DocumentPreprocessingError
    from ..utils.logger import get_logger

logger = get_logger("DocumentLayoutParser")


class DocumentLayoutParser:
    """
    Layout Parser extracting spatial bounding polygon contours and cropped line patches.
    """

    def __init__(self, min_area: int = 100, kernel_size: Tuple[int, int] = (15, 3)) -> None:
        """
        Args:
            min_area (int): Minimum contour area threshold in pixels to filter out noise artifacts.
            kernel_size (Tuple[int, int]): Rectangular morphological kernel for connecting text lines.
        """
        self.min_area = min_area
        self.kernel_size = kernel_size

    def extract_text_regions(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects text line bounding rectangles (x, y, w, h).

        Args:
            gray_image (np.ndarray): Grayscale document image array.

        Returns:
            List[Tuple[int, int, int, int]]: List of sorted bounding boxes (x, y, w, h).
        """
        if len(gray_image.shape) != 2:
            raise DocumentPreprocessingError("Layout Parser requires single-channel Grayscale image.")

        # Morphological dilation to merge adjacent characters into continuous lines
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        dilated = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append((x, y, w, h))

        # Sort reading order: Top-to-Bottom primary, Left-to-Right secondary
        bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1] // 15, b[0]))
        logger.debug(f"Extracted {len(bounding_boxes)} document layout text region contours.")
        return bounding_boxes

    def crop_regions(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Crops image patches corresponding to layout bounding boxes."""
        crops = []
        h_img, w_img = image.shape[:2]

        for x, y, w, h in bboxes:
            x_min, y_min = max(0, x), max(0, y)
            x_max, y_max = min(w_img, x + w), min(h_img, y + h)
            crops.append(image[y_min:y_max, x_min:x_max])

        return crops
