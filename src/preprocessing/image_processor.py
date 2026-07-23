"""
OpenCV Document Image Preprocessing Module.

Implements production-quality document image pre-processing steps:
1. Skew angle detection and automated deskewing (Hough Line Transform & minAreaRect)
2. Adaptive binarization and Sauvola/Otsu thresholding
3. Contrast Limited Adaptive Histogram Equalization (CLAHE)
4. Aspect-ratio preserving resizing with padding for tensor ingestion
"""

import math
from typing import Tuple
import cv2
import numpy as np

try:
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from ..utils.logger import get_logger

logger = get_logger("DocumentImageProcessor")


class DocumentImageProcessor:
    """
    Modular Document Image Preprocessor performing OpenCV enhancement pipelines.
    Designed for production deployment in OCR recognition systems.
    """

    def __init__(
        self,
        target_height: int = 32,
        target_width: int = 128,
        grayscale: bool = True,
        binarization_method: str = "adaptive",
        block_size: int = 11,
        c_value: int = 2
    ) -> None:
        """
        Initializes the Document Image Preprocessor.

        Args:
            target_height (int): Height for normalized output image.
            target_width (int): Width for normalized output image.
            grayscale (bool): Whether to convert images to single-channel grayscale.
            binarization_method (str): Binarization method ('adaptive', 'otsu', 'none').
            block_size (int): Block size for adaptive thresholding.
            c_value (int): Constant subtracted from mean in adaptive thresholding.
        """
        self.target_height = target_height
        self.target_width = target_width
        self.grayscale = grayscale
        self.binarization_method = binarization_method
        self.block_size = block_size
        self.c_value = c_value

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Converts BGR/RGB image to single-channel Grayscale."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def detect_skew_angle(self, gray: np.ndarray, max_angle: float = 45.0) -> float:
        """
        Detects document skew angle using Canny Edge Extraction and MinAreaRect.

        Args:
            gray (np.ndarray): Grayscale document image.
            max_angle (float): Upper bound threshold for valid rotation correction.

        Returns:
            float: Detected skew angle in degrees (-45.0 to +45.0).
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 200, apertureSize=3)

        coords = np.column_stack(np.where(edges > 0))
        if len(coords) < 10:
            return 0.0

        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        if abs(angle) > max_angle:
            return 0.0

        return angle

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotates an image by specified angle around its center with border reflection.

        Args:
            image (np.ndarray): Input image array.
            angle (float): Rotation angle in degrees.

        Returns:
            np.ndarray: Deskewed image array.
        """
        if abs(angle) < 0.1:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detects and corrects skew angle for input document image.

        Args:
            image (np.ndarray): Input BGR/Grayscale image.

        Returns:
            Tuple[np.ndarray, float]: (Deskewed Image, Angle applied).
        """
        gray = self.to_grayscale(image)
        angle = self.detect_skew_angle(gray)
        if abs(angle) > 0.1:
            logger.debug(f"Deskewing document by {angle:.2f} degrees")
            deskewed = self.rotate_image(image, angle)
            return deskewed, angle
        return image, 0.0

    def apply_binarization(self, gray: np.ndarray) -> np.ndarray:
        """
        Applies adaptive thresholding or Otsu binarization.

        Args:
            gray (np.ndarray): Single channel grayscale image.

        Returns:
            np.ndarray: Binarized document image.
        """
        if self.binarization_method == "adaptive":
            # Ensure block_size is odd and > 1
            bs = self.block_size if self.block_size % 2 != 0 else self.block_size + 1
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, bs, self.c_value
            )
        elif self.binarization_method == "otsu":
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return bin_img
        return gray

    def apply_clahe(self, gray: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(gray)

    def resize_and_pad(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes document text image to fixed (target_height, target_width) maintaining aspect ratio.
        Pads right and bottom with white background (value 255).
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_height, self.target_width

        scaling_factor = min(target_h / float(h), target_w / float(w))
        new_w = max(1, int(w * scaling_factor))
        new_h = max(1, int(h * scaling_factor))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(image.shape) == 3:
            padded = np.full((target_h, target_w, image.shape[2]), 255, dtype=np.uint8)
            padded[:new_h, :new_w, :] = resized
        else:
            padded = np.full((target_h, target_w), 255, dtype=np.uint8)
            padded[:new_h, :new_w] = resized

        return padded

    def process(self, image: np.ndarray, do_deskew: bool = True, do_resize: bool = True) -> np.ndarray:
        """
        Full End-to-End document pre-processing execution pipeline.

        Args:
            image (np.ndarray): Raw input image array.
            do_deskew (bool): Flag toggling automatic skew correction.
            do_resize (bool): Whether to resize and pad the image to target dimensions.

        Returns:
            np.ndarray: Fully processed, binarized, and normalized document image.
        """
        processed = image.copy()

        if do_deskew:
            processed, _ = self.deskew(processed)

        if self.grayscale:
            processed = self.to_grayscale(processed)

        processed = self.apply_clahe(processed)

        if self.binarization_method != "none":
            processed = self.apply_binarization(processed)

        if do_resize:
            processed = self.resize_and_pad(processed)
        return processed
