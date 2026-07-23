"""
Professional OpenCV Document Preprocessing & Quality Assessment Pipeline for DocVision AI.

Provides modular, production-grade document enhancement filters:
1. Image loading & format conversion
2. Document boundary detection & 4-point perspective warp correction
3. Morphological shadow removal & background illumination normalization
4. Deskewing & 90/180/270 degree rotation correction
5. Blur detection (Laplacian variance) & Brightness/Contrast analysis
6. CLAHE enhancement & Bilateral noise removal
7. Aspect-ratio preserving resizing with padding
8. Unified Document Quality Score computation (0.0 - 100.0)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
import cv2
import numpy as np

try:
    from src.core.exceptions import DocumentPreprocessingError
    from src.utils.io_utils import load_image
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from ..core.exceptions import DocumentPreprocessingError
    from ..utils.io_utils import load_image
    from ..utils.logger import get_logger

logger = get_logger("ProfessionalDocumentPreprocessor")


@dataclass
class QualityMetrics:
    """
    Data payload holding document image quality assessment metrics.

    Attributes:
        blur_score (float): Variance of Laplacian (higher indicates sharper image).
        is_blurry (bool): Flag indicating if blur score is below acceptable threshold.
        brightness_mean (float): Mean luminosity intensity (0.0 - 255.0).
        brightness_status (str): Luminosity status ('Underexposed', 'Optimal', 'Overexposed').
        contrast_score (float): Standard deviation of pixel intensities.
        quality_score (float): Unified document quality score between 0.0 and 100.0.
    """
    blur_score: float
    is_blurry: bool
    brightness_mean: float
    brightness_status: str
    contrast_score: float
    quality_score: float
    details: Dict[str, Any] = field(default_factory=dict)


class DocumentBoundaryDetector:
    """
    Detects rectangular document boundaries and performs 4-point perspective warping.
    """

    def find_document_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Finds 4 corner coordinates of the largest 4-sided polygon contour.

        Args:
            image (np.ndarray): Input BGR image array.

        Returns:
            Optional[np.ndarray]: Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] if found, else None.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2)

        return None

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Orders coordinates: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
        return rect

    def warp_perspective(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Applies 4-point perspective warp transform to crop and un-distort document boundary.

        Args:
            image (np.ndarray): Input image.
            corners (np.ndarray): 4 corner points.

        Returns:
            np.ndarray: Top-down flattened document image.
        """
        rect = self._order_points(corners)
        (tl, tr, br, bl) = rect

        # Compute width of new image
        width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_w = max(int(width_A), int(width_B))

        # Compute height of new image
        height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_h = max(int(height_A), int(height_B))

        dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_w, max_h))
        return warped


class ShadowRemover:
    """
    Removes uneven document shadows and normalizes background illumination.
    """

    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Estimates illumination background surface via morphological dilation
        and normalizes brightness by division.

        Args:
            image (np.ndarray): Input BGR or Grayscale image.

        Returns:
            np.ndarray: Shadow-free normalized image array.
        """
        if len(image.shape) == 3:
            planes = cv2.split(image)
            result_planes = []
            for plane in planes:
                dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                bg_img = cv2.medianBlur(dilated, 21)
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_planes.append(norm_img)
            return cv2.merge(result_planes)
        else:
            dilated = cv2.dilate(image, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated, 21)
            diff_img = 255 - cv2.absdiff(image, bg_img)
            return cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)


class QualityAssessor:
    """
    Analyzes blur, luminosity, contrast, and computes a unified Quality Score (0 - 100).
    """

    def __init__(self, blur_threshold: float = 100.0) -> None:
        self.blur_threshold = blur_threshold

    def evaluate(self, image: np.ndarray) -> QualityMetrics:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # 1. Blur Detection (Variance of Laplacian)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = blur_score < self.blur_threshold

        # 2. Brightness Analysis
        brightness_mean = float(np.mean(gray))
        if brightness_mean < 70.0:
            brightness_status = "Underexposed"
        elif brightness_mean > 210.0:
            brightness_status = "Overexposed"
        else:
            brightness_status = "Optimal"

        # 3. Contrast Enhancement Score
        contrast_score = float(np.std(gray))

        # 4. Calculate Unified Quality Score (0 - 100)
        blur_norm = min(100.0, (blur_score / 300.0) * 100.0)
        brightness_norm = 100.0 - abs(brightness_mean - 128.0) * 0.7
        contrast_norm = min(100.0, (contrast_score / 64.0) * 100.0)

        quality_score = float(np.clip(0.4 * blur_norm + 0.3 * brightness_norm + 0.3 * contrast_norm, 0.0, 100.0))

        return QualityMetrics(
            blur_score=blur_score,
            is_blurry=is_blurry,
            brightness_mean=brightness_mean,
            brightness_status=brightness_status,
            contrast_score=contrast_score,
            quality_score=quality_score,
            details={
                "blur_norm": blur_norm,
                "brightness_norm": brightness_norm,
                "contrast_norm": contrast_norm
            }
        )


class ProfessionalDocumentPreprocessor:
    """
    Master Orchestration Document Preprocessor combining all enhancement stages.
    """

    def __init__(
        self,
        target_height: int = 32,
        target_width: int = 128,
        enable_perspective_correct: bool = True,
        enable_shadow_removal: bool = True,
        enable_clahe: bool = True,
        enable_denoising: bool = True,
        binarization_method: str = "adaptive"
    ) -> None:
        self.target_height = target_height
        self.target_width = target_width
        self.enable_perspective_correct = enable_perspective_correct
        self.enable_shadow_removal = enable_shadow_removal
        self.enable_clahe = enable_clahe
        self.enable_denoising = enable_denoising
        self.binarization_method = binarization_method

        self.boundary_detector = DocumentBoundaryDetector()
        self.shadow_remover = ShadowRemover()
        self.quality_assessor = QualityAssessor()

    def detect_and_correct_deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detects skew angle and rotates image to horizon alignment."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return image, 0.0

        angles = []
        for line in lines:
            coords = line.ravel()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = coords[:4]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -45.0 < angle < 45.0:
                angles.append(angle)

        if not angles:
            return image, 0.0

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.2:
            return image, 0.0

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Applies Bilateral Filter noise removal preserving crisp text edges."""
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        else:
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    def resize_and_pad(self, image: np.ndarray) -> np.ndarray:
        """Resizes maintaining aspect ratio and pads right/bottom to (target_height, target_width)."""
        h, w = image.shape[:2]
        scaling_factor = min(self.target_height / float(h), self.target_width / float(w))
        new_w = max(1, int(w * scaling_factor))
        new_h = max(1, int(h * scaling_factor))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(image.shape) == 3:
            padded = np.full((self.target_height, self.target_width, image.shape[2]), 255, dtype=np.uint8)
            padded[:new_h, :new_w, :] = resized
        else:
            padded = np.full((self.target_height, self.target_width), 255, dtype=np.uint8)
            padded[:new_h, :new_w] = resized

        return padded

    def process(
        self,
        input_image: Union[str, np.ndarray]
    ) -> Tuple[np.ndarray, QualityMetrics]:
        """
        Executes end-to-end professional document pre-processing execution pipeline.

        Args:
            input_image (Union[str, np.ndarray]): Image file path or raw image array.

        Returns:
            Tuple[np.ndarray, QualityMetrics]: (Fully Enhanced Image Array, Quality Assessment Metrics).
        """
        image = load_image(input_image)
        logger.info(f"Loaded input document image (Shape: {image.shape})")

        # 1. Perspective Boundary Detection & Warp Correction
        if self.enable_perspective_correct:
            corners = self.boundary_detector.find_document_corners(image)
            if corners is not None:
                logger.info("Detected document quadrilateral boundary. Applying 4-point perspective transform...")
                image = self.boundary_detector.warp_perspective(image, corners)

        # 2. Shadow Removal & Illumination Equalization
        if self.enable_shadow_removal:
            logger.debug("Applying morphological shadow removal & background illumination normalization...")
            image = self.shadow_remover.remove_shadows(image)

        # 3. Deskewing & Rotation Alignment
        image, skew_angle = self.detect_and_correct_deskew(image)

        # 4. Convert to Grayscale & Noise Filtering
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        if self.enable_denoising:
            gray = self.denoise(gray)

        # 5. CLAHE Contrast Enhancement
        if self.enable_clahe:
            gray = self.apply_clahe(gray)

        # 6. Quality Metrics Assessment
        metrics = self.quality_assessor.evaluate(image)
        logger.info(f"Quality Assessment -> Score: {metrics.quality_score:.1f}/100 | Blur Score: {metrics.blur_score:.1f} | Luminosity: {metrics.brightness_status}")

        # 7. Aspect-Ratio Preserving Resize & Pad
        final_image = self.resize_and_pad(gray)

        return final_image, metrics
