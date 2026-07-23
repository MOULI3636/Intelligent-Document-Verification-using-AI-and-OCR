"""
Dataset Validator Module for DocVision AI.

Performs automated image corruption checks, resolution thresholds, annotation schema
validation, and document label verification for Invoices, Receipts, and ID Cards (PAN, Aadhaar, Passport, Driving License).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

from src.core.exceptions import DocVisionException
from src.utils.logger import get_logger

logger = get_logger("DatasetValidator")


@dataclass
class ValidationReport:
    """
    Data payload summarizing dataset validation results.
    """
    total_found: int = 0
    valid_samples: int = 0
    corrupt_samples: int = 0
    invalid_annotations: int = 0
    errors: List[str] = field(default_factory=list)
    valid_sample_paths: List[Tuple[str, str, str]] = field(default_factory=list) # (image_path, text_label, doc_category)


class DatasetValidator:
    """
    Validates document image integrity and annotation schema formatting.
    """

    SUPPORTED_CATEGORIES = {
        "invoice", "receipt", "passport", "pan_card", "aadhaar_card", "driving_license", "generic"
    }

    def __init__(
        self,
        min_width: int = 32,
        min_height: int = 16,
        allowed_extensions: Optional[List[str]] = None
    ) -> None:
        """
        Args:
            min_width (int): Minimum acceptable pixel width.
            min_height (int): Minimum acceptable pixel height.
            allowed_extensions (Optional[List[str]]): List of valid file extensions.
        """
        self.min_width = min_width
        self.min_height = min_height
        self.allowed_extensions = set(allowed_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tiff"])

    def validate_image_file(self, image_path: str) -> Tuple[bool, Optional[str]]:
        """
        Verifies if an image file exists, is uncorrupted, and satisfies resolution bounds.

        Args:
            image_path (str): Filepath to image.

        Returns:
            Tuple[bool, Optional[str]]: (Is_Valid, Error_Reason).
        """
        if not os.path.exists(image_path):
            return False, f"File does not exist: {image_path}"

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in self.allowed_extensions:
            return False, f"Unsupported file extension '{ext}': {image_path}"

        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, f"OpenCV failed to decode image: {image_path}"
            
            h, w = img.shape[:2]
            if w < self.min_width or h < self.min_height:
                return False, f"Image resolution ({w}x{h}) below minimum threshold ({self.min_width}x{self.min_height}): {image_path}"

            return True, None

        except Exception as e:
            return False, f"Exception reading image '{image_path}': {str(e)}"

    def validate_dataset_samples(
        self,
        samples: List[Dict[str, Any]]
    ) -> ValidationReport:
        """
        Validates list of sample dictionaries.

        Sample format:
        {
            "image_path": "path/to/img.png",
            "text": "LABEL TEXT",
            "category": "passport" # invoice, receipt, pan_card, aadhaar_card, driving_license, passport
        }

        Returns:
            ValidationReport: Summary payload containing validated sample list and errors.
        """
        report = ValidationReport(total_found=len(samples))

        for sample in samples:
            img_path = sample.get("image_path", "")
            label_text = str(sample.get("text", "")).strip()
            category = str(sample.get("category", "generic")).lower()

            if category not in self.SUPPORTED_CATEGORIES:
                category = "generic"

            # Check image integrity
            is_valid_img, error_msg = self.validate_image_file(img_path)
            if not is_valid_img:
                report.corrupt_samples += 1
                report.errors.append(error_msg or f"Invalid image at {img_path}")
                continue

            # Check annotation validity
            if not label_text:
                report.invalid_annotations += 1
                report.errors.append(f"Empty text label for sample: {img_path}")
                continue

            report.valid_samples += 1
            report.valid_sample_paths.append((img_path, label_text, category))

        logger.info(f"Dataset Validation Complete -> {report.valid_samples}/{report.total_found} valid samples.")
        return report
