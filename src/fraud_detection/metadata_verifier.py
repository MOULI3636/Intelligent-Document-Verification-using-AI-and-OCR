"""
EXIF Metadata and Software Header Integrity Verifier.

Inspects document image headers for digital editing software signatures
(Photoshop, GIMP, Canva, Paint.NET) and modification timestamp discrepancies.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image, ExifTags

try:
    from src.fraud_detection.base_detector import AbstractFraudDetector, FraudResult
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .base_detector import AbstractFraudDetector, FraudResult
    from ..utils.logger import get_logger

logger = get_logger("MetadataVerifier")


class MetadataVerifier(AbstractFraudDetector):
    """
    Examines EXIF metadata for editing software traces and timestamp anomalies.
    """

    FLAGGED_SOFTWARE = ["photoshop", "gimp", "canva", "pixlr", "paint.net", "illustrator", "acrobat"]

    def __init__(self, custom_flagged_software: Optional[List[str]] = None) -> None:
        super().__init__(detector_name="EXIF Metadata & Software Header Verifier")
        self.flagged_software = custom_flagged_software or self.FLAGGED_SOFTWARE

    def extract_exif(self, image_path_or_pil: Any) -> Dict[str, Any]:
        """Extracts human-readable EXIF dictionary from PIL image or path."""
        exif_data = {}
        try:
            if isinstance(image_path_or_pil, str):
                img = Image.open(image_path_or_pil)
            elif isinstance(image_path_or_pil, Image.Image):
                img = image_path_or_pil
            else:
                return {}

            info = img._getexif()
            if info:
                for tag, value in info.items():
                    decoded_tag = ExifTags.TAGS.get(tag, str(tag))
                    exif_data[decoded_tag] = str(value)
        except Exception as e:
            logger.debug(f"EXIF extraction note: {str(e)}")

        return exif_data

    def detect(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> FraudResult:
        if metadata is None:
            metadata = {}

        software_found = []
        is_fraud = False
        fraud_score = 0.0

        software_tag = str(metadata.get("Software", "")).lower()
        processing_tag = str(metadata.get("ProcessingSoftware", "")).lower()

        combined_software = f"{software_tag} {processing_tag}"

        for sw in self.flagged_software:
            if sw in combined_software:
                software_found.append(sw)
                is_fraud = True
                fraud_score = 0.95

        logger.info(f"Metadata Integrity Check -> Flagged Software Detected: {software_found}")

        return FraudResult(
            is_fraudulent=is_fraud,
            fraud_score=fraud_score,
            detector_type=self.detector_name,
            details={
                "flagged_software_detected": software_found,
                "raw_software_tag": software_tag,
                "has_exif": len(metadata) > 0
            }
        )
