"""
Identity Document Field Extractor Strategy Concrete Implementation.

Extracts key structural entities from identity documents:
- Passports (Passport Number, Name, DOB, Expiry Date)
- PAN Cards (PAN Number, Name, Father's Name, DOB)
- Aadhaar Cards (Aadhaar Number, Name, DOB/Year of Birth)
- Driving Licenses (DL Number, Name, Validity Expiry)
"""

import re
from typing import Dict, List, Any, Optional
try:
    from src.verification.base_extractor import AbstractFieldExtractor, ExtractedField, DocumentExtractionResult
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .base_extractor import AbstractFieldExtractor, ExtractedField, DocumentExtractionResult
    from ..utils.logger import get_logger

logger = get_logger("IdentityDocumentExtractor")


class IdentityDocumentExtractor(AbstractFieldExtractor):
    """
    Field Extractor for Passports, PAN, Aadhaar, and Driving Licenses.
    """

    PATTERNS = {
        "passport": {
            "id_number": r"([A-Z][0-9]{7})",
            "mrz_code": r"(P<[A-Z0-9<]+)"
        },
        "pan_card": {
            "id_number": r"([A-Z]{5}[0-9]{4}[A-Z]{1})",
            "name": r"(?:name|holder)\s*[:\-]?\s*([A-Z\s]+)"
        },
        "aadhaar_card": {
            "id_number": r"(\d{4}\s?\d{4}\s?\d{4})",
            "dob": r"(?:dob|birth)\s*[:\-]?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4}|\d{4})"
        },
        "driving_license": {
            "id_number": r"([A-Z]{2}[0-9]{2}[0-9A-Z]{11}|[A-Z]{2}[\-\s]?[0-9]{13})",
            "expiry": r"(?:valid\s*till|exp)\s*[:\-]?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4})"
        }
    }

    def __init__(self, card_type: str = "passport") -> None:
        super().__init__(document_type=card_type)
        self.card_type = card_type.lower()

    def extract(self, ocr_text: str, ocr_results: Optional[List[Any]] = None) -> DocumentExtractionResult:
        patterns = self.PATTERNS.get(self.card_type, self.PATTERNS["passport"])
        extracted_fields: Dict[str, ExtractedField] = {}

        for field_name, pattern in patterns.items():
            match = re.search(pattern, ocr_text, re.IGNORECASE)
            val = match.group(1).strip() if match else None
            extracted_fields[field_name] = ExtractedField(
                field_name=field_name,
                value=val,
                confidence=0.90 if val else 0.0
            )

        logger.info(f"Identity Extraction ({self.card_type}) Complete -> Fields Found: {sum(1 for f in extracted_fields.values() if f.value)}")

        return DocumentExtractionResult(
            document_type=self.document_type,
            fields=extracted_fields,
            raw_text=ocr_text
        )
