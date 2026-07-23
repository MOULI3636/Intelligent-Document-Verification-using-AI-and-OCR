"""
Document Field Extractor using Regex Patterns and Spatial Keyword Association.

Extracts key structural fields (Invoice #, Date, Total Amount, Tax ID, Names)
from recognized text sequences.
"""

import re
from typing import Dict, Any, List

from src.utils.logger import get_logger

logger = get_logger("DocumentFieldExtractor")


class DocumentFieldExtractor:
    """
    Regex and spatial field extractor for structured document understanding.
    """

    PATTERNS = {
        "invoice_number": r"(?:invoice|inv|bill)\s*#?\s*[:\-]?\s*([A-Z0-9\-]+)",
        "total_amount": r"(?:total|amount|due|pay)\s*[:\-]?\s*\$?\s*([\d,]+\.\d{2})",
        "date": r"(?:date|dated)\s*[:\-]?\s*(\d{4}[\-\/]\d{2}[\-\/]\d{2}|\d{2}[\-\/]\d{2}[\-\/]\d{4})",
        "tax_id": r"(?:tax\s*id|vat|gstin)\s*[:\-]?\s*([A-Z0-9]+)"
    }

    def extract_fields(self, full_text: str) -> Dict[str, Any]:
        """
        Extracts key-value fields from document text using regex rules.

        Args:
            full_text (str): Consolidated OCR text prediction string.

        Returns:
            Dict[str, Any]: Dictionary mapping field name to extracted string or None.
        """
        extracted = {}
        for field_name, pattern in self.PATTERNS.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                extracted[field_name] = match.group(1).strip()
            else:
                extracted[field_name] = None

        logger.debug(f"Extracted document fields: {extracted}")
        return extracted
