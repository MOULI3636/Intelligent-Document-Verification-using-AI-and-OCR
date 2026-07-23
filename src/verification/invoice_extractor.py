"""
Invoice Information Extractor Strategy Concrete Implementation.

Extracts key structural invoice entities:
1. Invoice Number
2. Vendor Name
3. Date
4. GST (GSTIN / Tax ID)
5. Total Amount
6. Address
7. Customer Name (Billed To)
"""

import re
from typing import Dict, List, Any, Optional
try:
    from src.verification.base_extractor import AbstractFieldExtractor, ExtractedField, DocumentExtractionResult
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .base_extractor import AbstractFieldExtractor, ExtractedField, DocumentExtractionResult
    from ..utils.logger import get_logger

logger = get_logger("InvoiceFieldExtractor")


class InvoiceFieldExtractor(AbstractFieldExtractor):
    """
    Regex and spatial rule-based Invoice Field Extractor.
    """

    PATTERNS = {
        "invoice_number": r"(?:invoice\s*no|invoice\s*num|invoice\s*#|inv\s*#|bill\s*#)\s*[:\-]?\s*([A-Z0-9\-\/]+)",
        "date": r"(?:date|dated|invoice\s*date)\s*[:\-]?\s*(\d{4}[\-\/]\d{2}[\-\/]\d{2}|\d{2}[\-\/]\d{2}[\-\/]\d{4})",
        "gst": r"(?:gstin|gst\s*no|vat\s*no|tax\s*id)\s*[:\-]?\s*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}|[A-Z0-9\-]+)",
        "total_amount": r"(?:total\s*amount|total\s*due|grand\s*total|net\s*payable|amount\s*payable|total)\s*[:\-]?\s*[\$₹]?\s*([\d,]+\.\d{2}|\d+)",
        "customer": r"(?:bill\s*to|customer|buyer|client)\s*[:\-]?\s*([A-Z0-9\s\,\.\-]+)"
    }

    def __init__(self) -> None:
        super().__init__(document_type="Invoice")

    def _extract_vendor(self, lines: List[str]) -> Optional[str]:
        """Header line usually contains vendor name."""
        for line in lines[:3]:
            cleaned = line.strip()
            if cleaned and not re.search(r"invoice|tax|date|bill|page", cleaned, re.IGNORECASE):
                if len(cleaned) > 2:
                    return cleaned
        return None

    def _extract_address(self, lines: List[str]) -> Optional[str]:
        """Extracts address strings using zip/pin code patterns or street keywords."""
        address_parts = []
        for line in lines:
            if re.search(r"street|road|st\.|ave|building|pin|zip|\d{5,6}", line, re.IGNORECASE):
                address_parts.append(line.strip())
        return ", ".join(address_parts[:2]) if address_parts else None

    def extract(self, ocr_text: str, ocr_results: Optional[List[Any]] = None) -> DocumentExtractionResult:
        """
        Executes field extraction over invoice OCR text.

        Args:
            ocr_text (str): Input OCR text string.
            ocr_results (Optional[List[Any]]): Optional list of OCRResult objects.

        Returns:
            DocumentExtractionResult: Extracted fields payload.
        """
        lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
        if not lines and ocr_text.strip():
            lines = [ocr_text.strip()]

        extracted_fields: Dict[str, ExtractedField] = {}

        # 1. Regex Pattern Extractions
        for field_name, pattern in self.PATTERNS.items():
            match = re.search(pattern, ocr_text, re.IGNORECASE)
            val = match.group(1).strip() if match else None
            conf = 0.95 if val else 0.0
            extracted_fields[field_name] = ExtractedField(field_name=field_name, value=val, confidence=conf)

        # 2. Vendor Name Extraction
        vendor_val = self._extract_vendor(lines)
        extracted_fields["vendor"] = ExtractedField(field_name="vendor", value=vendor_val, confidence=0.85 if vendor_val else 0.0)

        # 3. Address Extraction
        address_val = self._extract_address(lines)
        extracted_fields["address"] = ExtractedField(field_name="address", value=address_val, confidence=0.80 if address_val else 0.0)

        logger.info(f"Invoice Extraction Complete -> Fields Found: {sum(1 for f in extracted_fields.values() if f.value)}")

        return DocumentExtractionResult(
            document_type=self.document_type,
            fields=extracted_fields,
            raw_text=ocr_text
        )
