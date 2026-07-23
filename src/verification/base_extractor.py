"""
Abstract Base Information Extractor Interface.

Follows SOLID Open/Closed Principles to define a unified contract for extracting
structured key-value entities from diverse document categories (Invoices, Receipts, Identity Cards).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
import json


@dataclass
class ExtractedField:
    """
    Standardized dataclass representing single extracted entity field.

    Attributes:
        field_name (str): Identifier of the field (e.g. 'invoice_number', 'vendor').
        value (Optional[str]): Extracted value text string or None if unextracted.
        confidence (float): Extraction confidence score (0.0 - 1.0).
        bbox (Optional[List[Tuple[int, int]]]): Optional spatial bounding box.
    """
    field_name: str
    value: Optional[str]
    confidence: float = 1.0
    bbox: Optional[List[Any]] = None


@dataclass
class DocumentExtractionResult:
    """
    Structured document extraction payload.
    """
    document_type: str
    fields: Dict[str, ExtractedField]
    raw_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts extraction payload to clean dictionary."""
        return {
            "document_type": self.document_type,
            "fields": {
                name: f.value for name, f in self.fields.items()
            },
            "detailed_fields": {
                name: asdict(f) for name, f in self.fields.items()
            },
            "raw_text": self.raw_text
        }

    def to_json(self, indent: int = 4) -> str:
        """Serializes extraction payload to structured JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class AbstractFieldExtractor(ABC):
    """
    Abstract Base Class for all document field extraction strategies.
    """

    def __init__(self, document_type: str) -> None:
        self.document_type = document_type

    @abstractmethod
    def extract(self, ocr_text: str, ocr_results: Optional[List[Any]] = None) -> DocumentExtractionResult:
        """
        Extracts key structural entities from OCR predictions.

        Args:
            ocr_text (str): Consolidated OCR text prediction string.
            ocr_results (Optional[List[Any]]): Optional list of OCRResult objects with bounding boxes.

        Returns:
            DocumentExtractionResult: Structured entity payload.
        """
        pass
