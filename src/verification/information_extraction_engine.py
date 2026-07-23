"""
Master Information Extraction Engine for DocVision AI.

Orchestrates entity extraction by dynamically resolving document type strategies
(Invoices, Receipts, Passports, PAN, Aadhaar, Driving Licenses, etc.).

Outputs clean, structured JSON payloads.
"""

from typing import Dict, List, Any, Optional, Union
import json

try:
    from src.verification.base_extractor import AbstractFieldExtractor, DocumentExtractionResult
    from src.verification.identity_extractor import IdentityDocumentExtractor
    from src.verification.invoice_extractor import InvoiceFieldExtractor
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .base_extractor import AbstractFieldExtractor, DocumentExtractionResult
    from .identity_extractor import IdentityDocumentExtractor
    from .invoice_extractor import InvoiceFieldExtractor
    from ..utils.logger import get_logger

logger = get_logger("InformationExtractionEngine")


class InformationExtractionEngine:
    """
    Extensible Master Information Extraction Engine.
    Follows Strategy Pattern for dynamic document type routing.
    """

    def __init__(self) -> None:
        self.strategies: Dict[str, AbstractFieldExtractor] = {}
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        self.register_strategy("invoice", InvoiceFieldExtractor())
        self.register_strategy("receipt", InvoiceFieldExtractor()) # Uses invoice schema for receipts
        self.register_strategy("passport", IdentityDocumentExtractor(card_type="passport"))
        self.register_strategy("pan_card", IdentityDocumentExtractor(card_type="pan_card"))
        self.register_strategy("pan", IdentityDocumentExtractor(card_type="pan_card"))
        self.register_strategy("aadhaar_card", IdentityDocumentExtractor(card_type="aadhaar_card"))
        self.register_strategy("aadhaar", IdentityDocumentExtractor(card_type="aadhaar_card"))
        self.register_strategy("driving_license", IdentityDocumentExtractor(card_type="driving_license"))

    def register_strategy(self, doc_type: str, strategy: AbstractFieldExtractor) -> None:
        """
        Registers a new extraction strategy for a document category (Open/Closed Principle).

        Args:
            doc_type (str): Document category key string.
            strategy (AbstractFieldExtractor): Strategy implementation instance.
        """
        key = doc_type.lower()
        self.strategies[key] = strategy
        logger.info(f"Registered extraction strategy for document type: '{key}'")

    def extract_information(
        self,
        ocr_text: str,
        document_category: str = "invoice",
        ocr_results: Optional[List[Any]] = None
    ) -> DocumentExtractionResult:
        """
        Executes information extraction based on document category strategy.

        Args:
            ocr_text (str): Input OCR text string.
            document_category (str): Target document type (e.g. 'invoice', 'passport').
            ocr_results (Optional[List[Any]]): Optional list of OCRResult objects.

        Returns:
            DocumentExtractionResult: Structured entity payload.
        """
        key = document_category.lower().replace(" ", "_")
        strategy = self.strategies.get(key, self.strategies.get("invoice"))

        if strategy is None:
            strategy = InvoiceFieldExtractor()

        logger.info(f"Extracting information using strategy: '{strategy.document_type}'")
        return strategy.extract(ocr_text, ocr_results)

    def extract_to_json(
        self,
        ocr_text: str,
        document_category: str = "invoice",
        indent: int = 4
    ) -> str:
        """
        Executes information extraction and returns structured JSON string.

        Args:
            ocr_text (str): Input OCR text string.
            document_category (str): Target document category string.
            indent (int): JSON formatting indent space.

        Returns:
            str: Pretty-printed JSON formatted string.
        """
        result = self.extract_information(ocr_text, document_category=document_category)
        return result.to_json(indent=indent)
