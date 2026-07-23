"""
DocVision AI Data Pipeline Subpackage.

Provides end-to-end dataset management: downloading, custom dataset registration,
validation, category-stratified splitting, on-disk caching, analytics, and PyTorch dataloaders.
"""

from src.data.dataset import DocumentOCRDataset, VocabularyEncoder
from src.data.dataloader import create_dataloader, OCRCollate
from src.data.dataset_validator import DatasetValidator, ValidationReport
from src.data.dataset_downloader import DatasetDownloader
from src.data.dataset_splitter import DatasetSplitter
from src.data.dataset_cache import DatasetCache
from src.data.dataset_analyzer import DatasetAnalyzer
from src.data.custom_dataset_builder import CustomDatasetBuilder

__all__ = [
    "DocumentOCRDataset",
    "VocabularyEncoder",
    "create_dataloader",
    "OCRCollate",
    "DatasetValidator",
    "ValidationReport",
    "DatasetDownloader",
    "DatasetSplitter",
    "DatasetCache",
    "DatasetAnalyzer",
    "CustomDatasetBuilder",
]
