"""
DocVision AI Neural Network Architectures, Classifiers, & Forgery Detectors Subpackage.
"""

from src.models.crnn_model import CRNNModel
from src.models.hf_wrapper import HuggingFaceTrOCRWrapper
from src.models.document_classifier import DocumentClassifier
from src.models.efficientnet_classifier import EfficientNetDocumentClassifier
from src.models.resnet_forgery_detector import ResNetForgeryDetector

__all__ = [
    "CRNNModel",
    "HuggingFaceTrOCRWrapper",
    "DocumentClassifier",
    "EfficientNetDocumentClassifier",
    "ResNetForgeryDetector",
]
