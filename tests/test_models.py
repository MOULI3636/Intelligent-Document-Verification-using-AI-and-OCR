"""
Neural Network Models PyTorch Unit Tests.
"""

import pytest
import torch

from src.models.crnn_model import CRNNModel
from src.models.efficientnet_classifier import EfficientNetDocumentClassifier
from src.models.resnet_forgery_detector import ResNetForgeryDetector


def test_crnn_model_forward():
    model = CRNNModel(num_classes=80, in_channels=1)
    dummy_input = torch.randn(2, 1, 32, 128) # Batch size 2, 1 channel, 32x128
    logits = model(dummy_input)
    # Output logits shape: [Sequence_Length, Batch, Num_Classes]
    assert logits.shape[1] == 2
    assert logits.shape[2] == 80


def test_efficientnet_classifier_forward():
    model = EfficientNetDocumentClassifier(num_classes=7, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 7)

    top_class, top_prob, dist = model.predict_class(dummy_input[0])
    assert top_class in model.class_names
    assert 0.0 <= top_prob <= 1.0


def test_resnet_forgery_detector_forward():
    model = ResNetForgeryDetector(num_classes=6, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 6)

    top_class, top_prob, dist = model.predict(dummy_input[0])
    assert top_class in model.class_names
    assert 0.0 <= top_prob <= 1.0
