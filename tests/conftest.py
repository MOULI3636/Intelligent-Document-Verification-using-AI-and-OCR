"""
Pytest Fixtures for DocVision AI Test Suite.
"""

import cv2
import numpy as np
import pytest
import torch

from src.data.dataset import VocabularyEncoder
from src.utils.config_parser import load_config, ConfigSchema


@pytest.fixture
def sample_config() -> ConfigSchema:
    return load_config("config.yaml")


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generates synthetic 100x400 BGR document image."""
    img = np.full((100, 400, 3), 250, dtype=np.uint8)
    cv2.putText(img, "INVOICE NUMBER: #98421", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    return img


@pytest.fixture
def sample_grayscale_image(sample_image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def vocab_encoder() -> VocabularyEncoder:
    return VocabularyEncoder("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")
