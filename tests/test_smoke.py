"""
Smoke Tests for DocVision AI.
"""

import pytest
import src
from src.utils.logger import get_logger
from src.utils.config_parser import load_config


def test_package_import():
    assert src.__version__ == "1.0.0"


def test_logger_initialization():
    logger = get_logger("SmokeTestLogger")
    assert logger is not None


def test_config_loading():
    config = load_config("config.yaml")
    assert config.project_name == "DocVision AI"
    assert config.preprocessing.target_height == 32
