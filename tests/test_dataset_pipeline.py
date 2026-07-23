"""
Dataset Pipeline Tests for DocVision AI.
"""

import os
import pytest
from src.data.dataset import VocabularyEncoder, DocumentOCRDataset
from src.data.dataset_validator import DatasetValidator
from src.data.dataset_splitter import DatasetSplitter
from src.data.custom_dataset_builder import CustomDatasetBuilder


def test_vocabulary_encoder(vocab_encoder):
    encoded = vocab_encoder.encode("INVOICE #98421")
    assert isinstance(encoded, list)
    decoded = vocab_encoder.decode(encoded)
    assert decoded == "INVOICE #98421"


def test_dataset_validator(tmp_path):
    validator = DatasetValidator()
    # Non-existent file test
    is_valid, msg = validator.validate_image_file(str(tmp_path / "non_existent.png"))
    assert not is_valid
    assert "File does not exist" in msg


def test_dataset_splitter():
    splitter = DatasetSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    samples = [
        ("path/img1.png", "Text 1", "invoice"),
        ("path/img2.png", "Text 2", "invoice"),
        ("path/img3.png", "Text 3", "invoice"),
        ("path/img4.png", "Text 4", "passport"),
        ("path/img5.png", "Text 5", "passport"),
        ("path/img6.png", "Text 6", "passport"),
    ]
    splits = splitter.split_samples(samples)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits


def test_custom_dataset_builder(tmp_path):
    builder = CustomDatasetBuilder(base_dir=str(tmp_path))
    res = builder.build_pipeline(generate_synthetic=True)
    assert res["status"] == "success"
    assert os.path.exists(res["metadata_path"])
