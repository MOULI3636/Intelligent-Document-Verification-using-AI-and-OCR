"""
Training Execution Entry Point Script for DocAI-Bench.

Usage:
    python train.py --config configs/default_config.yaml
"""

import argparse
import os
from typing import Dict, List, Any, Optional
import torch

from src.data.dataset import DocumentOCRDataset, VocabularyEncoder
from src.data.dataloader import create_dataloader
from src.models.crnn_model import CRNNModel
from src.preprocessing.augmentations import DocumentAugmentationPipeline
from src.preprocessing.image_processor import DocumentImageProcessor
from src.training.trainer import DocumentTrainer
from src.utils.config_parser import load_config
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

logger = get_logger("TrainScript")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocAI-Bench Fine-Tuning Execution Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    seed_everything(args.seed)
    logger.info("Initializing DocAI-Bench Training Engine...")
    config = load_config(args.config)

    # Initialize vocabulary encoder
    vocab_str = config.raw_dict.get("dataset", {}).get(
        "vocabulary",
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    )
    vocab = VocabularyEncoder(vocab_str)
    logger.info(f"Vocabulary initialized with {len(vocab)} character tokens.")

    # Initialize preprocessor and augmentation pipeline
    processor = DocumentImageProcessor(
        target_height=config.preprocessing.target_height,
        target_width=config.preprocessing.target_width,
        grayscale=config.preprocessing.grayscale,
        binarization_method=config.preprocessing.binarization_method
    )
    aug_pipeline = DocumentAugmentationPipeline() if config.raw_dict.get("augmentation", {}).get("enabled", True) else None

    # Construct synthetic research samples if raw dataset paths do not exist
    dummy_samples = [
        ("data/synthetic_sample_1.png", "INVOICE #98421"),
        ("data/synthetic_sample_2.png", "TOTAL: $1,250.00"),
        ("data/synthetic_sample_3.png", "DATE: 2026-07-22"),
        ("data/synthetic_sample_4.png", "TAX ID: US9876543"),
        ("data/synthetic_sample_5.png", "HyperVerge AI Research"),
        ("data/synthetic_sample_6.png", "Deep Learning OCR Suite"),
    ] * 8

    train_samples = dummy_samples[:40]
    val_samples = dummy_samples[40:]

    train_dataset = DocumentOCRDataset(train_samples, vocab, processor, transform=aug_pipeline)
    val_dataset = DocumentOCRDataset(val_samples, vocab, processor, transform=None)

    train_loader = create_dataloader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    # Build CRNN neural network model
    model = CRNNModel(
        num_classes=len(vocab),
        in_channels=1 if config.preprocessing.grayscale else 3,
        lstm_hidden_size=256
    )

    trainer = DocumentTrainer(
        model=model,
        vocab_encoder=vocab,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config.device
    )

    trainer.fit()


if __name__ == "__main__":
    main()
