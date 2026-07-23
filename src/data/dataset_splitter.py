"""
Stratified Dataset Partitioner Module.

Splits dataset samples into train, validation, and test subsets while preserving
exact document category ratios (Invoices, Receipts, Passports, PAN, Aadhaar, Driving License).
"""

import random
from typing import Dict, List, Tuple, Any
from src.utils.logger import get_logger

logger = get_logger("DatasetSplitter")


class DatasetSplitter:
    """
    Performs stratified partitioning into Train, Validation, and Test subsets.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> None:
        """
        Args:
            train_ratio (float): Fraction of data for training.
            val_ratio (float): Fraction of data for validation.
            test_ratio (float): Fraction of data for testing.
            seed (int): Random seed for reproducibility.
        """
        total = train_ratio + val_ratio + test_ratio
        self.train_ratio = train_ratio / total
        self.val_ratio = val_ratio / total
        self.test_ratio = test_ratio / total
        self.seed = seed

    def split_samples(
        self,
        samples: List[Tuple[str, str, str]]
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Performs category-stratified dataset split.

        Args:
            samples (List[Tuple[str, str, str]]): List of (image_path, text_label, category) tuples.

        Returns:
            Dict[str, List[Tuple[str, str, str]]]: Dictionary mapping 'train', 'val', and 'test' to sample lists.
        """
        random.seed(self.seed)

        # Group samples by category
        category_map: Dict[str, List[Tuple[str, str, str]]] = {}
        for item in samples:
            cat = item[2]
            category_map.setdefault(cat, []).append(item)

        train_set, val_set, test_set = [], [], []

        for cat, cat_samples in category_map.items():
            random.shuffle(cat_samples)
            n_total = len(cat_samples)

            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)

            train_subset = cat_samples[:n_train]
            val_subset = cat_samples[n_train:n_train + n_val]
            test_subset = cat_samples[n_train + n_val:]

            train_set.extend(train_subset)
            val_set.extend(val_subset)
            test_set.extend(test_subset)

            logger.info(f"Category '{cat}' -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set
        }
