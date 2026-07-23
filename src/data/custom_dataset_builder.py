"""
Unified Custom Dataset Pipeline Builder.

Orchestrates downloading open-source/synthetic datasets, ingesting custom user directories,
validating image & label integrity, performing category-stratified splits, generating analytics,
and saving metadata JSON records for Invoices, Receipts, Passports, PAN, Aadhaar, and Driving Licenses.
"""

import os
from typing import Dict, List, Tuple, Any, Optional

from src.data.dataset_analyzer import DatasetAnalyzer
from src.data.dataset_cache import DatasetCache
from src.data.dataset_downloader import DatasetDownloader
from src.data.dataset_splitter import DatasetSplitter
from src.data.dataset_validator import DatasetValidator
from src.utils.io_utils import save_json, load_json
from src.utils.logger import get_logger

logger = get_logger("CustomDatasetBuilder")


class CustomDatasetBuilder:
    """
    Unified end-to-end dataset pipeline builder.
    """

    def __init__(
        self,
        base_dir: str = "data",
        min_width: int = 32,
        min_height: int = 16,
        seed: int = 42
    ) -> None:
        self.base_dir = base_dir
        self.downloader = DatasetDownloader(download_dir=os.path.join(base_dir, "raw"))
        self.validator = DatasetValidator(min_width=min_width, min_height=min_height)
        self.splitter = DatasetSplitter(seed=seed)
        self.cache = DatasetCache(cache_dir=os.path.join(base_dir, "cache"))
        self.analyzer = DatasetAnalyzer()

    def register_custom_dataset(
        self,
        custom_dir: str,
        category: str = "generic"
    ) -> List[Dict[str, Any]]:
        """
        Scans a custom user directory for image-text pairs.
        Expects either paired `.txt` files alongside images, or a `labels.json` manifest.

        Args:
            custom_dir (str): Path to user's dataset directory.
            category (str): Document category (e.g. 'invoice', 'passport', 'pan_card').

        Returns:
            List[Dict[str, Any]]: List of sample dictionaries.
        """
        if not os.path.exists(custom_dir):
            logger.warning(f"Custom directory '{custom_dir}' does not exist.")
            return []

        manifest_path = os.path.join(custom_dir, "labels.json")
        samples = []

        if os.path.exists(manifest_path):
            manifest_data = load_json(manifest_path)
            for item in manifest_data:
                samples.append({
                    "image_path": os.path.join(custom_dir, item.get("image", "")),
                    "text": item.get("text", ""),
                    "category": item.get("category", category)
                })
        else:
            # Pair image files with text files of same basename
            for fname in os.listdir(custom_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.validator.allowed_extensions:
                    base_name = os.path.splitext(fname)[0]
                    txt_path = os.path.join(custom_dir, f"{base_name}.txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            text_label = f.read().strip()
                        samples.append({
                            "image_path": os.path.join(custom_dir, fname),
                            "text": text_label,
                            "category": category
                        })

        logger.info(f"Registered {len(samples)} custom samples from '{custom_dir}' under category '{category}'.")
        return samples

    def build_pipeline(
        self,
        custom_dirs: Optional[List[Tuple[str, str]]] = None,
        generate_synthetic: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end dataset pipeline:
        1. Downloads/generates synthetic identity & invoice samples.
        2. Ingests user custom dataset directories.
        3. Validates image corruption & annotation schema.
        4. Performs category-stratified train/val/test splitting.
        5. Computes dataset statistics & plots analysis charts.
        6. Stores dataset metadata JSON records.

        Returns:
            Dict[str, Any]: Pipeline metadata execution report.
        """
        all_raw_samples = []

        # 1. Generate synthetic baseline samples for all identity document categories
        if generate_synthetic:
            synth_samples = self.downloader.generate_synthetic_identity_samples(
                output_dir=os.path.join(self.base_dir, "synthetic"),
                num_per_category=10
            )
            all_raw_samples.extend(synth_samples)

        # 2. Ingest custom datasets
        if custom_dirs:
            for dir_path, cat_name in custom_dirs:
                custom_samples = self.register_custom_dataset(dir_path, category=cat_name)
                all_raw_samples.extend(custom_samples)

        # 3. Validate datasets
        validation_report = self.validator.validate_dataset_samples(all_raw_samples)
        valid_samples = validation_report.valid_sample_paths

        if not valid_samples:
            logger.error("No valid dataset samples found to build pipeline.")
            return {"status": "failed", "reason": "No valid samples"}

        # 4. Stratified split
        split_data = self.splitter.split_samples(valid_samples)

        # 5. Calculate statistics
        stats = self.analyzer.analyze_samples(valid_samples)
        self.analyzer.plot_statistics(stats, output_path=os.path.join(self.base_dir, "dataset_stats.png"))

        # 6. Store metadata
        metadata_payload = {
            "total_samples": len(valid_samples),
            "split_counts": {k: len(v) for k, v in split_data.items()},
            "statistics": stats,
            "supported_document_types": list(DatasetDownloader.DOCUMENT_TYPES),
            "splits": {
                split_name: [
                    {"image_path": item[0], "text": item[1], "category": item[2]}
                    for item in items
                ]
                for split_name, items in split_data.items()
            }
        }

        metadata_path = os.path.join(self.base_dir, "dataset_metadata.json")
        save_json(metadata_payload, metadata_path)
        logger.info(f"Saved complete dataset pipeline metadata to: {metadata_path}")

        return {
            "status": "success",
            "metadata_path": metadata_path,
            "summary": stats
        }
