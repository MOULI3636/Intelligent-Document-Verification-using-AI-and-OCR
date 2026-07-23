"""
Dataset Pipeline Execution Script for DocVision AI.

Ingests, downloads, validates, splits, caches, and analyzes document datasets across:
- Invoices
- Receipts
- Identity Documents (Passport, PAN Card, Aadhaar Card, Driving License)

Usage:
    python build_dataset.py
"""

import argparse
from src.data.custom_dataset_builder import CustomDatasetBuilder
from src.utils.logger import get_logger

logger = get_logger("BuildDatasetScript")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocVision AI Dataset Pipeline Execution Script")
    parser.add_argument("--base-dir", type=str, default="data", help="Target dataset root directory")
    args = parser.parse_args()

    logger.info("Initializing DocVision AI Complete Dataset Pipeline...")

    builder = CustomDatasetBuilder(base_dir=args.base_dir)
    res = builder.build_pipeline(generate_synthetic=True)

    print("\n" + "="*70)
    print("            DOCVISION AI DATASET PIPELINE BUILD REPORT")
    print("="*70)
    print(f"Status: {res.get('status', 'unknown')}")
    print(f"Metadata Saved To: {res.get('metadata_path', 'N/A')}")
    summary = res.get("summary", {})
    print(f"Total Valid Samples: {summary.get('total_samples', 0)}")
    print(f"Document Categories: {summary.get('category_distribution', {})}")
    print(f"Vocabulary Size: {summary.get('vocabulary_size', 0)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
