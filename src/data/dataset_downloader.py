"""
Dataset Downloader and Benchmark Generator Module.

Fetches open-source OCR benchmark datasets (SROIE, FUNSD, ICDAR) and generates
synthetic sample data for Indian & Global Identity Documents (PAN, Aadhaar, Passport, Driving License, Invoices, Receipts).
"""

import os
import urllib.request
import zipfile
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("DatasetDownloader")


class DatasetDownloader:
    """
    Downloads open-source document datasets and builds synthetic samples for testing.
    """

    BENCHMARK_URLS = {
        "sroie_receipts": "https://github.com/ocr-benchmark/sroie-sample/archive/refs/heads/main.zip",
    }

    DOCUMENT_TYPES = [
        "invoice",
        "receipt",
        "passport",
        "pan_card",
        "aadhaar_card",
        "driving_license"
    ]

    def __init__(self, download_dir: str = "data/raw") -> None:
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def download_file(self, url: str, destination: str) -> bool:
        """Downloads file over HTTP with progress logging."""
        try:
            logger.info(f"Downloading benchmark dataset from '{url}'...")
            urllib.request.urlretrieve(url, destination)
            logger.info(f"Successfully downloaded file to '{destination}'")
            return True
        except Exception as e:
            logger.warning(f"Failed to download from '{url}': {str(e)}")
            return False

    def generate_synthetic_identity_samples(
        self,
        output_dir: str,
        num_per_category: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generates synthetic document patches for Passports, PAN, Aadhaar, Driving Licenses, Invoices, & Receipts.

        Args:
            output_dir (str): Destination directory for generated synthetic dataset.
            num_per_category (int): Number of synthetic image-text pairs per category.

        Returns:
            List[Dict[str, Any]]: List of sample dictionaries.
        """
        os.makedirs(output_dir, exist_ok=True)
        synthetic_samples = []

        category_templates = {
            "invoice": ["INVOICE #INV-98421", "TOTAL DUE: $1,250.00", "TAX ID: US9876543"],
            "receipt": ["STARBUCKS #1042", "TOTAL: $14.50", "CASH PAYMENT RECEIVED"],
            "passport": ["PASSPORT P<IND<KUMAR<<AMIT<<<<<<<", "P1234567 IND 9508124 M"],
            "pan_card": ["INCOME TAX DEPARTMENT", "ABCDE1234F - PERMANENT ACCOUNT NUMBER"],
            "aadhaar_card": ["GOVERNMENT OF INDIA", "9876 5432 1098 - AADHAAR"],
            "driving_license": ["DL-1420110012345", "DRIVING LICENSE - CLASS LMV"]
        }

        for category in self.DOCUMENT_TYPES:
            cat_dir = os.path.join(output_dir, category)
            os.makedirs(cat_dir, exist_ok=True)

            templates = category_templates.get(category, ["DOCUMENT SAMPLE"])

            for idx in range(num_per_category):
                text_label = templates[idx % len(templates)]
                file_name = f"{category}_synth_{idx+1:03d}.png"
                file_path = os.path.join(cat_dir, file_name)

                # Render image patch
                img = np.full((60, 450, 3), 245, dtype=np.uint8)
                cv2.rectangle(img, (2, 2), (447, 57), (200, 200, 200), 2)
                cv2.putText(
                    img, text_label, (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA
                )

                cv2.imwrite(file_path, img)

                synthetic_samples.append({
                    "image_path": file_path,
                    "text": text_label,
                    "category": category
                })

        logger.info(f"Generated {len(synthetic_samples)} synthetic document samples across {len(self.DOCUMENT_TYPES)} categories.")
        return synthetic_samples
