"""
Extensible OCR Benchmarking Framework Execution Script.

Usage:
    python evaluate.py --config config.yaml
"""

import argparse
import os
import cv2
import numpy as np

from src.engines.easyocr_engine import EasyOCREngine
from src.engines.paddleocr_engine import PaddleOCREngine
from src.engines.trocr_engine import TrOCREngine
from src.metrics.ocr_benchmark_framework import OCRBenchmarkingFramework
from src.utils.config_parser import load_config
from src.utils.logger import get_logger

logger = get_logger("EvaluateScript")


def create_synthetic_test_image(text: str) -> np.ndarray:
    """Generates synthetic text line image patch for benchmark testing."""
    img = np.full((60, 450, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (447, 57), (200, 200, 200), 2)
    cv2.putText(img, text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="DocVision AI Extensible OCR Benchmarking Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    logger.info("Initializing OCR Benchmarking Framework...")

    engines = [
        EasyOCREngine(gpu=False),
        PaddleOCREngine(),
        TrOCREngine(device="cpu")
    ]

    framework = OCRBenchmarkingFramework(engines=engines)

    test_image = create_synthetic_test_image("INVOICE NUMBER: #98421-2026 TOTAL: $1,450.00")
    reference_text = "INVOICE NUMBER: #98421-2026 TOTAL: $1,450.00"

    results = framework.benchmark_image(test_image, reference_text=reference_text)
    df_results = framework.to_dataframe(results)

    print("\n" + "="*80)
    print("                 DOCVISION AI OCR BENCHMARK REPORT")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80 + "\n")

    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "ocr_benchmark.json")

    framework.export_json(results, json_path)


if __name__ == "__main__":
    main()
