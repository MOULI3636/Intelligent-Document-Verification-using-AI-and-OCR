"""
Multi-Engine Research Benchmarker and Evaluator.

Executes comparative evaluation across multiple OCR engines on test document datasets.
Generates structured Pandas dataframes, latency statistics, and error summaries.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from src.engines.base_engine import AbstractOCREngine, OCRResult
from src.metrics.ocr_metrics import calculate_cer, calculate_wer, calculate_exact_match, calculate_levenshtein
from src.utils.logger import get_logger

logger = get_logger("OCREvaluator")


class OCREvaluator:
    """
    Evaluator executing comparative benchmark experiments across multiple registered OCR engines.
    """

    def __init__(self, engines: List[AbstractOCREngine]) -> None:
        """
        Args:
            engines (List[AbstractOCREngine]): List of OCR engine instances to evaluate.
        """
        self.engines = [e for e in engines if e.is_available()]
        logger.info(f"Initialized OCREvaluator with {len(self.engines)} active engines: {[e.engine_name for e in self.engines]}")

    def evaluate_sample(
        self,
        image: np.ndarray,
        ground_truth: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates all registered engines on a single document sample.

        Args:
            image (np.ndarray): Document image array.
            ground_truth (str): Ground truth reference text.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping engine name to individual sample metrics.
        """
        sample_results = {}

        for engine in self.engines:
            ocr_results: List[OCRResult] = engine.recognize(image)
            predicted_text = " ".join([res.text for res in ocr_results]).strip()
            total_latency = sum([res.inference_time_ms for res in ocr_results])

            cer = calculate_cer([predicted_text], [ground_truth])
            wer = calculate_wer([predicted_text], [ground_truth])
            edit_dist = calculate_levenshtein(predicted_text, ground_truth)

            sample_results[engine.engine_name] = {
                "prediction": predicted_text,
                "ground_truth": ground_truth,
                "cer": cer,
                "wer": wer,
                "edit_dist": edit_dist,
                "latency_ms": total_latency,
                "num_bboxes": len(ocr_results)
            }

        return sample_results

    def benchmark_dataset(
        self,
        dataset_samples: List[Tuple[np.ndarray, str]]
    ) -> pd.DataFrame:
        """
        Runs batch evaluation over dataset samples and builds summary benchmark report.

        Args:
            dataset_samples (List[Tuple[np.ndarray, str]]): List of (image, ground_truth) pairs.

        Returns:
            pd.DataFrame: Summary table comparing Mean CER, Mean WER, Mean Latency, and Exact Match.
        """
        logger.info(f"Running benchmark evaluation over {len(dataset_samples)} dataset samples...")
        engine_stats: Dict[str, Dict[str, List[float]]] = {
            e.engine_name: {"cer": [], "wer": [], "latency": [], "exact_match": []} for e in self.engines
        }

        for idx, (img, gt) in enumerate(dataset_samples):
            res = self.evaluate_sample(img, gt)
            for eng_name, stats in res.items():
                engine_stats[eng_name]["cer"].append(stats["cer"])
                engine_stats[eng_name]["wer"].append(stats["wer"])
                engine_stats[eng_name]["latency"].append(stats["latency_ms"])
                engine_stats[eng_name]["exact_match"].append(1.0 if stats["prediction"] == gt else 0.0)

        rows = []
        for eng_name, stats in engine_stats.items():
            rows.append({
                "Engine": eng_name,
                "Mean CER": np.mean(stats["cer"]) if stats["cer"] else 0.0,
                "Mean WER": np.mean(stats["wer"]) if stats["wer"] else 0.0,
                "Exact Match %": np.mean(stats["exact_match"]) * 100.0 if stats["exact_match"] else 0.0,
                "Mean Latency (ms)": np.mean(stats["latency"]) if stats["latency"] else 0.0
            })

        df_summary = pd.DataFrame(rows)
        logger.info("Dataset Benchmark Completed.")
        return df_summary
