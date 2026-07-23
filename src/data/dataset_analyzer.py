"""
Dataset Analytics and Statistics Generator Module.

Computes document category distributions, resolution statistics, character sequence lengths,
and exports graphical summaries for research monitoring.
"""

from typing import Dict, List, Tuple, Any
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.io_utils import save_json
from src.utils.logger import get_logger

logger = get_logger("DatasetAnalyzer")


class DatasetAnalyzer:
    """
    Computes statistical analysis over dataset splits and samples.
    """

    def analyze_samples(
        self,
        samples: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """
        Calculates category breakdown, mean resolutions, and character length stats.

        Args:
            samples (List[Tuple[str, str, str]]): List of (image_path, text_label, category) tuples.

        Returns:
            Dict[str, Any]: Statistical metrics dictionary.
        """
        category_counts: Dict[str, int] = {}
        text_lengths: List[int] = []
        widths: List[int] = []
        heights: List[int] = []
        vocab_set = set()

        for img_path, text, category in samples:
            category_counts[category] = category_counts.get(category, 0) + 1
            text_lengths.append(len(text))
            vocab_set.update(list(text))

            if cv2.imread(img_path) is not None:
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)

        stats = {
            "total_samples": len(samples),
            "category_distribution": category_counts,
            "vocabulary_size": len(vocab_set),
            "mean_text_length": float(np.mean(text_lengths)) if text_lengths else 0.0,
            "max_text_length": int(np.max(text_lengths)) if text_lengths else 0,
            "min_text_length": int(np.min(text_lengths)) if text_lengths else 0,
            "mean_width": float(np.mean(widths)) if widths else 0.0,
            "mean_height": float(np.mean(heights)) if heights else 0.0
        }

        logger.info(f"Dataset Analytics -> Total: {stats['total_samples']} | Categories: {category_counts} | Vocab Size: {stats['vocabulary_size']}")
        return stats

    def plot_statistics(self, stats: Dict[str, Any], output_path: str = "evaluation_results/dataset_stats.png") -> None:
        """Renders graphical bar charts of category counts and exports image."""
        fig, ax = plt.subplots(figsize=(10, 5))

        cats = list(stats["category_distribution"].keys())
        counts = list(stats["category_distribution"].values())

        ax.bar(cats, counts, color="#6366F1")
        ax.set_title("DocVision AI Dataset Category Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Document Category", fontsize=12)
        ax.set_ylabel("Sample Count", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved dataset analytics plot to: {output_path}")
