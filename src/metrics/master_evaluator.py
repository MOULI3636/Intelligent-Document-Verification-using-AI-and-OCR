"""
Master Evaluation Framework for DocVision AI.

Computes complete multi-dimensional metrics:
- Accuracy, Precision, Recall, F1 (Macro & Micro)
- ROC Curve & Area Under Curve (AUC)
- Execution Latency percentiles (mean, p50, p95, p99) & Throughput (FPS)
- RAM Memory usage & PyTorch CUDA GPU VRAM peak allocation
- Renders & saves Confusion Matrix, ROC Curve, Precision-Recall Curve, and Training Curves
- Automatically persists timestamped experiment summaries via ExperimentTracker.
"""

from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from src.metrics.experiment_tracker import ExperimentTracker
from src.metrics.system_monitor import SystemResourceMonitor, HardwareProfile
from src.utils.logger import get_logger

logger = get_logger("MasterEvaluationFramework")


class MasterEvaluationFramework:
    """
    Master Evaluation Engine conducting metrics calculation, hardware profiling, chart rendering, and automated experiment tracking.
    """

    def __init__(self, experiment_name: Optional[str] = None) -> None:
        self.tracker = ExperimentTracker(experiment_name=experiment_name)
        self.monitor = SystemResourceMonitor()

    def compute_classification_metrics(
        self,
        y_true: List[int],
        y_pred: List[int]
    ) -> Dict[str, float]:
        """Calculates Accuracy, Precision, Recall, and F1 (Macro and Micro)."""
        acc = float(accuracy_score(y_true, y_pred))

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)

        return {
            "accuracy": acc,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1)
        }

    def generate_all_plots(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_probs: np.ndarray,
        class_names: List[str],
        train_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Renders and exports Confusion Matrix, ROC Curve, PR Curve, and Training Curves.

        Returns:
            List[str]: List of generated chart filepaths.
        """
        plot_paths = []

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title("DocVision AI Evaluation Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=12)
        ax.set_ylabel("True Class", fontsize=12)
        plt.tight_layout()
        cm_path = os.path.join(self.tracker.plots_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        plot_paths.append(cm_path)

        # 2. ROC & AUC Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        auc_scores = []
        for i, name in enumerate(class_names):
            y_bin = [1 if y == i else 0 for y in y_true]
            if len(y_probs.shape) > 1 and i < y_probs.shape[1]:
                fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
                class_auc = float(auc(fpr, tpr))
                auc_scores.append(class_auc)
                ax.plot(fpr, tpr, label=f"{name} (AUC = {class_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title("DocVision AI ROC Curves", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        roc_path = os.path.join(self.tracker.plots_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        plot_paths.append(roc_path)

        # 3. Precision-Recall Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, name in enumerate(class_names):
            y_bin = [1 if y == i else 0 for y in y_true]
            if len(y_probs.shape) > 1 and i < y_probs.shape[1]:
                prec, rec, _ = precision_recall_curve(y_bin, y_probs[:, i])
                ax.plot(rec, prec, label=f"{name}")

        ax.set_title("DocVision AI Precision-Recall Curves", fontsize=14, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.legend(loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        pr_path = os.path.join(self.tracker.plots_dir, "precision_recall_curve.png")
        plt.savefig(pr_path)
        plt.close()
        plot_paths.append(pr_path)

        # 4. Training Loss & Accuracy Curves (If history provided)
        if train_history:
            epochs = [item.get("epoch", idx+1) for idx, item in enumerate(train_history)]
            train_losses = [item.get("train_loss", 0.0) for item in train_history]
            val_losses = [item.get("val_loss", 0.0) for item in train_history]
            val_accs = [item.get("accuracy", 0.0) * 100.0 for item in train_history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
            ax1.plot(epochs, val_losses, "r-s", label="Val Loss")
            ax1.set_title("Training & Validation Loss Curves", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Epoch", fontsize=10)
            ax1.set_ylabel("Loss", fontsize=10)
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2.plot(epochs, val_accs, "g-^", label="Val Accuracy (%)")
            ax2.set_title("Validation Accuracy Curve", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Epoch", fontsize=10)
            ax2.set_ylabel("Accuracy (%)", fontsize=10)
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            curve_path = os.path.join(self.tracker.plots_dir, "training_curves.png")
            plt.savefig(curve_path)
            plt.close()
            plot_paths.append(curve_path)

        logger.info(f"Generated {len(plot_paths)} evaluation charts in '{self.tracker.plots_dir}'")
        return plot_paths

    def evaluate_model(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_probs: np.ndarray,
        class_names: List[str],
        latencies_ms: List[float],
        config: Dict[str, Any],
        train_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete evaluation suite, hardware profiling, chart rendering, and automated experiment tracking.

        Returns:
            Dict[str, Any]: Consolidated evaluation report dictionary.
        """
        metrics = self.compute_classification_metrics(y_true, y_pred)
        hardware_profile = self.monitor.capture_profile(latencies_ms)
        plot_paths = self.generate_all_plots(y_true, y_pred, y_probs, class_names, train_history)

        summary_path = self.tracker.log_experiment(
            config=config,
            metrics=metrics,
            hardware_profile=hardware_profile,
            plot_paths=plot_paths
        )

        return {
            "metrics": metrics,
            "hardware_profile": hardware_profile,
            "plot_paths": plot_paths,
            "experiment_summary_path": summary_path
        }
