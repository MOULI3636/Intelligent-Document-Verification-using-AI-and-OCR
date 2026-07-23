"""
DocVision AI Evaluation Metrics, Profiling, Failure Analysis, & Experiment Tracking Subpackage.
"""

from src.metrics.ocr_metrics import calculate_cer, calculate_wer, calculate_levenshtein, OCRMetricSuite
from src.metrics.fraud_metrics import compute_fraud_metrics
from src.metrics.evaluator import OCREvaluator
from src.metrics.ocr_benchmark_framework import OCRBenchmarkingFramework, EngineBenchmarkResult, OCRQualityCalculator
from src.metrics.system_monitor import SystemResourceMonitor, HardwareProfile
from src.metrics.experiment_tracker import ExperimentTracker
from src.metrics.master_evaluator import MasterEvaluationFramework
from src.metrics.failure_analyzer import FailureAnalysisEngine, FailureCase
from src.metrics.wandb_logger import WandbLogger

__all__ = [
    "calculate_cer",
    "calculate_wer",
    "calculate_levenshtein",
    "OCRMetricSuite",
    "compute_fraud_metrics",
    "OCREvaluator",
    "OCRBenchmarkingFramework",
    "EngineBenchmarkResult",
    "OCRQualityCalculator",
    "SystemResourceMonitor",
    "HardwareProfile",
    "ExperimentTracker",
    "MasterEvaluationFramework",
    "FailureAnalysisEngine",
    "FailureCase",
    "WandbLogger",
]
