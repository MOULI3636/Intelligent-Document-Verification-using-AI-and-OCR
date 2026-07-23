"""
DocAI-Bench: Enterprise Multi-Engine Document Understanding & Fine-Tuning Framework.

This package provides research-grade abstractions for document pre-processing,
multi-engine OCR evaluation, deep learning fine-tuning (PyTorch & HuggingFace),
evaluation metrics (TorchMetrics, CER/WER), and production deployment pipelines.
"""

import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

__version__ = "1.0.0"
__author__ = "AI Research Engineer"
