"""
Seed Reproducibility & Deterministic Execution Utility for DocVision AI.

Ensures exact experiment reproducibility across CPU and CUDA GPU environments:
- Python built-in random seed
- NumPy random seed
- PyTorch CPU and CUDA seeds
- PyTorch CUDNN deterministic flags
"""

import os
import random
import numpy as np
import torch

try:
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .logger import get_logger

logger = get_logger("SeedReproducibility")


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> int:
    """
    Sets global random seeds for Python, NumPy, and PyTorch to guarantee experiment reproducibility.

    Args:
        seed (int): Integer seed value (default: 42).
        deterministic_cudnn (bool): Whether to enforce CUDNN deterministic flags.

    Returns:
        int: Applied seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info(f"Global random seed set to {seed} (CUDNN Deterministic: {deterministic_cudnn})")
    return seed
