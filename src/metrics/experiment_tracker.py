"""
Automated Experiment Tracker Module for DocVision AI.

Creates timestamped experiment directories (experiments/exp_YYYYMMDD_HHMMSS/),
persists hyperparameters, evaluation metrics, hardware profiles, and plot artifacts.
"""

from dataclasses import asdict
from datetime import datetime
import os
from typing import Dict, List, Any, Optional

from src.utils.io_utils import save_json
from src.utils.logger import get_logger

logger = get_logger("ExperimentTracker")


class ExperimentTracker:
    """
    Automated Experiment Tracker saving experiment artifacts into isolated timestamped directories.
    """

    def __init__(self, experiments_root: str = "experiments", experiment_name: Optional[str] = None) -> None:
        self.experiments_root = experiments_root
        os.makedirs(self.experiments_root, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_folder = f"exp_{experiment_name}_{timestamp}" if experiment_name else f"exp_{timestamp}"
        self.exp_dir = os.path.join(self.experiments_root, exp_folder)
        self.plots_dir = os.path.join(self.exp_dir, "plots")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        logger.info(f"Initialized Experiment Tracker -> Target Directory: '{self.exp_dir}'")

    def log_experiment(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        hardware_profile: Optional[Any] = None,
        plot_paths: Optional[List[str]] = None
    ) -> str:
        """
        Saves experiment summary JSON file.

        Args:
            config (Dict[str, Any]): Hyperparameters configuration dict.
            metrics (Dict[str, Any]): Evaluated performance metrics.
            hardware_profile (Optional[Any]): HardwareProfile dataclass instance.
            plot_paths (Optional[List[str]]): List of generated chart filepaths.

        Returns:
            str: Path to generated experiment_summary.json.
        """
        payload = {
            "timestamp": datetime.now().isoformat(),
            "experiment_dir": self.exp_dir,
            "configuration": config,
            "metrics": metrics,
            "hardware_profile": asdict(hardware_profile) if hardware_profile and hasattr(hardware_profile, "__dataclass_fields__") else hardware_profile,
            "plots": plot_paths or []
        }

        summary_path = os.path.join(self.exp_dir, "experiment_summary.json")
        save_json(payload, summary_path)
        logger.info(f"Saved automated experiment log summary to: {summary_path}")
        return summary_path
