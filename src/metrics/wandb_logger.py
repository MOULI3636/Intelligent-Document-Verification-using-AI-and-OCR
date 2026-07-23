"""
Weights & Biases (W&B) Experiment Tracking Logger for DocVision AI.

Provides unified reusable experiment tracking for training and fine-tuning:
- Training and validation loss
- Accuracy, Precision, Recall, and F1-Score
- Learning rate schedule monitoring
- Epoch execution time
- GPU VRAM utilization and RAM memory tracking
- Hyperparameters configuration logging
- Model checkpoint artifact saving to W&B cloud / offline storage
"""

import os
import time
from typing import Dict, Any, Optional
import torch

from src.utils.logger import get_logger

logger = get_logger("WandbLogger")


class WandbLogger:
    """
    Reusable Weights & Biases Logger with graceful offline/disabled fallback safety guards.
    """

    def __init__(
        self,
        project_name: str = "DocVision-AI",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        mode: str = "online" # 'online', 'offline', or 'disabled'
    ) -> None:
        """
        Args:
            project_name (str): W&B project name.
            run_name (Optional[str]): Optional custom run title.
            config (Optional[Dict[str, Any]]): Dictionary of hyperparameters to log.
            enabled (bool): Flag toggling W&B logging.
            mode (str): W&B execution mode ('online', 'offline', 'disabled').
        """
        self.enabled = enabled
        self.wandb = None

        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                os.environ["WANDB_SILENT"] = "true"
                self.run = self.wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config or {},
                    mode=mode,
                    reinit=True
                )
                logger.info(f"Initialized Weights & Biases Run: '{self.run.name}' (Project: '{project_name}')")
            except Exception as e:
                logger.warning(f"W&B initialization failed or wandb package unavailable ({str(e)}). W&B logging disabled.")
                self.enabled = False
                self.wandb = None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Logs metric key-value pairs to W&B.

        Args:
            metrics (Dict[str, Any]): Dictionary of metrics (loss, accuracy, lr, etc.).
            step (Optional[int]): Epoch or global step iteration.
        """
        if not self.enabled or self.wandb is None:
            return

        try:
            # Query hardware GPU allocation if CUDA active
            if torch.cuda.is_available():
                metrics["gpu/vram_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                metrics["gpu/vram_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

            self.wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"W&B log_metrics exception: {str(e)}")

    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        accuracy: float,
        f1_score: float,
        learning_rate: float,
        epoch_time_sec: float
    ) -> None:
        """
        Convenience method logging standard epoch training metrics.
        """
        payload = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/accuracy": accuracy,
            "val/f1_score": f1_score,
            "train/learning_rate": learning_rate,
            "time/epoch_duration_sec": epoch_time_sec
        }
        self.log_metrics(payload, step=epoch)

    def save_checkpoint_artifact(
        self,
        checkpoint_path: str,
        artifact_name: str = "model-checkpoint",
        artifact_type: str = "model"
    ) -> None:
        """
        Uploads model checkpoint file as a W&B Artifact.

        Args:
            checkpoint_path (str): Filepath to PyTorch .pt / .pth checkpoint.
            artifact_name (str): W&B artifact name.
            artifact_type (str): Artifact category type ('model', 'dataset').
        """
        if not self.enabled or self.wandb is None or not os.path.exists(checkpoint_path):
            return

        try:
            artifact = self.wandb.Artifact(name=artifact_name, type=artifact_type)
            artifact.add_file(checkpoint_path)
            self.wandb.log_artifact(artifact)
            logger.info(f"Saved model checkpoint artifact to W&B: '{checkpoint_path}'")
        except Exception as e:
            logger.warning(f"Failed to upload W&B artifact '{checkpoint_path}': {str(e)}")

    def finish() -> None:
        """Closes W&B run instance."""
        if self.enabled and self.wandb is None:
            pass
        elif self.enabled and self.wandb:
            self.wandb.finish()
            logger.info("Closed Weights & Biases run.")
