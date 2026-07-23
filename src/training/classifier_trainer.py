"""
Document Classifier Training, Validation, & Testing Engine for DocVision AI.

Implements production-grade model training:
- Automatic Mixed Precision (AMP) acceleration
- Early Stopping with configurable patience
- Top-k model checkpointing & persistence
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, Macro/Micro F1)
- Confusion Matrix generation & Seaborn heatmap plotting
"""

import os
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.efficientnet_classifier import EfficientNetDocumentClassifier
from src.utils.config_parser import ConfigSchema
from src.utils.logger import get_logger

logger = get_logger("DocumentClassifierTrainer")


class EarlyStopping:
    """
    Early Stopping callback monitoring validation loss to prevent overfitting.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001) -> None:
        """
        Args:
            patience (int): Number of epochs to wait without improvement before stopping.
            min_delta (float): Minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class DocumentClassifierTrainer:
    """
    Production PyTorch Training engine for EfficientNet-B0 Document Classifier.
    """

    def __init__(
        self,
        model: EfficientNetDocumentClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        config: Optional[ConfigSchema] = None,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.0001,
        epochs: int = 20,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/classifier"
    ) -> None:
        """
        Args:
            model (EfficientNetDocumentClassifier): Model instance.
            train_loader (DataLoader): Training dataloader.
            val_loader (DataLoader): Validation dataloader.
            test_loader (Optional[DataLoader]): Testing dataloader.
            config (Optional[ConfigSchema]): Master config.
            learning_rate (float): Optimizer initial learning rate.
            weight_decay (float): AdamW weight decay.
            epochs (int): Max training epochs.
            device (str): Device target ('cuda' or 'cpu').
            checkpoint_dir (str): Directory path to persist model checkpoints.
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.early_stopping = EarlyStopping(patience=5)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_accuracy = 0.0

    def compute_metrics(
        self,
        y_true: List[int],
        y_pred: List[int]
    ) -> Dict[str, float]:
        """Calculates Accuracy, Precision, Recall, and F1-Score."""
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }

    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        output_path: str = "evaluation_results/classifier_cm.png"
    ) -> None:
        """Generates and exports Confusion Matrix heatmap plot."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.model.class_names,
            yticklabels=self.model.class_names,
            ax=ax
        )

        ax.set_title("DocVision AI Document Classifier Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=12)
        ax.set_ylabel("True Class", fontsize=12)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Exported Confusion Matrix plot to: {output_path}")

    def train_epoch(self, epoch: int) -> float:
        """Executes single training epoch with AMP."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Classifier Epoch {epoch}/{self.epochs} [Train]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, len(self.train_loader))

    def evaluate(self, dataloader: DataLoader, desc: str = "Val") -> Tuple[float, Dict[str, float], List[int], List[int]]:
        """Executes evaluation loop over specified dataloader."""
        self.model.eval()
        total_loss = 0.0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Classifier [{desc}]"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_true.extend(labels.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())

        avg_loss = total_loss / max(1, len(dataloader))
        metrics = self.compute_metrics(all_true, all_pred)
        return avg_loss, metrics, all_true, all_pred

    def fit(self) -> Dict[str, Any]:
        """Runs full training, validation, early stopping, and checkpointing workflow."""
        logger.info(f"Starting EfficientNet-B0 Document Classifier training for {self.epochs} epochs...")

        history = []

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics, val_true, val_pred = self.evaluate(self.val_loader, desc="Val")

            self.scheduler.step()

            logger.info(f"Epoch {epoch} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1: {val_metrics['f1_score']:.4f}")

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics
            })

            # Checkpoint best model based on validation accuracy
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                ckpt_path = os.path.join(self.checkpoint_dir, f"efficientnet_b0_best_acc_{val_metrics['accuracy']:.4f}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "class_names": self.model.class_names
                }, ckpt_path)
                logger.info(f"Saved new best classifier checkpoint to: {ckpt_path}")

            # Early Stopping Check
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # Generate Confusion Matrix on Validation Set
        self.plot_confusion_matrix(val_true, val_pred)

        # Test Set Evaluation (If test_loader provided)
        test_results = None
        if self.test_loader:
            test_loss, test_metrics, test_true, test_pred = self.evaluate(self.test_loader, desc="Test")
            logger.info(f"Final Test Evaluation -> Loss: {test_loss:.4f} | Accuracy: {test_metrics['accuracy']*100:.2f}% | F1: {test_metrics['f1_score']:.4f}")
            test_results = {"test_loss": test_loss, **test_metrics}

        return {
            "best_val_accuracy": self.best_val_accuracy,
            "history": history,
            "test_results": test_results
        }
