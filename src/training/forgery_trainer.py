"""
Deep Learning Document Forgery Detector Trainer & Evaluation Engine.

Provides complete training, evaluation, prediction, and visual analytics:
- Model checkpoint saving
- Metrics calculation (Accuracy, Precision, Recall, F1)
- GradCAM feature activation heatmap visualization
- Confusion Matrix heatmap plotting
- Multi-class Receiver Operating Characteristic (ROC) curve plotting
- Precision-Recall (PR) curve plotting
"""

import os
import time
from typing import Dict, List, Tuple, Any, Optional
import cv2
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.resnet_forgery_detector import ResNetForgeryDetector
from src.utils.gradcam import GradCAM
from src.utils.logger import get_logger

logger = get_logger("ForgeryDetectorTrainer")


class ForgeryDetectorTrainer:
    """
    Production Training and Evaluation Engine for ResNet-50 Document Forgery System.
    """

    def __init__(
        self,
        model: ResNetForgeryDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.0001,
        epochs: int = 15,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/forgery"
    ) -> None:
        """
        Args:
            model (ResNetForgeryDetector): ResNet-50 forgery detector instance.
            train_loader (DataLoader): Training dataloader.
            val_loader (DataLoader): Validation dataloader.
            test_loader (Optional[DataLoader]): Testing dataloader.
            learning_rate (float): AdamW initial learning rate.
            epochs (int): Total training epochs.
            device (str): Device target ('cuda' or 'cpu').
            checkpoint_dir (str): Checkpoint persistence directory.
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_f1 = 0.0

    def compute_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculates Accuracy, Precision, Recall, and F1-Score."""
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], output_path: str = "evaluation_results/forgery_cm.png") -> None:
        """Plots Confusion Matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            xticklabels=self.model.class_names,
            yticklabels=self.model.class_names,
            ax=ax
        )
        ax.set_title("DocVision AI Document Forgery Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Integrity State", fontsize=12)
        ax.set_ylabel("True Integrity State", fontsize=12)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved Forgery Confusion Matrix to: {output_path}")

    def plot_roc_curve(self, y_true: List[int], y_probs: np.ndarray, output_path: str = "evaluation_results/forgery_roc.png") -> None:
        """Plots Multi-Class ROC Curves."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, class_name in enumerate(self.model.class_names):
            y_binary = [1 if y == i else 0 for y in y_true]
            fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title("DocVision AI Document Forgery ROC Curves", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved Forgery ROC Curve to: {output_path}")

    def plot_precision_recall_curve(self, y_true: List[int], y_probs: np.ndarray, output_path: str = "evaluation_results/forgery_pr.png") -> None:
        """Plots Precision-Recall Curves."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, class_name in enumerate(self.model.class_names):
            y_binary = [1 if y == i else 0 for y in y_true]
            prec, rec, _ = precision_recall_curve(y_binary, y_probs[:, i])
            ax.plot(rec, prec, label=f"{class_name}")

        ax.set_title("DocVision AI Forgery Precision-Recall Curves", fontsize=14, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.legend(loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved Precision-Recall Curve to: {output_path}")

    def generate_gradcam_visualizations(self, dataloader: DataLoader, output_path: str = "evaluation_results/forgery_gradcam.png") -> None:
        """Generates GradCAM feature activation heatmap overlay on test sample."""
        try:
            target_layer = self.model.conv_layers[-1] # ResNet-50 layer4
            gradcam = GradCAM(self.model, target_layer)

            self.model.eval()
            for images, labels in dataloader:
                img_tensor = images[0:1].to(self.device)
                _, color_heatmap = gradcam.generate_heatmap(img_tensor)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, color_heatmap)
                logger.info(f"Saved GradCAM activation visualization overlay to: {output_path}")
                break
        except Exception as e:
            logger.warning(f"GradCAM visualization failed: {str(e)}")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Forgery Epoch {epoch}/{self.epochs} [Train]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, len(self.train_loader))

    def evaluate(self, dataloader: DataLoader, desc: str = "Val") -> Tuple[float, Dict[str, float], List[int], List[int], np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_true = []
        all_pred = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Forgery [{desc}]"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_true.extend(labels.cpu().numpy().tolist())
                all_pred.extend(preds.tolist())
                all_probs.append(probs)

        avg_loss = total_loss / max(1, len(dataloader))
        probs_matrix = np.vstack(all_probs) if all_probs else np.zeros((0, self.model.num_classes))
        metrics = self.compute_metrics(all_true, all_pred)

        return avg_loss, metrics, all_true, all_pred, probs_matrix

    def fit(self) -> Dict[str, Any]:
        """Runs training, evaluation, checkpointing, and plots ROC/PR/GradCAM graphics."""
        logger.info(f"Starting ResNet-50 Document Forgery training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics, val_true, val_pred, val_probs = self.evaluate(self.val_loader, desc="Val")

            self.scheduler.step()

            logger.info(f"Epoch {epoch} -> Loss: {train_loss:.4f} | Val F1: {val_metrics['f1_score']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}%")

            # Save best checkpoint based on F1-Score
            if val_metrics["f1_score"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1_score"]
                ckpt_path = os.path.join(self.checkpoint_dir, f"resnet50_forgery_best_f1_{val_metrics['f1_score']:.4f}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_metrics": val_metrics,
                    "class_names": self.model.class_names
                }, ckpt_path)
                logger.info(f"Saved best forgery detector checkpoint to: {ckpt_path}")

        # Render evaluation graphics
        self.plot_confusion_matrix(val_true, val_pred)
        self.plot_roc_curve(val_true, val_probs)
        self.plot_precision_recall_curve(val_true, val_probs)
        self.generate_gradcam_visualizations(self.val_loader)

        return {
            "best_val_f1": self.best_val_f1,
            "final_val_metrics": val_metrics
        }
