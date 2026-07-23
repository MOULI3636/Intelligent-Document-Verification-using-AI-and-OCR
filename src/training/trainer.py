"""
Production PyTorch Training Engine for Document Recognition Models.

Implements full-featured training and evaluation loops:
- Automatic Mixed Precision (AMP) acceleration
- Gradient norm clipping to prevent gradient explosion in LSTMs
- Cosine Annealing learning rate scheduling
- Metric tracking (CER, WER, Loss)
- Automated model checkpointing and top-k weight saving
"""

import os
import time
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VocabularyEncoder
from src.metrics.ocr_metrics import OCRMetricSuite
from src.training.loss import CTCLossWrapper
from src.utils.config_parser import ConfigSchema
from src.utils.logger import get_logger

logger = get_logger("DocumentTrainer")


class DocumentTrainer:
    """
    Modular Trainer managing model optimization, validation routines, and checkpoint persistence.
    """

    def __init__(
        self,
        model: nn.Module,
        vocab_encoder: VocabularyEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ConfigSchema,
        device: str = "cuda"
    ) -> None:
        """
        Args:
            model (nn.Module): PyTorch CRNN or Transformer model.
            vocab_encoder (VocabularyEncoder): Vocabulary tokenizer.
            train_loader (DataLoader): Training dataloader.
            val_loader (DataLoader): Validation dataloader.
            config (ConfigSchema): Central configuration instance.
            device (str): Execution device ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.vocab = vocab_encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.criterion = CTCLossWrapper(blank_idx=self.vocab.blank_idx)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=1e-5
        )

        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.metric_suite = OCRMetricSuite()

        self.checkpoint_dir = config.training.save_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_cer = float("inf")

    def _decode_predictions(self, logits: torch.Tensor) -> List[str]:
        """Greedy CTC decoding over output character logits [W_seq, B, Num_Classes]."""
        # Argmax over character classes -> [W_seq, B]
        argmax_indices = torch.argmax(logits, dim=2).detach().cpu().numpy()
        decoded_texts = []
        
        # Iterate over batch dimension B
        for b in range(argmax_indices.shape[1]):
            seq_indices = argmax_indices[:, b]
            text = self.vocab.decode(seq_indices.tolist())
            decoded_texts.append(text)

        return decoded_texts

    def train_epoch(self, epoch: int) -> float:
        """Executes single training epoch."""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.training.epochs} [Train]")
        for batch in pbar:
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Forward pass -> [W_seq, B, Num_Classes]
                logits = self.model(images)
                loss = self.criterion(logits, targets, target_lengths)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(1, len(self.train_loader))
        logger.info(f"Epoch {epoch} Training Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")
        return avg_loss

    def validate(self, epoch: int) -> Dict[str, float]:
        """Executes validation epoch and computes CER/WER metrics."""
        self.model.eval()
        self.metric_suite.reset()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                images = batch["images"].to(self.device)
                targets = batch["targets"].to(self.device)
                target_lengths = batch["target_lengths"].to(self.device)
                raw_texts = batch["raw_texts"]

                logits = self.model(images)
                loss = self.criterion(logits, targets, target_lengths)
                total_loss += loss.item()

                preds = self._decode_predictions(logits)
                self.metric_suite.update(preds, raw_texts)

        metrics = self.metric_suite.compute()
        metrics["val_loss"] = total_loss / max(1, len(self.val_loader))

        logger.info(f"Validation Epoch {epoch} -> Loss: {metrics['val_loss']:.4f} | CER: {metrics['cer']:.4f} | WER: {metrics['wer']:.4f}")
        return metrics

    def fit(self) -> None:
        """Runs complete multi-epoch training pipeline with checkpointing."""
        logger.info(f"Starting model training for {self.config.training.epochs} epochs...")

        for epoch in range(1, self.config.training.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            self.scheduler.step()

            # Checkpoint best model based on validation CER
            if val_metrics["cer"] < self.best_val_cer:
                self.best_val_cer = val_metrics["cer"]
                save_path = os.path.join(self.checkpoint_dir, f"best_model_epoch_{epoch}_cer_{val_metrics['cer']:.4f}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "vocab": self.vocab.chars
                }, save_path)
                logger.info(f"Saved new best model checkpoint to: {save_path}")

        logger.info("Training pipeline execution completed.")
