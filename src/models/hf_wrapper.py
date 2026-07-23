"""
HuggingFace TrOCR Model Fine-Tuning Wrapper.

Provides clean modular interfaces for loading, freezing layers, computing loss,
and fine-tuning HuggingFace VisionEncoderDecoderModel architectures.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger("HuggingFaceTrOCRWrapper")


class HuggingFaceTrOCRWrapper(nn.Module):
    """
    Wrapper for fine-tuning pretrained VisionEncoderDecoderModel (TrOCR) models.
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-printed", freeze_encoder: bool = False) -> None:
        """
        Args:
            model_name (str): Pretrained HuggingFace hub model ID.
            freeze_encoder (bool): Whether to freeze ViT vision encoder weights.
        """
        super().__init__()
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import VisionEncoderDecoderModel
            logger.info(f"Loading pretrained TrOCR model '{self.model_name}'...")
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            if self.freeze_encoder:
                logger.info("Freezing Vision Encoder weights for transfer learning...")
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

        except Exception as e:
            logger.error(f"Failed to load TrOCR model '{self.model_name}': {str(e)}")
            self.model = None

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Forward Pass.

        Args:
            pixel_values (torch.Tensor): Vision encoder input pixels [B, C, H, W].
            labels (Optional[torch.Tensor]): Target token sequence [B, Seq_Len].

        Returns:
            Dict[str, Any]: Output containing 'loss' and 'logits'.
        """
        if self.model is None:
            raise RuntimeError("TrOCR model is not loaded.")
        
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits
        }

    def generate(self, pixel_values: torch.Tensor, max_length: int = 128) -> torch.Tensor:
        """Autoregressive text sequence generation."""
        if self.model is None:
            raise RuntimeError("TrOCR model is not loaded.")
        return self.model.generate(pixel_values, max_length=max_length)
