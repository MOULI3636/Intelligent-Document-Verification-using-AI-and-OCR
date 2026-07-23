"""
EfficientNet-B0 PyTorch Document Classification Architecture.

Classifies input document images into 7 distinct document categories:
1. Invoice
2. Receipt
3. Passport
4. PAN
5. Aadhaar
6. Driving License
7. Cheque
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.models as models

from src.utils.logger import get_logger

logger = get_logger("EfficientNetDocumentClassifier")


class EfficientNetDocumentClassifier(nn.Module):
    """
    EfficientNet-B0 Backbone with custom linear classification head for 7 document categories.
    """

    DOCUMENT_CLASSES = [
        "Invoice",
        "Receipt",
        "Passport",
        "PAN",
        "Aadhaar",
        "Driving License",
        "Cheque"
    ]

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ) -> None:
        """
        Args:
            num_classes (int): Number of document categories (default: 7).
            pretrained (bool): Whether to load ImageNet pre-trained weights.
            dropout_rate (float): Dropout probability before linear classification head.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = self.DOCUMENT_CLASSES[:num_classes]

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Replace default classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        logger.info(f"Initialized EfficientNet-B0 Document Classifier ({num_classes} classes: {self.class_names})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass.

        Args:
            x (torch.Tensor): Input tensor [Batch, 3, Height, Width] normalized to RGB.

        Returns:
            torch.Tensor: Logits tensor [Batch, Num_Classes].
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1) # Expand 1-channel grayscale to 3 RGB channels
        return self.backbone(x)

    def predict_class(self, image_tensor: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """
        Predicts top document category and class probability distribution.

        Args:
            image_tensor (torch.Tensor): Input image tensor [3, H, W] or [1, 3, H, W].

        Returns:
            Tuple[str, float, Dict[str, float]]: (Predicted Class Name, Top Probability, Probability Distribution Dict).
        """
        self.eval()
        with torch.no_grad():
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)

            logits = self.forward(image_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

            top_idx = torch.argmax(probs).item()
            top_prob = probs[top_idx].item()
            top_class = self.class_names[top_idx]

            prob_dist = {
                self.class_names[i]: float(probs[i].item()) for i in range(len(self.class_names))
            }

            return top_class, float(top_prob), prob_dist
