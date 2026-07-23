"""
ResNet-50 PyTorch Deep Learning Document Forgery Classifier Architecture.

Classifies input document images into 6 physical/digital integrity states:
1. Original
2. Edited
3. Screenshot
4. Blurred
5. Photocopy
6. Tampered
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.models as models

from src.utils.logger import get_logger

logger = get_logger("ResNetForgeryDetector")


class ResNetForgeryDetector(nn.Module):
    """
    ResNet-50 Backbone with custom classification head for 6 document forgery categories.
    """

    FORGERY_CLASSES = [
        "Original",
        "Edited",
        "Screenshot",
        "Blurred",
        "Photocopy",
        "Tampered"
    ]

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ) -> None:
        """
        Args:
            num_classes (int): Number of forgery categories (default: 6).
            pretrained (bool): Whether to load ImageNet pre-trained weights.
            dropout_rate (float): Dropout probability before linear classification head.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = self.FORGERY_CLASSES[:num_classes]

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Retain feature extractor up to avgpool
        self.conv_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4 # Target layer for GradCAM
        )
        self.avgpool = resnet.avgpool

        # Replace linear FC head (2048 in_features for ResNet-50)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        logger.info(f"Initialized ResNet-50 Forgery Detector ({num_classes} categories: {self.class_names})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass.

        Args:
            x (torch.Tensor): Input image tensor [Batch, 3, Height, Width].

        Returns:
            torch.Tensor: Logits tensor [Batch, Num_Classes].
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.conv_layers(x)
        pooled = self.avgpool(features).flatten(1)
        logits = self.fc(pooled)
        return logits

    def predict(self, image_tensor: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """
        Predicts forgery category and probability scores for single image.

        Args:
            image_tensor (torch.Tensor): Input image tensor [3, H, W] or [1, 3, H, W].

        Returns:
            Tuple[str, float, Dict[str, float]]: (Top Category Name, Confidence Probability, Probability Dict).
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
