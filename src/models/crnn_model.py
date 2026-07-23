"""
Custom PyTorch CRNN (Convolutional Recurrent Neural Network) Architecture.

Combines a ResNet Convolutional Feature Extractor with a 2-layer Bidirectional LSTM
and Linear character vocabulary projection for CTC (Connectionist Temporal Classification) alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.utils.logger import get_logger

logger = get_logger("CRNNModel")


class ResNetFeatureExtractor(nn.Module):
    """
    Modified ResNet-18 Backbone for document text feature map extraction.
    Replaces stride pooling in deep layers to preserve horizontal feature sequence length.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        resnet = models.resnet18(weights=None)

        # Adapt first conv layer for single channel grayscale input if needed
        if in_channels == 1:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Downscale vertical height while preserving horizontal sequence length
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class CRNNModel(nn.Module):
    """
    Complete CRNN Architecture: ResNet CNN + Map-to-Sequence + BiLSTM + CTC Logits.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.2
    ) -> None:
        """
        Args:
            num_classes (int): Size of character vocabulary including CTC blank token.
            in_channels (int): Input image channels (1 for grayscale, 3 for RGB).
            lstm_hidden_size (int): Hidden dimension size in LSTM layers.
            lstm_num_layers (int): Number of stacked Bidirectional LSTM layers.
            dropout (float): Dropout probability between LSTM layers.
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = ResNetFeatureExtractor(in_channels=in_channels)

        # CNN feature maps out channels is 512
        self.map_to_sequence = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)), # Pool vertical dimension H -> 1
            nn.Dropout(dropout)
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)
        logger.info(f"Initialized CRNN Architecture (Classes={num_classes}, Hidden={lstm_hidden_size})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass.

        Args:
            x (torch.Tensor): Input image tensor [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: Logits tensor [Sequence_Length, Batch, Num_Classes] for CTC loss.
        """
        # Feature extraction -> [B, 512, H_feat, W_seq]
        features = self.feature_extractor(x)
        
        # Map to sequence -> [B, 512, 1, W_seq] -> squeeze H
        seq_features = self.map_to_sequence(features).squeeze(2) # [B, 512, W_seq]
        
        # Permute for RNN batch_first -> [B, W_seq, 512]
        seq_features = seq_features.permute(0, 2, 1)

        # Recurrent sequence processing -> [B, W_seq, 2 * hidden_size]
        rnn_out, _ = self.rnn(seq_features)

        # Project to character logits -> [B, W_seq, Num_Classes]
        logits = self.fc(rnn_out)

        # Permute to [W_seq, B, Num_Classes] as required by PyTorch CTCLoss
        return logits.permute(1, 0, 2)
