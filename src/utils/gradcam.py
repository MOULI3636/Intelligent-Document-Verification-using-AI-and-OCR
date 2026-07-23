"""
Gradient-weighted Class Activation Mapping (GradCAM) for Visual Document Explanation.

Generates visual attention heatmaps highlighting image regions that influenced
the model's forgery detection prediction.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .logger import get_logger

logger = get_logger("GradCAM")


class GradCAM:
    """
    GradCAM implementation attaching forward/backward hooks to target convolutional layers.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Args:
            model (nn.Module): PyTorch model instance.
            target_layer (nn.Module): Target conv layer (e.g. model.conv_layers[-1]).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes GradCAM activation heatmap.

        Args:
            input_tensor (torch.Tensor): Input tensor [1, 3, H, W].
            target_class_idx (Optional[int]): Target class index to visualize. Defaults to top prediction if None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Normalized Heatmap Array [H, W], Color Overlay Image [H, W, 3]).
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = torch.argmax(output, dim=1).item()

        score = output[0, target_class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks failed to capture gradients or activations.")

        # Global average pooling over gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)

        # Apply ReLU to retain positive influence
        cam = torch.clamp(cam, min=0).cpu().numpy()

        # Resize heatmap to input tensor dimensions
        h, w = input_tensor.shape[2:]
        cam_resized = cv2.resize(cam, (w, h))

        # Normalize to [0.0, 1.0]
        if np.max(cam_resized) > 0:
            cam_normalized = (cam_resized - np.min(cam_resized)) / (np.max(cam_resized) - np.min(cam_resized) + 1e-8)
        else:
            cam_normalized = cam_resized

        heatmap_uint8 = (cam_normalized * 255.0).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        logger.debug(f"Generated GradCAM heatmap for class index {target_class_idx}")
        return cam_normalized, color_heatmap
