"""
Albumentations Document Augmentation Pipeline for Research & Robustness Testing.

Applies realistic document noise and physical degradation transforms:
- Motion blur & Gaussian blur (simulating camera shake / out-of-focus capture)
- Perspective warping (simulating non-planar document capture)
- Illumination gradients & random shadows (simulating uneven lighting)
- Pixel-level salt-and-pepper noise
"""

from typing import Dict, Any
import albumentations as A
import numpy as np

try:
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from ..utils.logger import get_logger

logger = get_logger("DocumentAugmentations")


class DocumentAugmentationPipeline:
    """
    Configurable Albumentations Pipeline tailored specifically for document images.
    """

    def __init__(self, p_overall: float = 0.8) -> None:
        """
        Initializes document degradation transform sequence.

        Args:
            p_overall (float): Probability of applying augmentations.
        """
        self.transform = A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.4),
            A.OneOf([
                A.Perspective(scale=(0.01, 0.05), p=0.5),
                A.PiecewiseAffine(scale=(0.01, 0.03), p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ], p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ], p=p_overall)
        logger.info("Initialized Albumentations Document Augmentation Pipeline")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies augmentations to an input image array.

        Args:
            image (np.ndarray): Input BGR or Grayscale image.

        Returns:
            np.ndarray: Augmented document image array.
        """
        if len(image.shape) == 2:
            image_3ch = np.stack([image] * 3, axis=-1)
            augmented = self.transform(image=image_3ch)["image"]
            return augmented[:, :, 0]
        else:
            return self.transform(image=image)["image"]
