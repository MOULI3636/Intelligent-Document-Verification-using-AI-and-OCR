"""
Input/Output Utilities for File Management, Image Serialization, and JSON Export.
"""

import json
import os
from typing import Any, Dict, Optional
import cv2
import numpy as np
from PIL import Image

try:
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .logger import get_logger

logger = get_logger("IOUtils")


def load_image(image_input: Any) -> np.ndarray:
    """
    Unified image loader handling file paths, PIL Images, or raw NumPy arrays.

    Args:
        image_input (Any): Filepath string, PIL Image, or NumPy array.

    Returns:
        np.ndarray: BGR image array.

    Raises:
        ValueError: If input format is unsupported or file cannot be opened.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image path does not exist: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"OpenCV failed to decode image at: {image_input}")
        return img
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        return image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def save_json(data: Dict[str, Any], save_path: str, indent: int = 4) -> None:
    """
    Saves dictionary data to structured JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Saved JSON export to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON to '{save_path}': {str(e)}")
        raise e


def load_json(json_path: str) -> Dict[str, Any]:
    """
    Loads data from structured JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
