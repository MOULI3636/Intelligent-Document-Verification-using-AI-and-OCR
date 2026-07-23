"""
Dataset On-Disk Caching Engine for DocVision AI.

Caches preprocessed document images, grayscale arrays, and tokenized label sequences
to disk for high-speed multi-epoch PyTorch training.
"""

import hashlib
import os
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from src.utils.io_utils import save_json, load_json
from src.utils.logger import get_logger

logger = get_logger("DatasetCache")


class DatasetCache:
    """
    On-disk binary and metadata cache manager.
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_key(self, image_path: str, params_str: str = "") -> str:
        """Generates MD5 hash key from image path and preprocessing parameters."""
        identifier = f"{image_path}_{params_str}_{os.path.getmtime(image_path) if os.path.exists(image_path) else 0}"
        return hashlib.md5(identifier.encode("utf-8")).hexdigest()

    def get(self, image_path: str, params_str: str = "") -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieves cached preprocessed image array and metadata dictionary if present.

        Returns:
            Optional[Tuple[np.ndarray, Dict[str, Any]]]: (Cached Image Array, Metadata) if found, else None.
        """
        key = self._get_key(image_path, params_str)
        img_cache_path = os.path.join(self.cache_dir, f"{key}.png")
        json_cache_path = os.path.join(self.cache_dir, f"{key}.json")

        if os.path.exists(img_cache_path) and os.path.exists(json_cache_path):
            img = cv2.imread(img_cache_path, cv2.IMREAD_UNCHANGED)
            meta = load_json(json_cache_path)
            return img, meta
        return None

    def put(self, image_path: str, processed_image: np.ndarray, metadata: Dict[str, Any], params_str: str = "") -> str:
        """
        Saves preprocessed image array and metadata dictionary to cache.

        Returns:
            str: Cache key identifier.
        """
        key = self._get_key(image_path, params_str)
        img_cache_path = os.path.join(self.cache_dir, f"{key}.png")
        json_cache_path = os.path.join(self.cache_dir, f"{key}.json")

        cv2.imwrite(img_cache_path, processed_image)
        save_json(metadata, json_cache_path)
        return key

    def clear(self) -> None:
        """Clears all cached items."""
        for fname in os.listdir(self.cache_dir):
            fpath = os.path.join(self.cache_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        logger.info("Cleared dataset cache directory.")
