"""
HuggingFace TrOCR Engine Strategy Concrete Implementation.

Wraps VisionEncoderDecoderModel for state-of-the-art transformer text recognition.
"""

import time
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
torch.set_num_threads(1)

# Monkeypatch torch.load to disable mmap (which causes CPU access violations on Windows)
original_load = torch.load
def safe_load(*args, **kwargs):
    if "mmap" in kwargs:
        kwargs["mmap"] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

try:
    from src.engines.base_engine import AbstractOCREngine, OCRResult
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .base_engine import AbstractOCREngine, OCRResult
    from ..utils.logger import get_logger

logger = get_logger("TrOCREngine")


class TrOCREngine(AbstractOCREngine):
    """
    HuggingFace TrOCR Vision-Encoder-Decoder Engine Implementation.
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-printed", device: str = "cuda") -> None:
        super().__init__(engine_name="HuggingFace-TrOCR")
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.processor = None
        self.model = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        try:
            import os
            import psutil
            import shutil

            # Resolve HF cache directory disk space
            hf_cache_dir = os.path.expanduser("~")
            free_disk_mb = shutil.disk_usage(hf_cache_dir).free / (1024 * 1024)
            free_ram_mb = psutil.virtual_memory().available / (1024 * 1024)

            logger.info(f"TrOCR resource check: Free RAM: {free_ram_mb:.1f} MB, Free Cache Disk: {free_disk_mb:.1f} MB")

            # TrOCR printed base model requires about 1.3 GB disk space and 1.5 GB free RAM to load.
            # If resources are too low, skip loading to prevent access violations / OOM crash.
            if free_ram_mb < 1500.0 or free_disk_mb < 500.0:
                logger.warning(
                    f"TrOCR initialization skipped due to low system resources. "
                    f"Requires at least 1500MB free RAM and 500MB free disk space. "
                    f"Current: RAM={free_ram_mb:.1f}MB, Disk={free_disk_mb:.1f}MB"
                )
                self.processor = None
                self.model = None
                return

            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            logger.info(f"Loading TrOCR weights '{self.model_name}' on device '{self.device}'...")
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name, low_cpu_mem_usage=False).to(self.device)
            self.model.eval()
            logger.info("TrOCR model loaded successfully.")
        except Exception as e:
            logger.warning(f"TrOCR initialization failed: {str(e)}. Engine marked unavailable.")
            self.processor = None
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None and self.processor is not None

    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        if not self.is_available():
            logger.error("TrOCR model is not available.")
            return []

        import cv2
        start_time = time.time()
        try:
            # 1. Convert to grayscale for layout parser
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # 2. Extract text regions
            from src.preprocessing.layout_parser import DocumentLayoutParser
            parser = DocumentLayoutParser()
            bboxes = parser.extract_text_regions(gray)

            # If no boxes found, treat the whole image as a single line
            if not bboxes:
                h, w = image.shape[:2]
                bboxes = [(0, 0, w, h)]

            ocr_results = []
            
            # To avoid slow CPU inference or OOM, cap the maximum number of lines processed by TrOCR to 15
            max_lines = 15
            for idx, (x, y, w, h) in enumerate(bboxes[:max_lines]):
                # Crop region
                crop = image[y:y+h, x:x+w]
                if crop.size == 0:
                    continue

                if len(crop.shape) == 2:
                    pil_crop = Image.fromarray(crop).convert("RGB")
                else:
                    pil_crop = Image.fromarray(crop[:, :, ::-1]).convert("RGB")

                # Run TrOCR on the crop
                pixel_values = self.processor(images=pil_crop, return_tensors="pt").pixel_values.to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values, max_new_tokens=64)

                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                # Format bounding box as polygon vertices
                poly_box = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                
                ocr_results.append(
                    OCRResult(
                        text=generated_text,
                        bbox=poly_box,
                        confidence=0.95,
                        engine_name=self.engine_name,
                        inference_time_ms=0.0
                    )
                )

            elapsed_ms = (time.time() - start_time) * 1000.0
            if ocr_results:
                avg_time = elapsed_ms / len(ocr_results)
                for res in ocr_results:
                    res.inference_time_ms = avg_time

            return ocr_results

        except Exception as e:
            logger.error(f"TrOCR inference error: {str(e)}")
            return []
