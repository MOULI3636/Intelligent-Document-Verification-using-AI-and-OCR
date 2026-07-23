"""
Production End-to-End Document Understanding Pipeline.

Encapsulates document acquisition, image pre-processing (deskewing, binarization),
OCR engine dispatching, metric computation, visualization rendering, and JSON formatting.
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np

try:
    from src.engines.base_engine import AbstractOCREngine, OCRResult
    from src.engines.easyocr_engine import EasyOCREngine
    from src.engines.paddleocr_engine import PaddleOCREngine
    from src.engines.trocr_engine import TrOCREngine
    from src.preprocessing.image_processor import DocumentImageProcessor
    from src.utils.io_utils import load_image, save_json
    from src.utils.logger import get_logger
    from src.utils.visualization import render_ocr_overlay
except (ImportError, ValueError):
    from ..engines.base_engine import AbstractOCREngine, OCRResult
    from ..engines.easyocr_engine import EasyOCREngine
    from ..engines.paddleocr_engine import PaddleOCREngine
    from ..engines.trocr_engine import TrOCREngine
    from ..preprocessing.image_processor import DocumentImageProcessor
    from ..utils.io_utils import load_image, save_json
    from ..utils.logger import get_logger
    from ..utils.visualization import render_ocr_overlay

logger = get_logger("DocumentUnderstandingPipeline")


class DocumentUnderstandingPipeline:
    """
    End-to-End Production Document AI Inference Pipeline.
    """

    def __init__(
        self,
        engine_name: str = "easyocr",
        image_processor: Optional[DocumentImageProcessor] = None,
        custom_engine: Optional[AbstractOCREngine] = None
    ) -> None:
        """
        Args:
            engine_name (str): Engine selector ('easyocr', 'paddleocr', 'trocr').
            image_processor (Optional[DocumentImageProcessor]): Image preprocessor instance.
            custom_engine (Optional[AbstractOCREngine]): Optional custom engine override.
        """
        self.processor = image_processor or DocumentImageProcessor(binarization_method="none")

        if custom_engine is not None:
            self.engine = custom_engine
        else:
            self.engine = self._resolve_engine(engine_name)

        logger.info(f"Initialized DocumentUnderstandingPipeline using engine: '{self.engine.engine_name}'")

    def _resolve_engine(self, name: str) -> AbstractOCREngine:
        name_lower = name.lower()
        if "easy" in name_lower:
            engine = EasyOCREngine()
        elif "paddle" in name_lower:
            engine = PaddleOCREngine()
        elif "trocr" in name_lower:
            engine = TrOCREngine()
        else:
            logger.warning(f"Engine name '{name}' not recognized. Defaulting to EasyOCR.")
            engine = EasyOCREngine()

        if not engine.is_available():
            logger.warning(f"Selected engine '{engine.engine_name}' is unavailable. Falling back to EasyOCR.")
            engine = EasyOCREngine()

        return engine

    def run(
        self,
        image_input: Union[str, np.ndarray],
        do_preprocess: bool = True,
        save_overlay_path: Optional[str] = None,
        save_json_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes document AI pipeline over single input image.

        Args:
            image_input (Union[str, np.ndarray]): Image path string or raw numpy array.
            do_preprocess (bool): Enable OpenCV deskewing/binarization before recognition.
            save_overlay_path (Optional[str]): Filepath to save rendered overlay output.
            save_json_path (Optional[str]): Filepath to save JSON parsing output.

        Returns:
            Dict[str, Any]: Structured output dictionary containing text predictions and metadata.
        """
        raw_image = load_image(image_input)

        if do_preprocess:
            processed_image = self.processor.process(raw_image, do_deskew=True, do_resize=False)
        else:
            processed_image = raw_image

        ocr_results: List[OCRResult] = self.engine.recognize(processed_image)

        full_text = " ".join([res.text for res in ocr_results]).strip()
        mean_confidence = float(np.mean([res.confidence for res in ocr_results])) if ocr_results else 0.0

        boxes_list = [res.bbox for res in ocr_results]
        texts_list = [res.text for res in ocr_results]
        confs_list = [res.confidence for res in ocr_results]

        annotated_img = render_ocr_overlay(processed_image, boxes_list, texts_list, confs_list)

        if save_overlay_path:
            import cv2
            cv2.imwrite(save_overlay_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved annotated overlay to: {save_overlay_path}")

        output_payload = {
            "engine_used": self.engine.engine_name,
            "full_text": full_text,
            "mean_confidence": mean_confidence,
            "predictions": [
                {
                    "text": res.text,
                    "bbox": res.bbox,
                    "confidence": res.confidence,
                    "latency_ms": res.inference_time_ms
                } for res in ocr_results
            ]
        }

        if save_json_path:
            save_json(output_payload, save_json_path)

        return output_payload
