"""
Object-Oriented Extensible OCR Benchmarking Framework for DocVision AI.

Computes comprehensive per-engine benchmark metrics:
1. Inference time (execution latency in milliseconds)
2. Average prediction confidence score (0.0 - 1.0)
3. Word count (total recognized words)
4. Character count (total recognized characters)
5. OCR Quality Score (Composite rating between 0.0 and 100.0)

Designed following SOLID Open/Closed principles to seamlessly accept any class
implementing the AbstractOCREngine interface (EasyOCR, PaddleOCR, TrOCR, custom engines).
"""

from dataclasses import dataclass, asdict, field
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd

from src.engines.base_engine import AbstractOCREngine, OCRResult
from src.engines.easyocr_engine import EasyOCREngine
from src.engines.paddleocr_engine import PaddleOCREngine
from src.engines.trocr_engine import TrOCREngine
from src.metrics.ocr_metrics import calculate_cer, calculate_wer, calculate_levenshtein
from src.utils.io_utils import save_json, load_image
from src.utils.logger import get_logger

logger = get_logger("OCRBenchmarkingFramework")


@dataclass
class EngineBenchmarkResult:
    """
    Standardized benchmark result payload emitted for each evaluated OCR engine.

    Attributes:
        engine_name (str): Unique identifier of the engine.
        predicted_text (str): Consolidated text prediction.
        inference_time_ms (float): Execution latency in milliseconds.
        confidence (float): Average confidence score between 0.0 and 1.0.
        word_count (int): Total recognized words.
        character_count (int): Total recognized characters.
        ocr_quality_score (float): Composite quality score between 0.0 and 100.0.
        cer (Optional[float]): Character Error Rate if reference text provided.
        wer (Optional[float]): Word Error Rate if reference text provided.
        predictions_breakdown (List[Dict[str, Any]]): Detailed box-level predictions.
    """
    engine_name: str
    predicted_text: str
    inference_time_ms: float
    confidence: float
    word_count: int
    character_count: int
    ocr_quality_score: float
    cer: Optional[float] = None
    wer: Optional[float] = None
    predictions_breakdown: List[Dict[str, Any]] = field(default_factory=list)


class OCRQualityCalculator:
    """
    Computes composite OCR Quality Score (0.0 - 100.0) combining confidence,
    character density, and CER alignment.
    """

    def compute_quality_score(
        self,
        confidence: float,
        character_count: int,
        cer: Optional[float] = None
    ) -> float:
        """
        Computes composite OCR quality score.

        Args:
            confidence (float): Mean engine confidence score [0.0 - 1.0].
            character_count (int): Total character count.
            cer (Optional[float]): Character Error Rate [0.0 - 1.0].

        Returns:
            float: Composite quality score between 0.0 and 100.0.
        """
        conf_part = min(1.0, max(0.0, confidence)) * 50.0

        if cer is not None:
            cer_part = max(0.0, 1.0 - min(1.0, cer)) * 50.0
            return float(np.clip(conf_part + cer_part, 0.0, 100.0))
        else:
            density_part = min(1.0, character_count / 20.0) * 50.0
            return float(np.clip(conf_part + density_part, 0.0, 100.0))


class OCRBenchmarkingFramework:
    """
    Extensible OOP OCR Benchmarking Suite.
    Accepts any engine implementing AbstractOCREngine and exports structured JSON reports.
    """

    def __init__(self, engines: Optional[List[AbstractOCREngine]] = None) -> None:
        """
        Args:
            engines (Optional[List[AbstractOCREngine]]): List of engine instances.
                Defaults to active implementations of EasyOCR, PaddleOCR, and TrOCR if None provided.
        """
        if engines is None:
            self.engines = self._default_engine_zoo()
        else:
            self.engines = [e for e in engines if e.is_available()]

        self.quality_calculator = OCRQualityCalculator()
        logger.info(f"Initialized OCRBenchmarkingFramework with {len(self.engines)} engines: {[e.engine_name for e in self.engines]}")

    def _default_engine_zoo(self) -> List[AbstractOCREngine]:
        active_engines = []
        for engine in [EasyOCREngine(gpu=False), PaddleOCREngine(), TrOCREngine(device="cpu")]:
            if engine.is_available():
                active_engines.append(engine)
        return active_engines

    def register_engine(self, engine: AbstractOCREngine) -> None:
        """
        Registers an additional OCR engine instance dynamically (Open/Closed Principle).

        Args:
            engine (AbstractOCREngine): Concrete implementation of AbstractOCREngine.
        """
        if engine.is_available():
            self.engines.append(engine)
            logger.info(f"Registered new engine to framework: '{engine.engine_name}'")
        else:
            logger.warning(f"Engine '{engine.engine_name}' is unavailable. Skipped registration.")

    def benchmark_image(
        self,
        image_input: Union[str, np.ndarray],
        reference_text: Optional[str] = None
    ) -> Dict[str, EngineBenchmarkResult]:
        """
        Runs comparative benchmark evaluation across all registered engines for single image.

        Args:
            image_input (Union[str, np.ndarray]): Input image path or numpy array.
            reference_text (Optional[str]): Optional ground truth reference string for CER/WER computation.

        Returns:
            Dict[str, EngineBenchmarkResult]: Dictionary mapping engine name to EngineBenchmarkResult payload.
        """
        image = load_image(image_input)
        benchmark_results: Dict[str, EngineBenchmarkResult] = {}

        for engine in self.engines:
            start_time = time.time()
            ocr_results: List[OCRResult] = engine.recognize(image)
            total_time_ms = (time.time() - start_time) * 1000.0

            predicted_text = " ".join([res.text for res in ocr_results]).strip()
            word_count = len(predicted_text.split())
            character_count = len(predicted_text)

            mean_conf = float(np.mean([res.confidence for res in ocr_results])) if ocr_results else 0.0

            cer = calculate_cer([predicted_text], [reference_text]) if reference_text else None
            wer = calculate_wer([predicted_text], [reference_text]) if reference_text else None

            quality_score = self.quality_calculator.compute_quality_score(
                confidence=mean_conf,
                character_count=character_count,
                cer=cer
            )

            breakdown = [
                {
                    "text": res.text,
                    "bbox": res.bbox,
                    "confidence": res.confidence,
                    "latency_ms": res.inference_time_ms
                } for res in ocr_results
            ]

            res_obj = EngineBenchmarkResult(
                engine_name=engine.engine_name,
                predicted_text=predicted_text,
                inference_time_ms=total_time_ms,
                confidence=mean_conf,
                word_count=word_count,
                character_count=character_count,
                ocr_quality_score=quality_score,
                cer=cer,
                wer=wer,
                predictions_breakdown=breakdown
            )

            benchmark_results[engine.engine_name] = res_obj
            logger.info(f"Engine '{engine.engine_name}' -> Time: {total_time_ms:.1f}ms | Conf: {mean_conf:.2f} | Words: {word_count} | Chars: {character_count} | Quality: {quality_score:.1f}/100")

        return benchmark_results

    def export_json(
        self,
        results: Dict[str, EngineBenchmarkResult],
        json_output_path: str
    ) -> None:
        """
        Exports benchmark results dictionary to structured JSON file.

        Args:
            results (Dict[str, EngineBenchmarkResult]): Benchmark output dictionary.
            json_output_path (str): Target filepath for JSON export.
        """
        serializable_dict = {
            engine_name: asdict(res) for engine_name, res in results.items()
        }
        save_json(serializable_dict, json_output_path)
        logger.info(f"Exported benchmark results JSON to: {json_output_path}")

    def to_dataframe(
        self,
        results: Dict[str, EngineBenchmarkResult]
    ) -> pd.DataFrame:
        """Converts benchmark results to Pandas DataFrame summary table."""
        rows = []
        for name, res in results.items():
            rows.append({
                "Engine": res.engine_name,
                "Inference Time (ms)": round(res.inference_time_ms, 2),
                "Mean Confidence": round(res.confidence, 4),
                "Word Count": res.word_count,
                "Character Count": res.character_count,
                "OCR Quality Score": round(res.ocr_quality_score, 2),
                "CER": round(res.cer, 4) if res.cer is not None else "N/A",
                "WER": round(res.wer, 4) if res.wer is not None else "N/A"
            })
        return pd.DataFrame(rows)
