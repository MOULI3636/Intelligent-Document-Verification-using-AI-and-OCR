"""
Configuration Parsing and Schema Validation Module.

Parses YAML files into robust Python dataclasses with strict type validation,
default value fallbacks, and dictionary access interfaces.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional
import yaml

try:
    from src.utils.logger import get_logger
except (ImportError, ValueError):
    from .logger import get_logger

logger = get_logger("ConfigParser")


@dataclass
class PreprocessingConfig:
    target_height: int = 32
    target_width: int = 128
    grayscale: bool = True
    deskew_enabled: bool = True
    max_deskew_angle: float = 45.0
    binarization_method: str = "adaptive"
    block_size: int = 11
    c_value: int = 2


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    use_amp: bool = True
    scheduler_name: str = "cosine"
    save_dir: str = "checkpoints"


@dataclass
class ConfigSchema:
    """
    Master Schema validating central configuration attributes.
    """
    project_name: str = "DocAI-Bench"
    device: str = "cuda"
    seed: int = 42
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    raw_dict: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str) -> ConfigSchema:
    """
    Loads and validates a YAML configuration file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        ConfigSchema: Parsed configuration object.

    Raises:
        FileNotFoundError: If configuration path does not exist.
        ValueError: If YAML parsing fails.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Config path '{config_path}' does not exist.")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

        project_cfg = raw_cfg.get("project", {})
        prep_raw = raw_cfg.get("preprocessing", {})
        train_raw = raw_cfg.get("training", {})

        prep_cfg = PreprocessingConfig(
            target_height=prep_raw.get("target_height", 32),
            target_width=prep_raw.get("target_width", 128),
            grayscale=prep_raw.get("grayscale", True),
            deskew_enabled=prep_raw.get("deskew", {}).get("enabled", True),
            max_deskew_angle=prep_raw.get("deskew", {}).get("max_angle", 45.0),
            binarization_method=prep_raw.get("binarization", {}).get("method", "adaptive"),
            block_size=prep_raw.get("binarization", {}).get("block_size", 11),
            c_value=prep_raw.get("binarization", {}).get("c_value", 2)
        )

        train_cfg = TrainingConfig(
            batch_size=train_raw.get("batch_size", 32),
            epochs=train_raw.get("epochs", 25),
            learning_rate=train_raw.get("learning_rate", 0.0005),
            weight_decay=train_raw.get("weight_decay", 0.0001),
            grad_clip=train_raw.get("grad_clip", 5.0),
            use_amp=train_raw.get("use_amp", True),
            scheduler_name=train_raw.get("scheduler", {}).get("name", "cosine"),
            save_dir=train_raw.get("checkpoint", {}).get("save_dir", "checkpoints")
        )

        config_obj = ConfigSchema(
            project_name=project_cfg.get("name", "DocAI-Bench"),
            device=project_cfg.get("device", "cuda"),
            seed=project_cfg.get("seed", 42),
            preprocessing=prep_cfg,
            training=train_cfg,
            raw_dict=raw_cfg
        )

        logger.info(f"Successfully loaded configuration from {config_path}")
        return config_obj

    except Exception as e:
        logger.error(f"Failed to parse config file '{config_path}': {str(e)}")
        raise ValueError(f"Invalid YAML config structure: {str(e)}")
