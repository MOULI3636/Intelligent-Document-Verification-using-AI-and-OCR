"""
Production-Grade Logging Module for DocAI-Bench.

Provides a unified logger with custom formatting, colored console outputs,
file rotation capabilities, and module-level tagging adhering to industry standards.
"""

import logging
import os
import sys
from typing import Optional


class CustomColorFormatter(logging.Formatter):
    """
    Custom ANSI Color Formatter for production terminal output readability.
    Differs log levels by distinct colors for rapid debugging during research runs.
    """

    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    
    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: GREEN + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_logger(name: str = "DocAI-Bench", level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Constructs or retrieves a configured logger instance.

    Args:
        name (str): Name of the logger, typically __name__ of calling module.
        level (str): Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file (Optional[str]): Optional filepath to dump persistent log files.

    Returns:
        logging.Logger: Fully configured Python logger instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if logger already exists
    if logger.hasHandlers():
        return logger

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(CustomColorFormatter())
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
