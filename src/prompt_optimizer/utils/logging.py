"""
Logging utilities for Prompt Optimizer.

This module provides logging configuration and utility functions
for consistent logging across the application.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger with a standard format.

    Args:
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR).

    Examples:
        >>> setup_logging("DEBUG")
        >>> logging.getLogger().level == logging.DEBUG
        True
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name. If None, returns the root logger.

    Returns:
        logging.Logger: A configured logger instance.

    Examples:
        >>> logger = get_logger("my_module")
        >>> logger.name
        'my_module'
    """
    return logging.getLogger(name)
