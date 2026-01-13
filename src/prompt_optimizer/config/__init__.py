"""
Configuration module for Prompt Optimizer.

This module provides configuration management including:
- Environment variable loading
- Settings dataclass for application configuration
"""

from prompt_optimizer.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
