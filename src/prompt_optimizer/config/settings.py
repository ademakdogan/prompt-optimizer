"""
Settings module for application configuration.

This module provides configuration management using pydantic-settings
for environment variable loading and validation.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes:
        openrouter_api_key: API key for OpenRouter service.
        agent_model: Model identifier for the agent (processes data).
        mentor_model: Model identifier for the mentor (generates prompts).
        window_size: Number of historical prompt packages to send to mentor.
        loop_count: Number of optimization iterations.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).

    Examples:
        >>> settings = Settings()
        >>> settings.window_size
        2
        >>> settings.loop_count
        3
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    openrouter_api_key: str = ""
    agent_model: str = "openai/gpt-5-nano"
    mentor_model: str = "openai/gpt-5-nano"
    window_size: int = 2
    loop_count: int = 3
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: The application settings instance.

    Examples:
        >>> settings = get_settings()
        >>> isinstance(settings, Settings)
        True
    """
    return Settings()
