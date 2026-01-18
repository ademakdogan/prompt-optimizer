"""
Unit tests for configuration module.
"""

import os
from unittest.mock import patch

import pytest

from prompt_optimizer.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            assert settings.window_size == 2
            assert settings.loop_count == 3
            assert settings.log_level == "INFO"

    def test_settings_from_env(self) -> None:
        """Test settings loaded from environment variables."""
        env_vars = {
            "OPENROUTER_API_KEY": "test-key-123",
            "AGENT_MODEL": "custom/agent-model",
            "MENTOR_MODEL": "custom/mentor-model",
            "WINDOW_SIZE": "5",
            "LOOP_COUNT": "10",
            "LOG_LEVEL": "DEBUG",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            assert settings.openrouter_api_key == "test-key-123"
            assert settings.agent_model == "custom/agent-model"
            assert settings.mentor_model == "custom/mentor-model"
            assert settings.window_size == 5
            assert settings.loop_count == 10
            assert settings.log_level == "DEBUG"

    def test_api_key_from_settings(self) -> None:
        """Test that API key can be loaded from environment."""
        env_vars = {
            "OPENROUTER_API_KEY": "test-key",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            assert settings.openrouter_api_key == "test-key"


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self) -> None:
        """Test that get_settings returns a Settings instance."""
        # Clear the cache first
        get_settings.cache_clear()
        
        settings = get_settings()
        
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self) -> None:
        """Test that get_settings returns cached instance."""
        get_settings.cache_clear()
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
