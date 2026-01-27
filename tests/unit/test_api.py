"""
Unit tests for API client module.
"""

from unittest.mock import MagicMock, patch

import pytest

from prompt_optimizer.api.client import OpenRouterClient


class TestOpenRouterClient:
    """Tests for OpenRouterClient class."""

    def test_init_without_api_key_raises_error(self) -> None:
        """Test that initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "prompt_optimizer.api.client.get_settings"
            ) as mock_settings:
                mock_settings.return_value.openrouter_api_key = ""
                
                with pytest.raises(ValueError, match="API key is required"):
                    OpenRouterClient(model="test-model")

    @patch("prompt_optimizer.api.client.instructor")
    @patch("prompt_optimizer.api.client.OpenAI")
    def test_init_with_api_key(
        self,
        mock_openai: MagicMock,
        mock_instructor: MagicMock,
    ) -> None:
        """Test successful initialization with API key."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_instructor.from_openai.return_value = mock_client
        
        client = OpenRouterClient(model="test-model", api_key="test-key")
        
        assert client.model == "test-model"
        assert client.api_key == "test-key"

    @patch("prompt_optimizer.api.client.instructor")
    @patch("prompt_optimizer.api.client.OpenAI")
    def test_chat_structured_response(
        self,
        mock_openai: MagicMock,
        mock_instructor: MagicMock,
    ) -> None:
        """Test chat method returns structured response."""
        from pydantic import BaseModel
        
        class MockResponse(BaseModel):
            value: str = ""
        
        mock_patched_client = MagicMock()
        mock_instructor.from_openai.return_value = mock_patched_client
        
        mock_response = MockResponse(value="test")
        mock_patched_client.chat.completions.create.return_value = mock_response
        
        client = OpenRouterClient(model="test-model", api_key="test-key")
        
        result = client.chat(
            messages=[{"role": "user", "content": "Test"}],
            response_model=MockResponse,
        )
        
        assert result == mock_response


class TestAgentModel:
    """Tests for AgentModel class."""

    @patch("prompt_optimizer.api.agent.OpenRouterClient")
    def test_process_data(self, mock_client_cls: MagicMock) -> None:
        """Test processing data with agent model."""
        from prompt_optimizer.api.agent import AgentModel
        
        mock_client = MagicMock()
        # Agent now uses chat_raw and returns dict
        mock_client.chat_raw.return_value = '{"firstname": "John", "email": "test@email.com"}'
        mock_client_cls.return_value = mock_client
        
        agent = AgentModel(model="test-model", api_key="test-key")
        result = agent.process_data(
            prompt="Extract data",
            data="Test data",
        )
        
        assert isinstance(result, dict)
        assert result.get("firstname") == "John"


class TestMentorModel:
    """Tests for MentorModel class."""

    @patch("prompt_optimizer.api.mentor.OpenRouterClient")
    def test_generate_initial_prompt(self, mock_client_cls: MagicMock) -> None:
        """Test generating initial prompt."""
        from prompt_optimizer.api.mentor import MentorModel
        from prompt_optimizer.models import GeneratedPrompt
        
        mock_client = MagicMock()
        mock_client.chat.return_value = GeneratedPrompt(
            prompt="Generated prompt",
            reasoning="Test reasoning",
        )
        mock_client_cls.return_value = mock_client
        
        mentor = MentorModel(model="test-model", api_key="test-key")
        result = mentor.generate_initial_prompt(
            sample_data="Test data",
            ground_truth={"firstname": "John"},  # Now dict
        )
        
        assert result.prompt == "Generated prompt"

    @patch("prompt_optimizer.api.mentor.OpenRouterClient")
    def test_generate_prompt(self, mock_client_cls: MagicMock) -> None:
        """Test generating improved prompt."""
        from prompt_optimizer.api.mentor import MentorModel
        from prompt_optimizer.models import GeneratedPrompt
        
        mock_client = MagicMock()
        mock_client.chat.return_value = GeneratedPrompt(
            prompt="Improved prompt",
            reasoning="Based on history",
        )
        mock_client_cls.return_value = mock_client
        
        mentor = MentorModel(model="test-model", api_key="test-key")
        # New signature: history instead of errors
        result = mentor.generate_prompt(
            history=[],
            current_prompt="Original prompt",
        )
        
        assert result.prompt == "Improved prompt"
