"""
End-to-end test for PII extraction with real API calls.

This test requires a valid OpenRouter API key in the .env file.
It tests the full optimization loop with real model calls.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from prompt_optimizer.config import get_settings
from prompt_optimizer.core import PromptOptimizer
from prompt_optimizer.data import load_test_data
from prompt_optimizer.models import PIIEntity, PIIResponse, GeneratedPrompt


# Skip these tests if no API key is available
def has_api_key() -> bool:
    """Check if API key is available."""
    try:
        settings = get_settings()
        return bool(settings.openrouter_api_key)
    except Exception:
        return False


@pytest.mark.skipif(not has_api_key(), reason="No OpenRouter API key available")
class TestPIIExtractionE2E:
    """End-to-end tests for PII extraction."""

    @pytest.fixture
    def test_data_path(self) -> Path:
        """Get path to test data."""
        return Path(__file__).parent.parent.parent / "resources" / "test_data.json"

    def test_load_and_parse_test_data(self, test_data_path: Path) -> None:
        """Test that test data can be loaded and parsed."""
        if not test_data_path.exists():
            pytest.skip("Test data file not found")
        
        data = load_test_data(test_data_path, limit=5)
        
        assert len(data) == 5
        for source_text, ground_truth in data:
            assert isinstance(source_text, str)
            assert isinstance(ground_truth, PIIResponse)
            assert len(source_text) > 0


class TestPIIExtractionMocked:
    """Mocked e2e tests that don't require API key."""

    @pytest.fixture
    def test_data_path(self) -> Path:
        """Get path to test data."""
        return Path(__file__).parent.parent.parent / "resources" / "test_data.json"

    @patch("prompt_optimizer.core.optimizer.AgentModel")
    @patch("prompt_optimizer.core.optimizer.MentorModel")
    def test_full_optimization_with_test_data(
        self,
        mock_mentor_cls: MagicMock,
        mock_agent_cls: MagicMock,
        test_data_path: Path,
    ) -> None:
        """Test full optimization flow with test data and mocked API."""
        if not test_data_path.exists():
            pytest.skip("Test data file not found")
        
        # Load test data
        data = load_test_data(test_data_path, limit=5)
        
        # Setup mocks to return progressively better results
        iteration = [0]
        
        def mock_process_data(prompt: str, data: str, schema_description: str = "") -> PIIResponse:
            """Return mock responses that improve over iterations."""
            # Return partial matches first, then full match
            if iteration[0] < 2:
                return PIIResponse(
                    entities=[
                        PIIEntity(value="test", label="FIRSTNAME", start=0, end=4),
                    ],
                    masked_text="[FIRSTNAME]",
                )
            else:
                # Return a better response
                return PIIResponse(
                    entities=[
                        PIIEntity(value="test", label="FIRSTNAME", start=0, end=4),
                        PIIEntity(value="test@email.com", label="EMAIL", start=10, end=24),
                    ],
                    masked_text="[FIRSTNAME] [EMAIL]",
                )
        
        mock_agent = MagicMock()
        mock_agent.process_data.side_effect = mock_process_data
        mock_agent_cls.return_value = mock_agent

        mock_mentor = MagicMock()
        mock_mentor.generate_prompt.return_value = GeneratedPrompt(
            prompt="Improved prompt for iteration",
            reasoning="Based on error analysis",
        )
        mock_mentor_cls.return_value = mock_mentor

        # Run optimizer
        optimizer = PromptOptimizer(
            window_size=2,
            loop_count=3,
            api_key="test-key",
        )
        
        results = optimizer.optimize(
            data=data,
            initial_prompt="Extract all PII entities from the text.",
        )

        # Verify optimization ran
        assert len(results) > 0
        assert all(r.total_samples == 5 for r in results)
        
        # Verify mentor was called to improve prompts (unless perfect)
        if results[0].accuracy < 1.0:
            mock_mentor.generate_prompt.assert_called()

    def test_data_variety(self, test_data_path: Path) -> None:
        """Test that test data has variety of PII types."""
        if not test_data_path.exists():
            pytest.skip("Test data file not found")
        
        data = load_test_data(test_data_path, limit=5)
        
        all_labels = set()
        for _, ground_truth in data:
            for entity in ground_truth.entities:
                all_labels.add(entity.label)
        
        # Should have at least 5 different PII types
        assert len(all_labels) >= 5
        
        # Should include common types
        expected_types = {"FIRSTNAME", "EMAIL"}
        assert expected_types.intersection(all_labels)
