"""
End-to-end tests for data extraction flow.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prompt_optimizer.data import load_test_data
from prompt_optimizer.models import GeneratedPrompt


# Path to test data relative to project root
TEST_DATA_PATH = Path(__file__).parent.parent.parent / "resources" / "test_data.json"


class TestDataExtractionE2E:
    """End-to-end tests for data extraction workflow."""

    @pytest.mark.skipif(
        not TEST_DATA_PATH.exists(),
        reason="Test data file not found",
    )
    def test_load_and_parse_test_data(self) -> None:
        """Test loading and parsing the test data file."""
        data = load_test_data(TEST_DATA_PATH, limit=5)
        
        assert len(data) == 5
        
        for source_text, ground_truth in data:
            assert isinstance(source_text, str)
            assert len(source_text) > 0
            assert isinstance(ground_truth, dict)

    @pytest.mark.skipif(
        not TEST_DATA_PATH.exists(),
        reason="Test data file not found",
    )
    def test_data_variety(self) -> None:
        """Test that test data covers various field types."""
        data = load_test_data(TEST_DATA_PATH, limit=5)
        
        all_fields = set()
        for _, ground_truth in data:
            for field in ground_truth.keys():
                all_fields.add(field)
        
        # Should have variety of field types
        assert len(all_fields) >= 5


class TestDataExtractionMocked:
    """E2E tests with mocked API for fast, reliable testing."""

    @patch("prompt_optimizer.core.optimizer.MentorModel")
    @patch("prompt_optimizer.core.optimizer.AgentModel")
    def test_full_optimization_flow_mocked(
        self,
        mock_agent_cls: MagicMock,
        mock_mentor_cls: MagicMock,
    ) -> None:
        """Test full optimization flow with mocked API."""
        from prompt_optimizer.core.optimizer import PromptOptimizer
        
        # Setup mocks
        mock_agent = MagicMock()
        mock_agent.update_field_descriptions = MagicMock()
        
        # Simulate improving accuracy over iterations
        responses = [
            {},  # First iteration: empty
            {"firstname": "John"},  # Second: partial
            {"firstname": "John", "email": "test@example.com"},  # Third: perfect
        ]
        mock_agent.process_data.side_effect = responses
        mock_agent_cls.return_value = mock_agent
        
        mock_mentor = MagicMock()
        mock_mentor.generate_initial_prompt.return_value = GeneratedPrompt(
            prompt="Extract key information from text",
            reasoning="Initial prompt for extraction",
            field_descriptions={"firstname": "A person's first name"},
        )
        mock_mentor.generate_prompt.return_value = GeneratedPrompt(
            prompt="Improved: Look for names and emails",
            reasoning="Added email detection",
            field_descriptions={"email": "Email address in user@domain format"},
        )
        mock_mentor_cls.return_value = mock_mentor
        
        optimizer = PromptOptimizer(loop_count=3, window_size=2)
        
        data = [
            ("John can be reached at test@example.com", 
             {"firstname": "John", "email": "test@example.com"}),
        ]
        
        results = optimizer.optimize(data)
        
        # Should run iterations
        assert len(results) >= 1
        
        # All results should have required fields
        for result in results:
            assert hasattr(result, "iteration")
            assert hasattr(result, "accuracy")
            assert hasattr(result, "prompt")
