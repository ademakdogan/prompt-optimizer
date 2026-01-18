"""
Integration tests for optimizer module.

These tests use mocked API responses to test the full optimization flow.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prompt_optimizer.core import Evaluator, PromptOptimizer
from prompt_optimizer.models import (
    PIIEntity,
    PIIResponse,
    GeneratedPrompt,
)


class TestOptimizerWithMockedAPI:
    """Tests for PromptOptimizer with mocked API calls."""

    @pytest.fixture
    def mock_agent_response(self) -> PIIResponse:
        """Create a mock agent response."""
        return PIIResponse(
            entities=[
                PIIEntity(value="John", label="FIRSTNAME", start=0, end=4),
                PIIEntity(value="test@email.com", label="EMAIL", start=10, end=24),
            ],
            masked_text="[FIRSTNAME] at [EMAIL]",
        )

    @pytest.fixture
    def mock_ground_truth(self) -> PIIResponse:
        """Create mock ground truth."""
        return PIIResponse(
            entities=[
                PIIEntity(value="John", label="FIRSTNAME", start=0, end=4),
                PIIEntity(value="test@email.com", label="EMAIL", start=10, end=24),
            ],
            masked_text="[FIRSTNAME] at [EMAIL]",
        )

    @pytest.fixture
    def sample_data(self, mock_ground_truth: PIIResponse) -> list[tuple[str, PIIResponse]]:
        """Create sample test data."""
        return [
            ("John at test@email.com", mock_ground_truth),
            ("Contact Jane at jane@test.com", mock_ground_truth),
        ]

    @patch("prompt_optimizer.core.optimizer.AgentModel")
    @patch("prompt_optimizer.core.optimizer.MentorModel")
    def test_optimizer_runs_all_iterations(
        self,
        mock_mentor_cls: MagicMock,
        mock_agent_cls: MagicMock,
        sample_data: list[tuple[str, PIIResponse]],
    ) -> None:
        """Test that optimizer runs configured number of iterations."""
        # Setup mocks - return partial match so optimizer continues
        partial_response = PIIResponse(
            entities=[
                PIIEntity(value="John", label="FIRSTNAME", start=0, end=4),
                # Missing EMAIL - this will cause accuracy < 1.0
            ],
            masked_text="[FIRSTNAME] at email",
        )
        mock_agent = MagicMock()
        mock_agent.process_data.return_value = partial_response
        mock_agent_cls.return_value = mock_agent

        mock_mentor = MagicMock()
        mock_mentor.generate_prompt.return_value = GeneratedPrompt(
            prompt="Improved prompt",
            reasoning="Made improvements",
        )
        mock_mentor_cls.return_value = mock_mentor

        # Create optimizer and run
        optimizer = PromptOptimizer(
            window_size=2,
            loop_count=3,
            api_key="test-key",
        )
        
        results = optimizer.optimize(
            data=sample_data,
            initial_prompt="Extract PII entities",
        )

        # Verify all iterations ran
        assert len(results) == 3
        assert all(r.iteration == i+1 for i, r in enumerate(results))

    @patch("prompt_optimizer.core.optimizer.AgentModel")
    @patch("prompt_optimizer.core.optimizer.MentorModel")
    def test_optimizer_stops_on_perfect_accuracy(
        self,
        mock_mentor_cls: MagicMock,
        mock_agent_cls: MagicMock,
        sample_data: list[tuple[str, PIIResponse]],
        mock_agent_response: PIIResponse,
    ) -> None:
        """Test that optimizer stops early on perfect accuracy."""
        # Setup mocks - agent returns exact match
        mock_agent = MagicMock()
        mock_agent.process_data.return_value = mock_agent_response
        mock_agent_cls.return_value = mock_agent

        mock_mentor = MagicMock()
        mock_mentor_cls.return_value = mock_mentor

        # Create optimizer
        optimizer = PromptOptimizer(
            window_size=2,
            loop_count=5,
            api_key="test-key",
        )
        
        results = optimizer.optimize(
            data=sample_data,
            initial_prompt="Extract PII entities",
        )

        # Should stop after first iteration with perfect accuracy
        assert len(results) == 1
        assert results[0].accuracy == 1.0

    @patch("prompt_optimizer.core.optimizer.AgentModel")
    @patch("prompt_optimizer.core.optimizer.MentorModel")
    def test_optimizer_generates_initial_prompt(
        self,
        mock_mentor_cls: MagicMock,
        mock_agent_cls: MagicMock,
        sample_data: list[tuple[str, PIIResponse]],
        mock_agent_response: PIIResponse,
    ) -> None:
        """Test that optimizer generates initial prompt when none provided."""
        # Setup mocks
        mock_agent = MagicMock()
        mock_agent.process_data.return_value = mock_agent_response
        mock_agent_cls.return_value = mock_agent

        mock_mentor = MagicMock()
        mock_mentor.generate_initial_prompt.return_value = GeneratedPrompt(
            prompt="Generated initial prompt",
            reasoning="Created from sample data",
        )
        mock_mentor.generate_prompt.return_value = GeneratedPrompt(
            prompt="Improved prompt",
            reasoning="Made improvements",
        )
        mock_mentor_cls.return_value = mock_mentor

        # Create optimizer and run without initial prompt
        optimizer = PromptOptimizer(
            window_size=2,
            loop_count=1,
            api_key="test-key",
        )
        
        results = optimizer.optimize(
            data=sample_data,
            initial_prompt=None,  # No initial prompt
        )

        # Verify mentor was called to generate initial prompt
        mock_mentor.generate_initial_prompt.assert_called_once()
        assert results[0].prompt == "Generated initial prompt"

    @patch("prompt_optimizer.core.optimizer.AgentModel")
    @patch("prompt_optimizer.core.optimizer.MentorModel")
    def test_optimizer_respects_window_size(
        self,
        mock_mentor_cls: MagicMock,
        mock_agent_cls: MagicMock,
        sample_data: list[tuple[str, PIIResponse]],
    ) -> None:
        """Test that optimizer respects window size for history."""
        # Setup mocks - agent returns partial match
        partial_response = PIIResponse(
            entities=[
                PIIEntity(value="John", label="FIRSTNAME", start=0, end=4),
            ],
            masked_text="[FIRSTNAME] at email",
        )
        
        mock_agent = MagicMock()
        mock_agent.process_data.return_value = partial_response
        mock_agent_cls.return_value = mock_agent

        mock_mentor = MagicMock()
        mock_mentor.generate_prompt.return_value = GeneratedPrompt(
            prompt="Improved prompt",
            reasoning="Made improvements",
        )
        mock_mentor_cls.return_value = mock_mentor

        # Create optimizer with window_size=2
        optimizer = PromptOptimizer(
            window_size=2,
            loop_count=4,
            api_key="test-key",
        )
        
        optimizer.optimize(
            data=sample_data,
            initial_prompt="Extract PII entities",
        )

        # Check that mentor was called with limited history
        # After iteration 3, history should have only 2 entries
        call_args = mock_mentor.generate_prompt.call_args_list
        if len(call_args) >= 3:
            # On iteration 4, history should have at most 2 items
            last_call = call_args[-1]
            history = last_call[1].get("history", []) if last_call[1] else []
            assert len(history) <= 2


class TestEvaluatorIntegration:
    """Integration tests for Evaluator with realistic data."""

    def test_full_evaluation_workflow(self) -> None:
        """Test complete evaluation workflow."""
        evaluator = Evaluator()
        
        # Create realistic test data
        responses = [
            PIIResponse(
                entities=[
                    PIIEntity(value="Alice", label="FIRSTNAME", start=0, end=5),
                    PIIEntity(value="alice@email.com", label="EMAIL", start=10, end=25),
                ],
                masked_text="[FIRSTNAME] at [EMAIL]",
            ),
            PIIResponse(
                entities=[
                    PIIEntity(value="Bob", label="FIRSTNAME", start=0, end=3),
                ],
                masked_text="[FIRSTNAME] missing email",
            ),
        ]
        
        ground_truths = [
            PIIResponse(
                entities=[
                    PIIEntity(value="Alice", label="FIRSTNAME", start=0, end=5),
                    PIIEntity(value="alice@email.com", label="EMAIL", start=10, end=25),
                ],
                masked_text="[FIRSTNAME] at [EMAIL]",
            ),
            PIIResponse(
                entities=[
                    PIIEntity(value="Bob", label="FIRSTNAME", start=0, end=3),
                    PIIEntity(value="bob@email.com", label="EMAIL", start=7, end=20),
                ],
                masked_text="[FIRSTNAME] at [EMAIL]",
            ),
        ]
        
        # Run evaluation
        accuracy, results = evaluator.evaluate_batch(responses, ground_truths)
        
        # First sample is perfect, second is 50%
        assert accuracy == 0.75
        assert results[0].is_correct is True
        assert results[1].is_correct is False
        assert len(results[1].missing_entities) == 1
        
        # Collect errors
        errors = evaluator.collect_errors(
            responses,
            ground_truths,
            ["Alice at alice@email.com", "Bob at bob@email.com"],
            "Extract PII",
        )
        
        # Should have one unique error type (missing EMAIL)
        assert len(errors) == 1
        assert errors[0].field_name == "EMAIL"
